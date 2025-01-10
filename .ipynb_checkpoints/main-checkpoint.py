import io
import os
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import shutil
import zstandard as zst
import torch.nn.functional as F
import numpy as np


from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from datasets import load_dataset


# Constants
TEXT_BATCH_SIZE = 4
TEXT_LEN = 128 * 8
LR = 1e-2
MIN_LR = 1e-6
LR_STEP_RATE = 2
LATENT_SIZE = 2304
SAE_LATENT_SIZE = 16384
PROTOTYPE_SIZE = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
access_token='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'
model_type = 'google/gemma-2-2b'
DATA_DIR = 'pile-uncopyrighted/train/'
num_latents_considered = 10
lr_decay_factor = 0.9
mlp_complexity = 8



tokenizer = AutoTokenizer.from_pretrained(model_type,
                                          token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_type,
                                            device_map='auto',
                                            token=access_token,
                                            torch_dtype=torch.float16)

sae, cfg_dict, sparsity = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical",
                                              sae_id="layer_20/width_16k/canonical",
                                              device=DEVICE)


def get_residules(model, sentences, tokenizer, layer_num):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=TEXT_LEN)
    activations = []
    
    def get_activations(layer, input, output):
        activations.append(output)
    
    layer = model.model.layers[layer_num]
    hook  = layer.register_forward_hook(get_activations)
    
    with torch.no_grad():
        outputs = model(**inputs.to(model.device), use_cache=False)
    
    hook.remove()
    logits = outputs[0]
    residules = activations[0][0]
    return residules, logits

def load_data(compressed_file_path) -> pd.DataFrame:
    def read_jsonl_zst(file_path):
        with open(file_path, 'rb') as file:
            decompressor = zst.ZstdDecompressor()
            stream_reader = decompressor.stream_reader(file)
            stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
            for line in stream:
                yield json.loads(line)
    data = list(read_jsonl_zst(compressed_file_path))
    return pd.DataFrame(data)


def plot_loss_and_sparsity(
    loss_data, avg_sparsity_proto, avg_sparsity_sae,
    save_path_loss='plots/loss_plot.png',
    save_path_sparsity_diff='plots/sparsity_diff_plot.png'
    ):
    # Plotting the loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_data, label='MSE Loss', alpha=0.7)
    plt.yscale('symlog')  # Log scale for loss
    plt.axhline(0, color='black', linestyle='--', alpha=0.7)  # Add a reference line at 0
    plt.xlabel('Iteration')
    plt.ylabel('Loss (symlog scale)')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path_loss)
    plt.close()

    # Calculating and plotting the difference in sparsity
    sparsity_diff = np.array(avg_sparsity_proto) - np.array(avg_sparsity_sae)
    plt.figure(figsize=(10, 5))
    plt.plot(sparsity_diff, label='Sparsity Difference (Prototype - SAE)', alpha=0.7, color='green')
    plt.yscale('symlog')  # Log scale for loss
    plt.axhline(0, color='black', linestyle='--', alpha=0.7)  # Add a reference line at 0
    plt.xlabel('Iteration')
    plt.ylabel('Sparsity Difference (symlog)')
    plt.title('Difference in Sparsity Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path_sparsity_diff)
    plt.close()


def activation_jump_relu(x, sae):
    return sae.activation_fn(x) * (x > sae.threshold)


class ProtoSAE(nn.Module):

    def __init__(self, n_latents: int, n_inputs: int, prototype_dim: int) -> None:
        super().__init__()
        
        self.encoders = nn.ModuleList(
            [self._build_mlp(n_inputs, prototype_dim, n_inputs // mlp_complexity) for _ in range(n_latents)]
        )
        self.prototype_dim = prototype_dim
        self.n_latents = n_latents
        self.prototypes = nn.Parameter(torch.zeros(n_latents, prototype_dim))
        self.epsilon = 1e-4  # A small value to avoid division by zero

    def _build_mlp(self, input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # First layer
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Second layer
        )
    
    def _proto_layer(self, x):
        
        b_size, seq_len, n_latents, prototype_dim = x.shape
    
        # Expand prototypes for batch processing
        c = self.prototypes.unsqueeze(0)  # Shape: (1, n_latents, prototype_dim)
        x = x.view(b_size * seq_len, n_latents, prototype_dim)  # Flatten sequence
    
        # Calculate L2 distances
        l2s = ((x - c) ** 2).sum(dim=-1)  # Shape: (b_size*seq_len, n_latents)
    
        # Calculate activations
        raw_act = torch.log((l2s + 1.) / (l2s + self.epsilon))  # Raw activations
    
        # Scale activations to range [-100, 200]
        max_possible_act = torch.log(torch.tensor(1.0 / self.epsilon))
        normalized_act = raw_act / max_possible_act  # Normalize to [0, 1]
        scaled_act = normalized_act * 300 - 100  # Scale to [-100, 200]
    
        return scaled_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
        x is tensor of shape (batch_size, sequence_len, n_inputs)
        """        
        # Encode the input using all encoders
        encoded = [encoder(x.view(-1, x.shape[-1])) for encoder in self.encoders]
        encoded = torch.stack(encoded, dim=1)  
        encoded = encoded.view(x.shape[0], x.shape[1], self.n_latents, self.prototype_dim)
        
        # Pass through the prototype layer
        proto_activations = self._proto_layer(encoded) 

        return proto_activations.view(x.shape[0], x.shape[1], self.n_latents)




## Training
# df = load_data('pile-uncopyrighted/train/00.jsonl.zst')
df = load_data('SlimPajama-627B/train/chunk1/example_train_0.jsonl.zst')
proto_sae = ProtoSAE(n_latents=num_latents_considered, n_inputs=LATENT_SIZE, prototype_dim=PROTOTYPE_SIZE)
proto_sae.to(DEVICE)

# Don't convert to half precision here - keep in float32
optimizer = optim.Adam(proto_sae.parameters(), lr=LR)
mse_loss = nn.MSELoss()

# Initialize gradient scaler
scaler = GradScaler()

# Setup tracking
loss_data = []
avg_sparisty_sae = list()
avg_sparsity_proto = list()

count = 0
file_count = 1
start_time = time.time()
root_data_dir = os.path.join(DATA_DIR, 'train')
times = []
    
text_data = df.text.values.tolist()
num_text_batches = len(text_data) // TEXT_BATCH_SIZE

print(f"Processing file {file_count + 1}")
print(f"DF Shape: {df.shape}, Text batches: {num_text_batches}")
print('Current file:', file_count)

for text_batch_idx in range(num_text_batches):
    
    start_time = time.time()
    text_batch = text_data[text_batch_idx * TEXT_BATCH_SIZE: 
                         (text_batch_idx + 1) * TEXT_BATCH_SIZE]
    
    with torch.no_grad():                        
        residules, _ = get_residules(model, text_batch, tokenizer, 20)
        residules = residules[:, 1:, :].to(DEVICE)  # Keep in FP32
        sae_labels = residules.float() @ sae.W_enc + sae.b_enc  # hidden pre-activation
        sae_labels = sae_labels[:, :, :num_latents_considered]

    # Use autocast for forward pass
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        latents = proto_sae(residules)        
        loss = mse_loss(latents, sae_labels)

    # Scale loss and do backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    sae_sparsity = (((sae.activation_fn(sae_labels) * (sae_labels > sae.threshold[:num_latents_considered])) > 0.).sum(-1).flatten()*1.).mean().item()
    proto_sparsity = (((sae.activation_fn(latents) * (latents > sae.threshold[:num_latents_considered])) > 0.).sum(-1).flatten()*1.).mean().item()
    
    # if not torch.isnan(torch.tensor(loss)):
    loss_data.append(loss.item())
    avg_sparisty_sae.append(sae_sparsity)
    avg_sparsity_proto.append(proto_sparsity)
        
    if count % 10 == 0:
        plot_loss_and_sparsity(loss_data, avg_sparsity_proto, avg_sparisty_sae)
    
    count += 1
    times.append(time.time() - start_time  )
    
    if count % 100 == 0:
        # plt.hist(sae_labels.flatten().cpu())
        # plt.hist(latents.flatten().detach().cpu())
        # plt.show()
        torch.save(proto_sae.state_dict(), f'weights/sae_{count}.pth')
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(
                param_group['lr'] * lr_decay_factor, 
                MIN_LR
            )
        print(f"Learning rate updated to {optimizer.param_groups[0]['lr']}")
        print("Current batch:", count, " -- Avg Time:", sum(times)/len(times)  )


