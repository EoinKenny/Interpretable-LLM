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

from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


# Constants
TEXT_BATCH_SIZE = 16
TEXT_LEN = 128
LR = 1e-1
MIN_LR = 1e-5
LR_STEP_RATE = 2
LATENT_SIZE = 2304
SAE_SCALING_FACTOR = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = 'SlimPajama-627B'
TRACK_WINDOW = 1000000
K = 10
K_AUX = 512
ALPHA = 1. / 32  # Scaling factor for auxiliary loss


def main():
    access_token='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'
    model_type = 'google/gemma-2-2b-it'
    
    tokenizer = AutoTokenizer.from_pretrained(model_type,
                                              token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_type,
                                                device_map='auto',
                                                token=access_token,
                                                torch_dtype=torch.float16)
    
    # Initialize SAE and optimizer
    sae = SparseAutoencoder(n_latents=int(SAE_SCALING_FACTOR*LATENT_SIZE), n_inputs=LATENT_SIZE, k=K, k_aux=K_AUX, track_window=TRACK_WINDOW)
    sae.to(DEVICE)
    # Don't convert to half precision here - keep in float32
    optimizer = optim.Adam(sae.parameters(), lr=LR)
    lr_decay_factor = 0.9
    
    # Initialize gradient scaler
    scaler = GradScaler()

    # Setup tracking
    sparsity_loss_data = []
    reconstruction_loss_data = []
    count = 0
    file_count = 0
    start_time = time.time()

    root_data_dir = os.path.join(DATA_DIR, 'train')

    times = []

    for sub_dir in os.listdir(root_data_dir):
        sub_dir_path = os.path.join(root_data_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue

        for zst_file in os.listdir(sub_dir_path):
            if not zst_file.endswith('.zst'):
                continue
                
            file_path = os.path.join(sub_dir_path, zst_file)
            df = load_data(file_path)
            text_data = df.text.values.tolist()
            num_text_batches = len(text_data) // TEXT_BATCH_SIZE

            print(f"Processing file {file_count + 1}")
            print(f"DF Shape: {df.shape}, Text batches: {num_text_batches}")

            for text_batch_idx in range(num_text_batches):
                start_time = time.time()
                
                text_batch = text_data[text_batch_idx * TEXT_BATCH_SIZE: 
                                     (text_batch_idx + 1) * TEXT_BATCH_SIZE]
                
                with torch.no_grad():                        
                    residules, _ = get_residules(model, text_batch, tokenizer, 20)
                
                residules = residules[:, 1:, :]
                residules = residules.reshape(-1, LATENT_SIZE)
                residules = residules.to(DEVICE)  # Keep in FP32
                
                # Use autocast for forward pass
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    recon, encoded = sae(residules)
                    loss_r = L2Loss(recon, residules)
                    loss_s = sae.aux_reconstruction(residules, recon) * ALPHA
                    loss = loss_s + loss_r

                if torch.isnan(loss):
                    print("NaN detected in loss!")
                    print(count)
                    print(f"Reconstruction loss: {loss_r}, Sparsity loss: {loss_s}")
                    continue

                # Scale loss and do backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Track losses
                sparsity_loss = loss_s.item()
                reconstruction_loss = loss_r.item()
                
                if not (torch.isnan(torch.tensor(sparsity_loss)) or torch.isnan(torch.tensor(reconstruction_loss))):
                    sparsity_loss_data.append(sparsity_loss)
                    reconstruction_loss_data.append(reconstruction_loss)
                    
                    if count % 10 == 0:
                        plot_loss(sparsity_loss_data, reconstruction_loss_data)
                
                count += 1
                times.append(time.time() - start_time  )
                # print("Current batch:", count, " -- Avg Time:", sum(times)/len(times)  )
                if count % 1000 == 0:
                    torch.save(sae.state_dict(), f'weights/sae_{count}.pth')
            

            file_count += 1

            if file_count % LR_STEP_RATE == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(
                        param_group['lr'] * lr_decay_factor, 
                        MIN_LR
                    )
                print(f"Learning rate updated to {optimizer.param_groups[0]['lr']}")

            print(f"\nSparsity Loss: {sum(sparsity_loss_data[-50:]) / 50:.2f}")
            print(f"Reconstruction Loss: {sum(reconstruction_loss_data[-50:]) / 50:.2f}")
            print(f"Iteration: {count}, File: {file_path}")
            print(f"Time Elapsed: {time.time() - start_time:.2f}s")



class SparseAutoencoder(nn.Module):
    
    def __init__(self, n_latents: int, n_inputs: int, k: int, k_aux: int, track_window: int = 10000) -> None:
        super().__init__()
        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.k = k
        self.k_aux = k_aux
        self.latents_org = None

        # Initialize decoder weights as transpose of encoder weights
        self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.t()

        # Track how long neurons have been inactive
        self.track_window = track_window
        self.activation_tracker = torch.zeros(n_latents, dtype=torch.long)  # Tracks inactivity for each neuron

    def aux_reconstruction(self, x: torch.Tensor, recons: torch.Tensor) -> torch.Tensor:
        """
        Compute the auxiliary loss using the top-k_aux dead latents.
        A neuron is considered dead if it hasn't activated for self.track_window iterations.
        """

        # Identify neurons that haven't activated for self.track_window iterations
        dead_neurons = (self.activation_tracker >= self.track_window).nonzero(as_tuple=True)[0]

        if dead_neurons.numel() == 0:
            # If no dead neurons, return a loss of zero
            return torch.tensor(0.0, device=x.device)

        # Select the top-k_aux inactive neurons
        topk_aux_neurons = dead_neurons[:self.k_aux] if dead_neurons.numel() > self.k_aux else dead_neurons

        # Keep only the activations of these top-k_aux dead neurons in the latent vectors
        dead_latents = torch.zeros_like(self.latents_org)
        dead_latents[:, topk_aux_neurons] = self.latents_org[:, topk_aux_neurons]  # Retain only the selected dead neuron activations
        dead_latents[:, topk_aux_neurons] = torch.exp(dead_latents[:, topk_aux_neurons])    

        # Reconstruct using only the selected dead latents
        dead_reconstruction = self.decode(dead_latents)

        # Compute the main reconstruction error e = x - recons
        error = x - recons

        # Auxiliary reconstruction error using dead latents e_hat = dead_reconstruction
        aux_error = error - dead_reconstruction

        # Auxiliary loss: L_aux = ||e - e_hat||^2_2
        aux_loss = torch.norm(aux_error, p=2) ** 2

        # Ensure numerical stability by zeroing NaNs (if any)
        if torch.isnan(aux_loss):
            aux_loss = torch.tensor(0.0, device=x.device)

        return aux_loss


    def topk_activation(self, latents_org: torch.Tensor) -> torch.Tensor:
        """
        Get the top k activations in the latent vector
        
        Also keep track of which neurons are firing or not
        """
        self.latents_org = latents_org.clone().detach()
        latents = F.relu(latents_org)
        topk = torch.topk(latents, self.k, dim=-1)
        result = torch.zeros_like(latents)
        result.scatter_(-1, topk.indices, topk.values)
        with torch.no_grad():
            self.activation_tracker += 1
            non_zero_activations = torch.sum(result > 0, dim=0)
            activated_idxs = non_zero_activations > 0
            self.activation_tracker[activated_idxs==1] = 0
        return result

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        latents = self.encoder(x) + self.latent_bias
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)  
        latents_topk = self.topk_activation(latents)
        recons = self.decode(latents_topk)
        return recons, latents

def get_residules(model, sentences, tokenizer, layer_num):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=TEXT_LEN)
    
    activations = []
    
    def get_activations(layer, input, output):
        activations.append(output)
    
    layer = model.model.layers[layer_num]
    hook = layer.register_forward_hook(get_activations)
    
    with torch.no_grad():
        outputs = model(**inputs.to(model.device), use_cache=False)
    
    hook.remove()

    logits = outputs[0]
    residules = activations[0][0]
    
    return residules, logits


def L1Loss(x):
    return torch.mean(torch.abs(x))


def L2Loss(x, y):
    return torch.mean((x - y) ** 2)


def plot_loss(sparsity_losses, reconstruction_losses, save_path='plots/loss_plot.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(sparsity_losses, label='Sparsity Loss', alpha=0.7)
    plt.plot(reconstruction_losses, label='Reconstruction Loss', alpha=0.7)
    plt.yscale('log')  # Set y-axis to log scale
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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


def setup_directories():
    directories = ['gifs', 'plots', 'weights']
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
        print(f"Created directory: {directory}")


if __name__ == '__main__':
    setup_directories()
    main()
    
