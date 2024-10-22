import io
import os
import pandas as pd
import zstandard as zst
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import shutil

from torch.cuda.amp import autocast
from torch.nn.parallel import DataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer


# Constants
TEXT_BATCH_SIZE = 512
TEXT_LEN = 128  # how much of each text document to use in training (just first 128 tokens here)
LR = 1e-1
L2 = 1e-8
MIN_LR = 1e-5
LR_STEP_RATE = 2
LATENT_SIZE = 2048
DEVICE ='cpu'# 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = 'SlimPajama-627B'
LAMBDA = 1.0  # sparsity loss scaling factor


def main():
    # Initialize model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tiny_llama = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize SAE and optimizer
    sae = SparseAutoEncoder()
    optimizer = optim.Adam(sae.parameters(), lr=LR, weight_decay=L2)
    lr_decay_factor = 0.9

    # Setup tracking
    sparsity_loss_data = []
    reconstruction_loss_data = []
    count = 0
    file_count = 0
    start_time = time.time()

    root_data_dir = os.path.join(DATA_DIR, 'train')

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
                text_batch = text_data[text_batch_idx * TEXT_BATCH_SIZE: 
                                     (text_batch_idx + 1) * TEXT_BATCH_SIZE]

                with torch.autocast(device_type=DEVICE):
                    with torch.no_grad():
                        input_ids = tokenizer(
                            text_batch, 
                            max_length=TEXT_LEN, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True
                        ).input_ids
                        
                        # Get residual stream activations
                        z = get_residual(tiny_llama, input_ids, layer_num=12)

                    # Forward pass through SAE
                    recon = sae(z)
                    breakpoint()
                    
                    # Compute losses
                    loss_r = L2Loss(recon, z)
                    loss_s = L1Loss(sae.encoder(z)) * LAMBDA
                    loss = loss_s + loss_r

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Track losses
                sparsity_loss_data.append(loss_s.item())
                reconstruction_loss_data.append(loss_r.item())
                
                if count % 10 == 0:  # Plot every 10 iterations
                    plot_loss(sparsity_loss_data, reconstruction_loss_data)
                
                count += 1

            file_count += 1

            # Learning rate decay
            if file_count % LR_STEP_RATE == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(
                        param_group['lr'] * lr_decay_factor, 
                        MIN_LR
                    )
                print(f"Learning rate updated to {optimizer.param_groups[0]['lr']}")

            # Print metrics
            print(f"\nSparsity Loss: {sum(sparsity_loss_data[-50:]) / 50:.2f}")
            print(f"Reconstruction Loss: {sum(reconstruction_loss_data[-50:]) / 50:.2f}")
            print(f"Iteration: {count}, File: {file_path}")
            print(f"Time Elapsed: {time.time() - start_time:.2f}s")

            # Save checkpoint
            if count % 1000 == 0:
                torch.save(sae.state_dict(), f'weights/sae_{count}.pth')


class SparseAutoEncoder(nn.Module):
    def __init__(self, input_size=2048, hidden_size=LATENT_SIZE):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_size, input_size)
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.encoder[0].weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        
        self.to(DEVICE)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_residual(model, input_ids, layer_num=12):
    """
    Extract residual stream activations from the specified layer using hooks.
    """
    
    residual_activations = []

    def hook_fn(module, input, output):
        residual_activations.append(output)

    # Ensure the model's parameters are on the correct device (cuda:0 or cuda:1, etc.)
    primary_device = next(model.parameters()).device

    # Register the hook on the correct layer
    hook_handle = model.model.layers[layer_num].register_forward_hook(hook_fn)

    # Move input_ids to the primary device
    input_ids = input_ids.to(primary_device)

    # Forward pass
    with torch.no_grad():
        model(input_ids)

    # Remove hook to avoid memory issues
    hook_handle.remove()

    # Extract the residual activations and move to the primary device
    hidden_states = residual_activations[0].to(primary_device).reshape(-1, residual_activations[0].size(-1))

    return hidden_states




def L1Loss(x):
    """Compute L1 loss for sparsity."""
    return torch.mean(torch.abs(x))


def L2Loss(x, y):
    """Compute L2 loss for reconstruction."""
    return torch.mean((x - y) ** 2)


def plot_loss(sparsity_losses, reconstruction_losses, save_path='plots/loss_plot.png'):
    """Plot training losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(sparsity_losses, label='Sparsity Loss', alpha=0.7)
    plt.plot(reconstruction_losses, label='Reconstruction Loss', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
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
