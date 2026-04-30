"""
Train Standard TopK SAE and ProtoSAE side-by-side on chunked activations.

Loads activation chunks produced by collect_activations.py, trains both a
TopK SAE and a ProtoSAE (with AuxK auxiliary loss for dead-feature revival),
evaluates each on held-out chunks, and writes a comparison plot.

Usage:
    python train_sae.py --k 128 --expansion_factor 4
    python train_sae.py --num_epochs_std 1 --num_epochs_proto 10
    python train_sae.py --quick   # 1 epoch x 5 batches each, smoke test
"""

import os
import json
import random
import gc
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from typing import List, Dict
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from src.topk_sae import TopK, Autoencoder
from src.proto_sae import ProtoAutoencoder


# ============================================================================
#  Utilities
# ============================================================================

def clean_gpus() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def load_chunk(chunks_dir: str, chunk_file: str) -> torch.Tensor:
    path = os.path.join(chunks_dir, chunk_file)
    chunk_data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(chunk_data, dict):
        for key in ["final_token_activations", "activations"]:
            if key in chunk_data:
                return chunk_data[key].float()
        raise KeyError(f"Unknown chunk format. Keys: {list(chunk_data.keys())}")
    return chunk_data.float()


def compute_neuron_activation_metrics(
    activation_counts: torch.Tensor,
) -> Dict[str, float]:
    n_neurons = activation_counts.numel()
    counts = activation_counts.float().cpu().numpy()
    pct_active = (counts > 0).mean() * 100.0
    if counts.sum() == 0:
        return {"activation_entropy": 0.0, "pct_active": 0.0, "gini_coefficient": 1.0}
    probs = counts / counts.sum()
    active_probs = probs[probs > 0]
    entropy = -(active_probs * np.log(active_probs + 1e-12)).sum()
    norm_entropy = float(entropy / np.log(n_neurons))
    sorted_counts = np.sort(counts)
    cum = np.cumsum(sorted_counts)
    gini = (
        2 * (np.arange(1, n_neurons + 1) * sorted_counts).sum()
    ) / (n_neurons * cum[-1]) - (n_neurons + 1) / n_neurons
    gini = float(np.clip(gini, 0.0, 1.0))
    return {
        "activation_entropy": norm_entropy,
        "pct_active": pct_active,
        "gini_coefficient": gini,
    }


# ============================================================================
#  Dead latent tracker
# ============================================================================

class DeadLatentTracker:
    def __init__(self, n_latents: int, dead_threshold: int = 10_000, device: str = "cuda"):
        self.n_latents = n_latents
        self.dead_threshold = dead_threshold
        self.device = device
        self.steps_since_activation = torch.zeros(
            n_latents, dtype=torch.long, device=device
        )
        self.total_steps = 0

    def update(self, latents: torch.Tensor) -> None:
        batch_active = (latents > 0).any(dim=0)
        self.steps_since_activation = torch.where(
            batch_active,
            torch.zeros_like(self.steps_since_activation),
            self.steps_since_activation + latents.size(0),
        )
        self.total_steps += 1

    def get_dead_latents(self) -> torch.Tensor:
        return self.steps_since_activation >= self.dead_threshold

    def get_dead_fraction(self) -> float:
        if self.total_steps < self.dead_threshold:
            return 0.0
        return float(self.get_dead_latents().float().mean().item())


# ============================================================================
#  Auxiliary loss (AuxK)
# ============================================================================

def compute_auxiliary_loss(
    sae,
    x: torch.Tensor,
    recon: torch.Tensor,
    tracker: DeadLatentTracker,
    k_aux: int = 512,
    alpha: float = 1 / 32,
) -> torch.Tensor:
    """
    AuxK loss: train currently-dead latents to explain the main model's
    reconstruction residual. Gives dead features gradient signal so they
    can come back to life.
    """
    dead_mask = tracker.get_dead_latents()
    if dead_mask.sum() < k_aux:
        return torch.tensor(0.0, device=x.device, requires_grad=True)

    lat_pre = sae.encode_pre_act(x)
    dead_only = lat_pre.clone()
    dead_only[:, ~dead_mask] = 0.0

    dead_idx = torch.nonzero(dead_mask, as_tuple=True)[0]
    dead_vals = dead_only[:, dead_idx]
    _, topk_in_dead = torch.topk(dead_vals.abs(), k_aux, dim=1)

    B = x.shape[0]
    batch_idx = torch.arange(B, device=x.device).unsqueeze(1)
    chosen_dead = dead_idx[topk_in_dead]

    aux_pre = torch.zeros_like(lat_pre)
    aux_pre[batch_idx, chosen_dead] = lat_pre[batch_idx, chosen_dead]

    # Standard SAE uses exp on the dead-latent pre-activations; ProtoSAE
    # already has monotonic similarities, so it overrides this hook.
    if hasattr(sae, "aux_dead_activation"):
        aux_act = sae.aux_dead_activation(aux_pre)
    else:
        aux_act = torch.exp(aux_pre)
    e_hat = sae.decoder(aux_act)

    aux_mse = F.mse_loss(x - recon, e_hat)
    return alpha * aux_mse


# ============================================================================
#  Training loop
# ============================================================================

def train_sae_chunked(
    sae,
    chunks_dir: str,
    train_chunk_files: List[str],
    *,
    num_epochs: int = 1,
    tokens_per_batch: int = 2048,
    n_batches: int = -1,
    lr: float = 3e-4,
    aux_coef: float = 1 / 32,
    k_aux: int = 512,
    dead_threshold: int = 1_000_000,
    device: str = "cuda",
    normalize_decoder: bool = True,
):
    optimizer = optim.Adam(sae.parameters(), lr=lr)
    scaler = GradScaler()
    tracker = DeadLatentTracker(sae.n_latents, dead_threshold, device)

    metrics = {
        "batch_recon_losses": [],
        "batch_fvu": [],
        "batch_l0": [],
        "batch_aux_losses": [],
        "chunk_activation_entropy": [],
        "chunk_active_neurons_pct": [],
        "chunk_gini_coefficient": [],
        "chunk_dead_frac": [],
        "chunk_mean_recon": [],
        "chunk_mean_fvu": [],
    }

    sae.train()
    total_batches_done = 0
    hit_limit = False

    if n_batches > 0:
        logging.getLogger("sae_train").info(f"  Training limited to {n_batches} batches")

    for epoch in range(num_epochs):
        if hit_limit:
            break
        random.shuffle(train_chunk_files)
        chunk_pbar = tqdm(train_chunk_files, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for chunk_file in chunk_pbar:
            if hit_limit:
                break
            acts = load_chunk(chunks_dir, chunk_file)
            loader = DataLoader(
                TensorDataset(acts),
                batch_size=tokens_per_batch,
                shuffle=True,
                pin_memory=device.startswith("cuda"),
            )

            chunk_act_counts = torch.zeros(
                sae.n_latents, dtype=torch.long, device=device
            )
            chunk_recon, chunk_aux, chunk_fvu, chunk_l0 = [], [], [], []

            for (tokens,) in loader:
                if n_batches > 0 and total_batches_done >= n_batches:
                    hit_limit = True
                    break

                x = tokens.to(device)

                optimizer.zero_grad()
                with autocast():
                    lat_pre, lat, recon = sae(x)
                    tracker.update(lat)
                    chunk_act_counts += (lat > 0).sum(dim=0)

                    recon_loss = F.mse_loss(recon, x)
                    aux_loss = compute_auxiliary_loss(
                        sae, x, recon, tracker, k_aux, aux_coef
                    )
                    loss = recon_loss + aux_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Unit-norm decoder columns after each step
                if normalize_decoder and hasattr(sae, "normalize") and sae.normalize:
                    with torch.no_grad():
                        sae.decoder.weight.data = F.normalize(
                            sae.decoder.weight.data, dim=0
                        )

                # Per-batch metrics
                with torch.no_grad():
                    x_var = (x - x.mean(dim=0)).pow(2).sum()
                    resid = (x - recon).pow(2).sum()
                    fvu = (resid / (x_var + 1e-8)).item()
                    l0 = (lat > 0).float().sum(dim=-1).mean().item()

                metrics["batch_recon_losses"].append(recon_loss.item())
                metrics["batch_fvu"].append(fvu)
                metrics["batch_l0"].append(l0)
                metrics["batch_aux_losses"].append(aux_loss.item())
                chunk_recon.append(recon_loss.item())
                chunk_fvu.append(fvu)
                chunk_l0.append(l0)
                chunk_aux.append(aux_loss.item())

                total_batches_done += 1

            if not chunk_recon:
                continue

            # Chunk-level summary
            avg_recon = np.mean(chunk_recon)
            avg_fvu = np.mean(chunk_fvu)
            avg_l0 = np.mean(chunk_l0)
            dead_frac = tracker.get_dead_fraction() * 100.0

            chunk_pbar.set_postfix({
                "batch": f"{total_batches_done}",
                "recon": f"{avg_recon:.3e}",
                "fvu": f"{avg_fvu:.4f}",
                "L0": f"{avg_l0:.0f}",
                "dead%": f"{dead_frac:.1f}",
            })

            cm = compute_neuron_activation_metrics(chunk_act_counts)
            metrics["chunk_activation_entropy"].append(cm["activation_entropy"])
            metrics["chunk_active_neurons_pct"].append(cm["pct_active"])
            metrics["chunk_gini_coefficient"].append(cm["gini_coefficient"])
            metrics["chunk_dead_frac"].append(dead_frac)
            metrics["chunk_mean_recon"].append(avg_recon)
            metrics["chunk_mean_fvu"].append(avg_fvu)

            del acts
            torch.cuda.empty_cache()

    return tracker, metrics


# ============================================================================
#  Evaluation
# ============================================================================

def evaluate_sae_chunked(
    sae,
    chunks_dir: str,
    chunk_files: List[str],
    *,
    tokens_per_batch: int = 2048,
    device: str = "cuda",
):
    """
    Evaluate SAE on held-out chunks.

    Two passes: one for the global mean (needed for honest FVU),
    one for the actual error and sparsity stats.
    """
    sae.eval()
    d_model = sae.pre_bias.shape[0]

    total_sse = 0.0
    total_x_var = 0.0
    total_x_sq = 0.0
    total_tokens = 0
    all_l0 = []
    act_counts = torch.zeros(sae.n_latents, dtype=torch.long, device=device)

    # Pass 1: global mean, used as baseline for FVU
    running_sum = torch.zeros(d_model, device=device, dtype=torch.float64)
    for cf in chunk_files:
        acts = load_chunk(chunks_dir, cf)
        running_sum += acts.sum(dim=0).to(device, dtype=torch.float64)
        total_tokens += acts.shape[0]
    global_mean = (running_sum / total_tokens).float()

    # Pass 2: reconstruction stats
    total_tokens = 0
    for cf in chunk_files:
        acts = load_chunk(chunks_dir, cf)
        loader = DataLoader(
            TensorDataset(acts),
            batch_size=tokens_per_batch,
            shuffle=False,
            pin_memory=device.startswith("cuda"),
        )
        with torch.no_grad():
            for (tokens,) in loader:
                x = tokens.to(device)
                _, lat, recon = sae(x)

                resid = x - recon
                total_sse += resid.pow(2).sum().item()
                total_x_sq += x.pow(2).sum().item()
                total_x_var += (x - global_mean).pow(2).sum().item()

                l0 = (lat > 0).float().sum(dim=-1)
                all_l0.append(l0.cpu())
                act_counts += (lat > 0).long().sum(dim=0)
                total_tokens += x.shape[0]

    all_l0 = torch.cat(all_l0)
    feature_density = act_counts.float() / total_tokens

    return {
        "recon_mse": total_sse / (total_tokens * d_model),
        "fvu": total_sse / (total_x_var + 1e-8),
        "relative_mse": total_sse / (total_x_sq + 1e-8),
        "variance_explained": 1.0 - total_sse / (total_x_var + 1e-8),
        "l0_mean": all_l0.mean().item(),
        "l0_std": all_l0.std().item(),
        "pct_active_neurons": act_counts.gt(0).float().mean().item() * 100.0,
        "pct_dead": act_counts.eq(0).float().mean().item() * 100.0,
        "pct_ultralow_density": (feature_density < 0.001).float().mean().item() * 100.0,
        "pct_ultrahigh_density": (feature_density > 0.1).float().mean().item() * 100.0,
        "feature_density_mean": feature_density.mean().item(),
        "feature_density_median": feature_density.median().item(),
        "feature_density_std": feature_density.std().item(),
        "total_tokens": total_tokens,
    }


# ============================================================================
#  Plotting
# ============================================================================

def plot_training_metrics(
    metrics: Dict[str, List[float]],
    *,
    output_dir: str = "plots",
    prefix: str = "sae",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(metrics["batch_recon_losses"], linewidth=0.5, alpha=0.7)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Reconstruction MSE (log)")
    axes[0, 0].set_xlabel("Batch")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(True, alpha=0.3)

    if metrics.get("batch_fvu"):
        axes[0, 1].plot(metrics["batch_fvu"], linewidth=0.5, alpha=0.7, color="tab:orange")
        axes[0, 1].set_title("FVU (lower = better)")
        axes[0, 1].set_xlabel("Batch")
        axes[0, 1].set_ylabel("FVU")
        axes[0, 1].grid(True, alpha=0.3)
        for ref in (0.1, 0.05, 0.01):
            axes[0, 1].axhline(y=ref, color="gray", ls="--", alpha=0.4, lw=0.8)

    if metrics.get("chunk_active_neurons_pct"):
        axes[1, 0].plot(metrics["chunk_active_neurons_pct"], "o-", ms=3, lw=1)
        axes[1, 0].set_title("Active neurons per chunk (%)")
        axes[1, 0].set_xlabel("Chunk")
        axes[1, 0].grid(True, alpha=0.3)

    if metrics.get("chunk_dead_frac") and metrics.get("chunk_mean_fvu"):
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        l1 = ax1.plot(metrics["chunk_dead_frac"], "s-", color="tab:red",
                      ms=3, lw=1, label="Dead %")
        l2 = ax2.plot(metrics["chunk_mean_fvu"], "o-", color="tab:blue",
                      ms=3, lw=1, label="FVU")
        ax1.set_xlabel("Chunk")
        ax1.set_ylabel("Dead %", color="tab:red")
        ax2.set_ylabel("FVU", color="tab:blue")
        ax1.set_title("Dead features & FVU per chunk")
        ax1.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc="upper right")
        ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_training.pdf")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ============================================================================
#  Comparison plot (Standard SAE vs ProtoSAE)
# ============================================================================

def plot_recon_comparison(
    metrics_std: Dict[str, List[float]],
    metrics_proto: Dict[str, List[float]],
    *,
    output_dir: str = "plots",
    filename: str = "sae_vs_protosae_recon.pdf",
) -> str:
    """Reconstruction error of both networks on the same axes."""
    os.makedirs(output_dir, exist_ok=True)

    def _ema(xs: List[float], alpha: float = 0.05) -> List[float]:
        if not xs:
            return []
        out, m = [], xs[0]
        for v in xs:
            m = alpha * v + (1.0 - alpha) * m
            out.append(m)
        return out

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Reconstruction MSE (log) ----
    ax = axes[0]
    ax.plot(metrics_std["batch_recon_losses"],   color="tab:blue",   alpha=0.20, lw=0.5)
    ax.plot(metrics_proto["batch_recon_losses"], color="tab:orange", alpha=0.20, lw=0.5)
    ax.plot(_ema(metrics_std["batch_recon_losses"]),
            color="tab:blue",   lw=2, label="Standard SAE (EMA)")
    ax.plot(_ema(metrics_proto["batch_recon_losses"]),
            color="tab:orange", lw=2, label="ProtoSAE (EMA)")
    ax.set_yscale("log")
    ax.set_xlabel("Training batch")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Reconstruction MSE (log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # ---- FVU ----
    ax = axes[1]
    ax.plot(metrics_std["batch_fvu"],   color="tab:blue",   alpha=0.20, lw=0.5)
    ax.plot(metrics_proto["batch_fvu"], color="tab:orange", alpha=0.20, lw=0.5)
    ax.plot(_ema(metrics_std["batch_fvu"]),
            color="tab:blue",   lw=2, label="Standard SAE (EMA)")
    ax.plot(_ema(metrics_proto["batch_fvu"]),
            color="tab:orange", lw=2, label="ProtoSAE (EMA)")
    ax.set_xlabel("Training batch")
    ax.set_ylabel("FVU")
    ax.set_title("Fraction of Variance Unexplained")
    ax.grid(True, alpha=0.3)
    for ref in (0.1, 0.05, 0.01):
        ax.axhline(y=ref, color="gray", ls="--", alpha=0.4, lw=0.8)
    ax.legend(loc="upper right")

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ============================================================================
#  Single-model train+eval+save (used twice: once for each architecture)
# ============================================================================

def train_one_model(
    sae,
    *,
    name: str,
    train_dir: str,
    test_dir: str,
    train_chunk_files: List[str],
    test_chunk_files: List[str],
    args,
    meta: Dict,
    logger: logging.Logger,
    seed: int,
    num_epochs: int,
):
    # Same data order for both architectures so the comparison is fair.
    random.seed(seed)
    torch.manual_seed(seed)

    n_params = sum(p.numel() for p in sae.parameters())
    logger.info(f"[{name}] Params: {n_params:,}  Epochs: {num_epochs}")

    clean_gpus()
    tracker, metrics = train_sae_chunked(
        sae, train_dir, train_chunk_files,
        num_epochs=num_epochs,
        tokens_per_batch=args.tokens_per_batch,
        n_batches=args.n_batches,
        lr=args.lr,
        aux_coef=args.aux_coef,
        k_aux=args.k_aux,
        dead_threshold=args.dead_threshold,
        device=args.device,
        normalize_decoder=not args.no_normalize_decoder,
    )
    logger.info(f"[{name}] Trained for {len(metrics['batch_recon_losses'])} batches")

    logger.info(f"[{name}] Evaluating on {len(test_chunk_files)} test chunks...")
    results = evaluate_sae_chunked(
        sae, test_dir, test_chunk_files,
        tokens_per_batch=args.tokens_per_batch,
        device=args.device,
    )

    # Per-architecture training-curve plot (existing 2x2 panel).
    plot_training_metrics(metrics, output_dir=args.plots_dir, prefix=name)

    layer = meta.get("layer_idx", "?")
    save_path = os.path.join(args.save_dir, f"{name}_layer{layer}_k{args.k}.pth")
    save_config = {
        "sae_type": name,
        "activation_type": "TopK",
        "d_model": meta["d_model"],
        "n_latents": meta["d_model"] * args.expansion_factor,
        "k": args.k,
        "expansion_factor": args.expansion_factor,
        "num_epochs": num_epochs,
        "lr": args.lr,
    }
    torch.save({
        "state_dict": sae.state_dict(),
        "config": save_config,
        "metadata": meta,
        "results": results,
        "training_metrics": metrics,
    }, save_path)
    logger.info(f"[{name}] Saved -> {save_path}")

    logger.info(f"[{name}] Test FVU={results['fvu']:.6f}  "
                f"VarExp={results['variance_explained']:.2%}  "
                f"L0={results['l0_mean']:.1f}  "
                f"Dead={results['pct_dead']:.2f}%")

    return metrics, results


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Standard SAE and ProtoSAE side-by-side and "
                    "compare reconstruction error."
    )

    # Architecture
    parser.add_argument("--expansion_factor", type=int, default=4,
                        help="n_latents = d_model * expansion_factor")
    parser.add_argument("--k", type=int, default=128,
                        help="TopK sparsity")
    parser.add_argument("--no_normalize_decoder", action="store_true",
                        help="Disable unit-norm decoder columns")
    parser.add_argument("--proto_eps", type=float, default=1e-3,
                        help="ProtoPNet stability constant; sim = log((d^2+1)/(d^2+eps))")

    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_epochs_std", type=int, default=1,
                        help="Epochs to train the Standard SAE.")
    parser.add_argument("--num_epochs_proto", type=int, default=10,
                        help="Epochs to train the ProtoSAE.")
    parser.add_argument("--tokens_per_batch", type=int, default=2048)
    parser.add_argument("--n_batches", type=int, default=-1,
                        help="Max batches to train. -1 = all data.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed used to make both runs see the same chunk order.")

    # Quick smoke-test mode
    parser.add_argument("--quick", action="store_true",
                        help="Run a tiny pipeline check: 1 epoch, 5 batches per "
                             "model. Useful for verifying the code wires up.")

    # Dead features / AuxK
    parser.add_argument("--dead_threshold", type=int, default=1_000_000)
    parser.add_argument("--aux_coef", type=float, default=1 / 32)
    parser.add_argument("--k_aux", type=int, default=512)

    # I/O
    parser.add_argument("--chunks_dir", type=str, default="data/activations/",
                        help="Root dir containing train/, test/, and metadata.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--plots_dir", type=str, default="plots")
    parser.add_argument("--save_dir", type=str, default="data")

    args = parser.parse_args()

    if args.quick:
        args.num_epochs_std = 1
        args.num_epochs_proto = 1
        args.n_batches = 5

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("sae_train")

    if args.quick:
        logger.info("--quick: running 1 epoch x 5 batches per model.")

    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- metadata ----------------------------------------------------------
    meta_path = os.path.join(args.chunks_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    d_model = meta["d_model"]
    n_latents = d_model * args.expansion_factor

    # ---- chunk discovery ---------------------------------------------------
    train_dir = os.path.join(args.chunks_dir, "train")
    test_dir = os.path.join(args.chunks_dir, "test")

    def _list_chunks(d):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")
        return sorted(
            [f for f in os.listdir(d) if f.startswith("chunk_") and f.endswith(".pt")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )

    train_chunk_files = _list_chunks(train_dir)
    test_chunk_files = _list_chunks(test_dir)

    if not train_chunk_files:
        raise FileNotFoundError(f"No chunk files found in {train_dir}")
    if not test_chunk_files:
        raise FileNotFoundError(f"No chunk files found in {test_dir}")

    logger.info(f"Chunks: {len(train_chunk_files)} train, {len(test_chunk_files)} test")
    logger.info(f"d_model={d_model}, n_latents={n_latents}, k={args.k}, "
                f"epochs_std={args.num_epochs_std}, "
                f"epochs_proto={args.num_epochs_proto}, "
                f"n_batches={args.n_batches}")

    # ---- build standard SAE -----------------------------------------------
    sae_std = Autoencoder(
        n_latents, d_model,
        activation=TopK(k=args.k),
        normalize=True,
    ).to(args.device)

    # ---- build ProtoSAE ----------------------------------------------------
    sae_proto = ProtoAutoencoder(
        n_latents, d_model,
        activation=TopK(k=args.k),
        normalize=True,
        eps=args.proto_eps,
    ).to(args.device)

    # ---- train both --------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("  Training Standard SAE")
    logger.info("=" * 60)
    metrics_std, results_std = train_one_model(
        sae_std, name="sae_standard",
        train_dir=train_dir, test_dir=test_dir,
        train_chunk_files=list(train_chunk_files),  # copy: each run shuffles in-place
        test_chunk_files=test_chunk_files,
        args=args, meta=meta, logger=logger,
        seed=args.seed,
        num_epochs=args.num_epochs_std,
    )

    logger.info("\n" + "=" * 60)
    logger.info("  Training ProtoSAE")
    logger.info("=" * 60)

    # Option B: calibrate the proto scale from one batch so the initial
    # reconstruction magnitude is in line with ||x||. Without this, the
    # ProtoSAE starts at FVU >> 1 just from the activation magnitude
    # mismatch and burns the first chunk or two climbing out of that
    # hole -- which is unfair against the standard SAE in the comparison.
    calib_acts = load_chunk(train_dir, train_chunk_files[0])
    calib_x = calib_acts[: args.tokens_per_batch].to(args.device)
    old_scale, new_scale = sae_proto.calibrate_scale_from_batch(
        calib_x, k=args.k
    )
    logger.info(f"[sae_proto] One-batch scale calibration: "
                f"{old_scale:.4f} -> {new_scale:.4f} "
                f"(target ||recon|| ~ ||x||)")
    del calib_acts, calib_x
    clean_gpus()

    metrics_proto, results_proto = train_one_model(
        sae_proto, name="sae_proto",
        train_dir=train_dir, test_dir=test_dir,
        train_chunk_files=list(train_chunk_files),
        test_chunk_files=test_chunk_files,
        args=args, meta=meta, logger=logger,
        seed=args.seed,
        num_epochs=args.num_epochs_proto,
    )

    # ---- comparison plot ---------------------------------------------------
    cmp_path = plot_recon_comparison(
        metrics_std, metrics_proto,
        output_dir=args.plots_dir,
        filename="sae_vs_protosae_recon.pdf",
    )
    logger.info(f"\nComparison plot saved -> {cmp_path}")

    # ---- summary -----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("  TEST-SET COMPARISON")
    logger.info("=" * 60)
    logger.info(f"{'metric':<22}{'Standard SAE':>16}{'ProtoSAE':>16}")
    for key, fmt in [
        ("recon_mse",          "{:>16.3e}"),
        ("fvu",                "{:>16.6f}"),
        ("variance_explained", "{:>16.2%}"),
        ("l0_mean",            "{:>16.1f}"),
        ("pct_dead",           "{:>15.2f}%"),
    ]:
        logger.info(f"  {key:<20}" + fmt.format(results_std[key]) + fmt.format(results_proto[key]))


if __name__ == "__main__":
    main()



    
    
# """
# Train a Standard TopK SAE on chunked activations.

# Loads activation chunks produced by collect_activations.py, trains a TopK
# sparse autoencoder with AuxK auxiliary loss for dead-feature revival, and
# evaluates on held-out chunks.

# Usage:
#     python train_sae.py --k 128 --expansion_factor 4
#     python train_sae.py --chunks_dir data/activations/train --k 64 --num_epochs 5
# """

# import os
# import json
# import random
# import gc
# import logging
# import argparse

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import torch.optim as optim

# from typing import List, Dict
# from tqdm import tqdm
# from torch.utils.data import DataLoader, TensorDataset
# from torch.cuda.amp import autocast, GradScaler
# from src.topk_sae import TopK, Autoencoder


# # ============================================================================
# #  Utilities
# # ============================================================================

# def clean_gpus() -> None:
#     gc.collect()
#     torch.cuda.empty_cache()


# def load_chunk(chunks_dir: str, chunk_file: str) -> torch.Tensor:
#     path = os.path.join(chunks_dir, chunk_file)
#     chunk_data = torch.load(path, map_location="cpu", weights_only=False)
#     if isinstance(chunk_data, dict):
#         for key in ["final_token_activations", "activations"]:
#             if key in chunk_data:
#                 return chunk_data[key].float()
#         raise KeyError(f"Unknown chunk format. Keys: {list(chunk_data.keys())}")
#     return chunk_data.float()


# def compute_neuron_activation_metrics(
#     activation_counts: torch.Tensor,
# ) -> Dict[str, float]:
#     n_neurons = activation_counts.numel()
#     counts = activation_counts.float().cpu().numpy()
#     pct_active = (counts > 0).mean() * 100.0
#     if counts.sum() == 0:
#         return {"activation_entropy": 0.0, "pct_active": 0.0, "gini_coefficient": 1.0}
#     probs = counts / counts.sum()
#     active_probs = probs[probs > 0]
#     entropy = -(active_probs * np.log(active_probs + 1e-12)).sum()
#     norm_entropy = float(entropy / np.log(n_neurons))
#     sorted_counts = np.sort(counts)
#     cum = np.cumsum(sorted_counts)
#     gini = (
#         2 * (np.arange(1, n_neurons + 1) * sorted_counts).sum()
#     ) / (n_neurons * cum[-1]) - (n_neurons + 1) / n_neurons
#     gini = float(np.clip(gini, 0.0, 1.0))
#     return {
#         "activation_entropy": norm_entropy,
#         "pct_active": pct_active,
#         "gini_coefficient": gini,
#     }


# # ============================================================================
# #  Dead latent tracker
# # ============================================================================

# class DeadLatentTracker:
#     def __init__(self, n_latents: int, dead_threshold: int = 10_000, device: str = "cuda"):
#         self.n_latents = n_latents
#         self.dead_threshold = dead_threshold
#         self.device = device
#         self.steps_since_activation = torch.zeros(
#             n_latents, dtype=torch.long, device=device
#         )
#         self.total_steps = 0

#     def update(self, latents: torch.Tensor) -> None:
#         batch_active = (latents > 0).any(dim=0)
#         self.steps_since_activation = torch.where(
#             batch_active,
#             torch.zeros_like(self.steps_since_activation),
#             self.steps_since_activation + latents.size(0),
#         )
#         self.total_steps += 1

#     def get_dead_latents(self) -> torch.Tensor:
#         return self.steps_since_activation >= self.dead_threshold

#     def get_dead_fraction(self) -> float:
#         if self.total_steps < self.dead_threshold:
#             return 0.0
#         return float(self.get_dead_latents().float().mean().item())


# # ============================================================================
# #  Auxiliary loss (AuxK)
# # ============================================================================

# def compute_auxiliary_loss(
#     sae,
#     x: torch.Tensor,
#     recon: torch.Tensor,
#     tracker: DeadLatentTracker,
#     k_aux: int = 512,
#     alpha: float = 1 / 32,
# ) -> torch.Tensor:
#     """
#     AuxK loss: train currently-dead latents to explain the main model's
#     reconstruction residual. Gives dead features gradient signal so they
#     can come back to life.
#     """
#     dead_mask = tracker.get_dead_latents()
#     if dead_mask.sum() < k_aux:
#         return torch.tensor(0.0, device=x.device, requires_grad=True)

#     lat_pre = sae.encode_pre_act(x)
#     dead_only = lat_pre.clone()
#     dead_only[:, ~dead_mask] = 0.0

#     dead_idx = torch.nonzero(dead_mask, as_tuple=True)[0]
#     dead_vals = dead_only[:, dead_idx]
#     _, topk_in_dead = torch.topk(dead_vals.abs(), k_aux, dim=1)

#     B = x.shape[0]
#     batch_idx = torch.arange(B, device=x.device).unsqueeze(1)
#     chosen_dead = dead_idx[topk_in_dead]

#     aux_pre = torch.zeros_like(lat_pre)
#     aux_pre[batch_idx, chosen_dead] = lat_pre[batch_idx, chosen_dead]

#     aux_act = torch.exp(aux_pre)
#     e_hat = sae.decoder(aux_act)

#     aux_mse = F.mse_loss(x - recon, e_hat)
#     return alpha * aux_mse


# # ============================================================================
# #  Training loop
# # ============================================================================

# def train_sae_chunked(
#     sae,
#     chunks_dir: str,
#     train_chunk_files: List[str],
#     *,
#     num_epochs: int = 1,
#     tokens_per_batch: int = 2048,
#     n_batches: int = -1,
#     lr: float = 3e-4,
#     aux_coef: float = 1 / 32,
#     k_aux: int = 512,
#     dead_threshold: int = 1_000_000,
#     device: str = "cuda",
#     normalize_decoder: bool = True,
# ):
#     optimizer = optim.Adam(sae.parameters(), lr=lr)
#     scaler = GradScaler()
#     tracker = DeadLatentTracker(sae.n_latents, dead_threshold, device)

#     metrics = {
#         "batch_recon_losses": [],
#         "batch_fvu": [],
#         "batch_l0": [],
#         "batch_aux_losses": [],
#         "chunk_activation_entropy": [],
#         "chunk_active_neurons_pct": [],
#         "chunk_gini_coefficient": [],
#         "chunk_dead_frac": [],
#         "chunk_mean_recon": [],
#         "chunk_mean_fvu": [],
#     }

#     sae.train()
#     total_batches_done = 0
#     hit_limit = False

#     if n_batches > 0:
#         logging.getLogger("sae_train").info(f"  Training limited to {n_batches} batches")

#     for epoch in range(num_epochs):
#         if hit_limit:
#             break
#         random.shuffle(train_chunk_files)
#         chunk_pbar = tqdm(train_chunk_files, desc=f"Epoch {epoch + 1}/{num_epochs}")

#         for chunk_file in chunk_pbar:
#             if hit_limit:
#                 break
#             acts = load_chunk(chunks_dir, chunk_file)
#             loader = DataLoader(
#                 TensorDataset(acts),
#                 batch_size=tokens_per_batch,
#                 shuffle=True,
#                 pin_memory=device.startswith("cuda"),
#             )

#             chunk_act_counts = torch.zeros(
#                 sae.n_latents, dtype=torch.long, device=device
#             )
#             chunk_recon, chunk_aux, chunk_fvu, chunk_l0 = [], [], [], []

#             for (tokens,) in loader:
#                 if n_batches > 0 and total_batches_done >= n_batches:
#                     hit_limit = True
#                     break

#                 x = tokens.to(device)

#                 optimizer.zero_grad()
#                 with autocast():
#                     lat_pre, lat, recon = sae(x)
#                     tracker.update(lat)
#                     chunk_act_counts += (lat > 0).sum(dim=0)

#                     recon_loss = F.mse_loss(recon, x)
#                     aux_loss = compute_auxiliary_loss(
#                         sae, x, recon, tracker, k_aux, aux_coef
#                     )
#                     loss = recon_loss + aux_loss

#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 scaler.update()

#                 # Unit-norm decoder columns after each step
#                 if normalize_decoder and hasattr(sae, "normalize") and sae.normalize:
#                     with torch.no_grad():
#                         sae.decoder.weight.data = F.normalize(
#                             sae.decoder.weight.data, dim=0
#                         )

#                 # Per-batch metrics
#                 with torch.no_grad():
#                     x_var = (x - x.mean(dim=0)).pow(2).sum()
#                     resid = (x - recon).pow(2).sum()
#                     fvu = (resid / (x_var + 1e-8)).item()
#                     l0 = (lat > 0).float().sum(dim=-1).mean().item()

#                 metrics["batch_recon_losses"].append(recon_loss.item())
#                 metrics["batch_fvu"].append(fvu)
#                 metrics["batch_l0"].append(l0)
#                 metrics["batch_aux_losses"].append(aux_loss.item())
#                 chunk_recon.append(recon_loss.item())
#                 chunk_fvu.append(fvu)
#                 chunk_l0.append(l0)
#                 chunk_aux.append(aux_loss.item())

#                 total_batches_done += 1

#             if not chunk_recon:
#                 continue

#             # Chunk-level summary
#             avg_recon = np.mean(chunk_recon)
#             avg_fvu = np.mean(chunk_fvu)
#             avg_l0 = np.mean(chunk_l0)
#             dead_frac = tracker.get_dead_fraction() * 100.0

#             chunk_pbar.set_postfix({
#                 "batch": f"{total_batches_done}",
#                 "recon": f"{avg_recon:.3e}",
#                 "fvu": f"{avg_fvu:.4f}",
#                 "L0": f"{avg_l0:.0f}",
#                 "dead%": f"{dead_frac:.1f}",
#             })

#             cm = compute_neuron_activation_metrics(chunk_act_counts)
#             metrics["chunk_activation_entropy"].append(cm["activation_entropy"])
#             metrics["chunk_active_neurons_pct"].append(cm["pct_active"])
#             metrics["chunk_gini_coefficient"].append(cm["gini_coefficient"])
#             metrics["chunk_dead_frac"].append(dead_frac)
#             metrics["chunk_mean_recon"].append(avg_recon)
#             metrics["chunk_mean_fvu"].append(avg_fvu)

#             del acts
#             torch.cuda.empty_cache()

#     return tracker, metrics


# # ============================================================================
# #  Evaluation
# # ============================================================================

# def evaluate_sae_chunked(
#     sae,
#     chunks_dir: str,
#     chunk_files: List[str],
#     *,
#     tokens_per_batch: int = 2048,
#     device: str = "cuda",
# ):
#     """
#     Evaluate SAE on held-out chunks.

#     Two passes: one for the global mean (needed for honest FVU),
#     one for the actual error and sparsity stats.
#     """
#     sae.eval()
#     d_model = sae.pre_bias.shape[0]

#     total_sse = 0.0
#     total_x_var = 0.0
#     total_x_sq = 0.0
#     total_tokens = 0
#     all_l0 = []
#     act_counts = torch.zeros(sae.n_latents, dtype=torch.long, device=device)

#     # Pass 1: global mean, used as baseline for FVU
#     running_sum = torch.zeros(d_model, device=device, dtype=torch.float64)
#     for cf in chunk_files:
#         acts = load_chunk(chunks_dir, cf)
#         running_sum += acts.sum(dim=0).to(device, dtype=torch.float64)
#         total_tokens += acts.shape[0]
#     global_mean = (running_sum / total_tokens).float()

#     # Pass 2: reconstruction stats
#     total_tokens = 0
#     for cf in chunk_files:
#         acts = load_chunk(chunks_dir, cf)
#         loader = DataLoader(
#             TensorDataset(acts),
#             batch_size=tokens_per_batch,
#             shuffle=False,
#             pin_memory=device.startswith("cuda"),
#         )
#         with torch.no_grad():
#             for (tokens,) in loader:
#                 x = tokens.to(device)
#                 _, lat, recon = sae(x)

#                 resid = x - recon
#                 total_sse += resid.pow(2).sum().item()
#                 total_x_sq += x.pow(2).sum().item()
#                 total_x_var += (x - global_mean).pow(2).sum().item()

#                 l0 = (lat > 0).float().sum(dim=-1)
#                 all_l0.append(l0.cpu())
#                 act_counts += (lat > 0).long().sum(dim=0)
#                 total_tokens += x.shape[0]

#     all_l0 = torch.cat(all_l0)
#     feature_density = act_counts.float() / total_tokens

#     return {
#         "recon_mse": total_sse / (total_tokens * d_model),
#         "fvu": total_sse / (total_x_var + 1e-8),
#         "relative_mse": total_sse / (total_x_sq + 1e-8),
#         "variance_explained": 1.0 - total_sse / (total_x_var + 1e-8),
#         "l0_mean": all_l0.mean().item(),
#         "l0_std": all_l0.std().item(),
#         "pct_active_neurons": act_counts.gt(0).float().mean().item() * 100.0,
#         "pct_dead": act_counts.eq(0).float().mean().item() * 100.0,
#         "pct_ultralow_density": (feature_density < 0.001).float().mean().item() * 100.0,
#         "pct_ultrahigh_density": (feature_density > 0.1).float().mean().item() * 100.0,
#         "feature_density_mean": feature_density.mean().item(),
#         "feature_density_median": feature_density.median().item(),
#         "feature_density_std": feature_density.std().item(),
#         "total_tokens": total_tokens,
#     }


# # ============================================================================
# #  Plotting
# # ============================================================================

# def plot_training_metrics(
#     metrics: Dict[str, List[float]],
#     *,
#     output_dir: str = "plots",
#     prefix: str = "sae",
# ) -> str:
#     os.makedirs(output_dir, exist_ok=True)
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#     axes[0, 0].plot(metrics["batch_recon_losses"], linewidth=0.5, alpha=0.7)
#     axes[0, 0].set_yscale("log")
#     axes[0, 0].set_title("Reconstruction MSE (log)")
#     axes[0, 0].set_xlabel("Batch")
#     axes[0, 0].set_ylabel("MSE")
#     axes[0, 0].grid(True, alpha=0.3)

#     if metrics.get("batch_fvu"):
#         axes[0, 1].plot(metrics["batch_fvu"], linewidth=0.5, alpha=0.7, color="tab:orange")
#         axes[0, 1].set_title("FVU (lower = better)")
#         axes[0, 1].set_xlabel("Batch")
#         axes[0, 1].set_ylabel("FVU")
#         axes[0, 1].grid(True, alpha=0.3)
#         for ref in (0.1, 0.05, 0.01):
#             axes[0, 1].axhline(y=ref, color="gray", ls="--", alpha=0.4, lw=0.8)

#     if metrics.get("chunk_active_neurons_pct"):
#         axes[1, 0].plot(metrics["chunk_active_neurons_pct"], "o-", ms=3, lw=1)
#         axes[1, 0].set_title("Active neurons per chunk (%)")
#         axes[1, 0].set_xlabel("Chunk")
#         axes[1, 0].grid(True, alpha=0.3)

#     if metrics.get("chunk_dead_frac") and metrics.get("chunk_mean_fvu"):
#         ax1 = axes[1, 1]
#         ax2 = ax1.twinx()
#         l1 = ax1.plot(metrics["chunk_dead_frac"], "s-", color="tab:red",
#                       ms=3, lw=1, label="Dead %")
#         l2 = ax2.plot(metrics["chunk_mean_fvu"], "o-", color="tab:blue",
#                       ms=3, lw=1, label="FVU")
#         ax1.set_xlabel("Chunk")
#         ax1.set_ylabel("Dead %", color="tab:red")
#         ax2.set_ylabel("FVU", color="tab:blue")
#         ax1.set_title("Dead features & FVU per chunk")
#         ax1.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc="upper right")
#         ax1.grid(True, alpha=0.3)

#     fig.tight_layout()
#     path = os.path.join(output_dir, f"{prefix}_training.pdf")
#     fig.savefig(path, dpi=200)
#     plt.close(fig)
#     return path


# # ============================================================================
# #  Main
# # ============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description="Train a TopK SAE on chunked activations"
#     )

#     # Architecture
#     parser.add_argument("--expansion_factor", type=int, default=4,
#                         help="n_latents = d_model * expansion_factor")
#     parser.add_argument("--k", type=int, default=128,
#                         help="TopK sparsity")
#     parser.add_argument("--no_normalize_decoder", action="store_true",
#                         help="Disable unit-norm decoder columns")

#     # Training
#     parser.add_argument("--lr", type=float, default=3e-4)
#     parser.add_argument("--num_epochs", type=int, default=3)
#     parser.add_argument("--tokens_per_batch", type=int, default=2048)
#     parser.add_argument("--n_batches", type=int, default=-1,
#                         help="Max batches to train. -1 = all data.")

#     # Dead features / AuxK
#     parser.add_argument("--dead_threshold", type=int, default=1_000_000)
#     parser.add_argument("--aux_coef", type=float, default=1 / 32)
#     parser.add_argument("--k_aux", type=int, default=512)

#     # I/O
#     parser.add_argument("--chunks_dir", type=str, default="data/activations/",
#                         help="Root dir containing train/, test/, and metadata.json")
#     parser.add_argument("--device", type=str, default="cuda:0")
#     parser.add_argument("--plots_dir", type=str, default="plots")
#     parser.add_argument("--save_dir", type=str, default="data")

#     args = parser.parse_args()

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#     )
#     logger = logging.getLogger("sae_train")

#     device = args.device
#     os.makedirs(args.plots_dir, exist_ok=True)
#     os.makedirs(args.save_dir, exist_ok=True)

#     # ---- metadata ----------------------------------------------------------
#     meta_path = os.path.join(args.chunks_dir, "metadata.json")
#     with open(meta_path) as f:
#         meta = json.load(f)
#     d_model = meta["d_model"]
#     n_latents = d_model * args.expansion_factor

#     # ---- chunk discovery ---------------------------------------------------
#     train_dir = os.path.join(args.chunks_dir, "train")
#     test_dir = os.path.join(args.chunks_dir, "test")

#     def _list_chunks(d):
#         if not os.path.isdir(d):
#             raise FileNotFoundError(f"Directory not found: {d}")
#         return sorted(
#             [f for f in os.listdir(d) if f.startswith("chunk_") and f.endswith(".pt")],
#             key=lambda x: int(x.split("_")[1].split(".")[0]),
#         )

#     train_chunk_files = _list_chunks(train_dir)
#     test_chunk_files = _list_chunks(test_dir)

#     if not train_chunk_files:
#         raise FileNotFoundError(f"No chunk files found in {train_dir}")
#     if not test_chunk_files:
#         raise FileNotFoundError(f"No chunk files found in {test_dir}")

#     logger.info(f"Chunks: {len(train_chunk_files)} train, {len(test_chunk_files)} test")
#     logger.info(f"Architecture: d_model={d_model}, n_latents={n_latents}, k={args.k}")

#     # ---- build model -------------------------------------------------------
#     activation = TopK(k=args.k)
#     sae = Autoencoder(n_latents, d_model, activation=activation, normalize=True).to(device)
#     n_params = sum(p.numel() for p in sae.parameters())
#     logger.info(f"Params: {n_params:,}")

#     if args.n_batches > 0:
#         logger.info(f"Training for {args.n_batches} batches (fast mode)")
#     else:
#         logger.info(f"Training on all data "
#                     f"({len(train_chunk_files)} chunks × {args.num_epochs} epochs)")

#     # ---- train -------------------------------------------------------------
#     clean_gpus()
#     tracker, metrics = train_sae_chunked(
#         sae, train_dir, train_chunk_files,
#         num_epochs=args.num_epochs,
#         tokens_per_batch=args.tokens_per_batch,
#         n_batches=args.n_batches,
#         lr=args.lr,
#         aux_coef=args.aux_coef,
#         k_aux=args.k_aux,
#         dead_threshold=args.dead_threshold,
#         device=device,
#         normalize_decoder=not args.no_normalize_decoder,
#     )
#     logger.info(f"Trained for {len(metrics['batch_recon_losses'])} batches total")

#     # ---- evaluate ----------------------------------------------------------
#     logger.info(f"Evaluating on {len(test_chunk_files)} test chunks...")
#     results = evaluate_sae_chunked(
#         sae, test_dir, test_chunk_files,
#         tokens_per_batch=args.tokens_per_batch,
#         device=device,
#     )

#     # ---- plot --------------------------------------------------------------
#     plot_training_metrics(metrics, output_dir=args.plots_dir, prefix="sae_standard")

#     # ---- save --------------------------------------------------------------
#     layer = meta.get("layer_idx", "?")
#     save_path = os.path.join(args.save_dir, f"sae_standard_layer{layer}_k{args.k}.pth")
#     save_config = {
#         "sae_type": "standard",
#         "activation_type": "TopK",
#         "d_model": d_model,
#         "n_latents": n_latents,
#         "k": args.k,
#         "expansion_factor": args.expansion_factor,
#         "num_epochs": args.num_epochs,
#         "lr": args.lr,
#     }
#     torch.save({
#         "state_dict": sae.state_dict(),
#         "config": save_config,
#         "metadata": meta,
#         "results": results,
#         "training_metrics": metrics,
#     }, save_path)
#     logger.info(f"Saved -> {save_path}")

#     # ---- summary -----------------------------------------------------------
#     logger.info(f"\n{'='*60}")
#     logger.info("  TEST SET RESULTS (held-out data)")
#     logger.info(f"{'='*60}")
#     logger.info(f"  FVU:                {results['fvu']:.6f}")
#     logger.info(f"  Variance explained: {results['variance_explained']:.2%}")
#     logger.info(f"  Recon MSE:          {results['recon_mse']:.3e}")
#     logger.info(f"  Relative MSE:       {results['relative_mse']:.6f}")
#     logger.info(f"  L0:                 {results['l0_mean']:.1f} ± {results['l0_std']:.2f}")
#     logger.info(f"  Dead features:      {results['pct_dead']:.2f}%")
#     logger.info(f"  Active features:    {results['pct_active_neurons']:.2f}%")
#     logger.info(f"  Ultra-low density:  {results['pct_ultralow_density']:.2f}%")
#     logger.info(f"  Ultra-high density: {results['pct_ultrahigh_density']:.2f}%")
#     logger.info(f"  Total test tokens:  {results['total_tokens']:,}")

#     # ---- sanity-check reconstruction ---------------------------------------
#     logger.info(f"\n{'='*60}")
#     logger.info("  RECONSTRUCTION EXAMPLE (1 sample)")
#     logger.info(f"{'='*60}")
#     sample_acts = load_chunk(train_dir, train_chunk_files[0])
#     x_sample = sample_acts[:1].to(device)
#     sae.eval()
#     with torch.no_grad():
#         _, lat, recon = sae(x_sample)
#         mse = (x_sample - recon).pow(2).sum().item()
#         n_active = (lat > 0).sum().item()
#         top_features = lat[0].topk(5)
#     logger.info(f"  Input  (first 10): {x_sample[0, :10].cpu().tolist()}")
#     logger.info(f"  Recon  (first 10): {recon[0, :10].cpu().tolist()}")
#     logger.info(f"  MSE: {mse:.4f},  Active features: {n_active}")
#     logger.info(f"  Top 5 feature indices: {top_features.indices.cpu().tolist()}")
#     logger.info(f"  Top 5 feature values:  "
#                 f"{[f'{v:.4f}' for v in top_features.values.cpu().tolist()]}")

#     logger.info(f"\nPlots saved to {args.plots_dir}/")
#     logger.info(f"Model saved to {save_path}")


# if __name__ == "__main__":
#     main()
