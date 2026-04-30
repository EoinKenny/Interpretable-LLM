#!/usr/bin/env python
"""
collect_top_activating_examples.py

For each SAE latent, find the TOP_K sequences that activate it most strongly
at the final token, then compute the latent's activation trajectory across
the full sequence (all seq_len positions). Save the data needed for later
visualization — no plotting here.

Pipeline:
  Pass 1: scan all activation chunks on disk. Each chunk contains both
          final-token activations and the input_ids that produced them.
          Run activations through the SAE, track top-K per latent using
          a fully vectorized GPU top-k merge. Track only (chunk_idx, pos)
          for each winner — input_ids are fetched lazily in pass 2.

  Pass 2: collect the unique (chunk, pos) winners, load their input_ids,
          re-run those sequences through the LLM to get residual stream
          at every position, pass that through the SAE to get per-latent
          trajectories, and redistribute trajectories back to each
          latent's top-K records.

Output:
  A single .pt file. Structure:
    {
      "metadata": {...},        # model, layer, seq_len, etc.
      "latents": {
          j: [  # up to TOP_K records, sorted descending by final activation
              {
                "sequence_ids":           list[int],   # length seq_len
                "sequence_text":          str,         # decoded
                "final_token_activation": float,       # ranking value
                "trajectory":             list[float], # length seq_len
              },
              ...
          ],
          ...
      }
    }

Usage:
  python collect_top_activating_examples.py \\
      --chunks_dir data/activations/train \\
      --activations_root data/activations \\
      --sae_ckpt data/sae_standard_layer8_k128.pth \\
      --top_k 5
"""

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.topk_sae import Autoencoder, TopK


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def get_layer_module(model, layer_idx: int):
    """Get transformer block by index across common HF architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = model.gpt_neox.layers
    else:
        raise ValueError(f"Unknown architecture: {type(model)}")
    return layers[layer_idx], len(layers)


def load_sae(sae_path: str, device: str):
    ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    meta = ckpt.get("metadata", {})

    activation = TopK(k=cfg["k"])
    sae = Autoencoder(
        n_latents=cfg["n_latents"],
        n_inputs=cfg["d_model"],
        activation=activation,
        normalize=True,
    )
    sae.load_state_dict(ckpt["state_dict"], strict=False)
    sae = sae.to(device).eval()
    for p in sae.parameters():
        p.requires_grad = False
    return sae, cfg, meta


# ---------------------------------------------------------------------------
#  Pass 1: find top-K per latent (fully vectorized)
# ---------------------------------------------------------------------------

def pass1_find_topk(
    chunk_files,
    sae,
    top_k: int,
    device: str,
    sae_batch_size: int = 2048,
):
    """
    Scan all chunks; for each latent, keep the top_k largest final-token
    activations and remember WHERE each winner came from (chunk_idx, pos).

    Tracker tensors are all shape (K, n_latents), living on GPU. Each batch:
      1. Compute SAE latents for the batch → (B, n_latents).
      2. Take batch-local top-k along dim=0 → (k, n_latents) with row indices.
      3. Concatenate with the running top-k, take top-k again along dim=0.
         gather() pulls matching chunk_idx/pos values into the new slots.
    """
    n_latents = sae.n_latents
    K = top_k

    topk_vals = torch.full((K, n_latents), float("-inf"), device=device)
    topk_chunk = torch.full((K, n_latents), -1, dtype=torch.long, device=device)
    topk_pos = torch.full((K, n_latents), -1, dtype=torch.long, device=device)

    for chunk_idx, chunk_path in enumerate(tqdm(chunk_files, desc="Pass 1: scanning")):
        data = torch.load(chunk_path, map_location="cpu", weights_only=False)
        acts = data["final_token_activations"].float()
        N = acts.shape[0]

        for i in range(0, N, sae_batch_size):
            end = min(i + sae_batch_size, N)
            x = acts[i:end].to(device)
            with torch.no_grad():
                _, lat, _ = sae(x)  # (B, n_latents)
            B = lat.shape[0]

            kk = min(K, B)
            batch_vals, batch_rows = lat.topk(kk, dim=0)  # (kk, n_latents)

            positions = torch.arange(i, end, device=device, dtype=torch.long)
            batch_pos = positions[batch_rows]             # (kk, n_latents)
            batch_chunks = torch.full_like(batch_pos, chunk_idx)

            # Pad if the final batch was smaller than K
            if kk < K:
                pad_shape = (K - kk, n_latents)
                batch_vals = torch.cat([
                    batch_vals,
                    torch.full(pad_shape, float("-inf"), device=device),
                ], dim=0)
                batch_pos = torch.cat([
                    batch_pos,
                    torch.full(pad_shape, -1, dtype=torch.long, device=device),
                ], dim=0)
                batch_chunks = torch.cat([
                    batch_chunks,
                    torch.full(pad_shape, -1, dtype=torch.long, device=device),
                ], dim=0)

            # Merge with running top-K, then re-take top-K
            merged_vals = torch.cat([topk_vals, batch_vals], dim=0)      # (2K, n_latents)
            merged_pos = torch.cat([topk_pos, batch_pos], dim=0)
            merged_chunks = torch.cat([topk_chunk, batch_chunks], dim=0)

            topk_vals, new_idx = merged_vals.topk(K, dim=0)
            topk_pos = merged_pos.gather(0, new_idx)
            topk_chunk = merged_chunks.gather(0, new_idx)

        del data, acts
        gc.collect()
        torch.cuda.empty_cache()

    return topk_vals.cpu(), topk_chunk.cpu(), topk_pos.cpu()


# ---------------------------------------------------------------------------
#  Pass 2: load unique sequences and compute per-latent trajectories
# ---------------------------------------------------------------------------

def pass2_trajectories(
    chunk_files,
    topk_vals,
    topk_chunk,
    topk_pos,
    sae,
    llm,
    layer_idx: int,
    tokenizer,
    device: str,
    *,
    seq_len: int,
    d_model: int,
    n_latents: int,
    top_k: int,
    llm_batch_size: int = 32,
    sae_batch_size: int = 4096,
):
    K = top_k

    # ---- Identify unique (chunk, pos) winners ------------------------------
    valid_mask = topk_vals > 0
    if not valid_mask.any():
        print("No positive activations found — all latents dead?")
        return {}

    locations = {}  # (chunk_idx, pos) -> unique_idx
    for j in range(n_latents):
        for r in range(K):
            if not valid_mask[r, j]:
                continue
            c = int(topk_chunk[r, j].item())
            p = int(topk_pos[r, j].item())
            if (c, p) not in locations:
                locations[(c, p)] = len(locations)
    n_unique = len(locations)
    print(f"Unique sequences needed: {n_unique:,}")

    # ---- Load input_ids for those locations (grouped by chunk) -------------
    unique_input_ids = torch.zeros(n_unique, seq_len, dtype=torch.long)
    by_chunk = defaultdict(list)  # chunk_idx -> [(pos, uidx), ...]
    for (c, p), uidx in locations.items():
        by_chunk[c].append((p, uidx))

    for chunk_idx in tqdm(sorted(by_chunk.keys()), desc="Pass 2a: loading ids"):
        data = torch.load(chunk_files[chunk_idx], map_location="cpu", weights_only=False)
        ids_in_chunk = data["input_ids"]
        for pos, uidx in by_chunk[chunk_idx]:
            unique_input_ids[uidx] = ids_in_chunk[pos].to(torch.long)
        del data, ids_in_chunk

    # ---- Build reverse map: unique seq -> [(latent, rank), ...] ------------
    seq_to_latents = defaultdict(list)
    for j in range(n_latents):
        for r in range(K):
            if not valid_mask[r, j]:
                continue
            c = int(topk_chunk[r, j].item())
            p = int(topk_pos[r, j].item())
            uidx = locations[(c, p)]
            seq_to_latents[uidx].append((j, r))

    # ---- Hook the target layer of the LLM ---------------------------------
    captured = {}

    def hook_fn(module, inputs, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    target_layer, _ = get_layer_module(llm, layer_idx)
    handle = target_layer.register_forward_hook(hook_fn)

    # ---- Allocate per-latent result slots ---------------------------------
    final_results = {j: [None] * K for j in range(n_latents)}

    try:
        for i in tqdm(range(0, n_unique, llm_batch_size), desc="Pass 2b: trajectories"):
            end = min(i + llm_batch_size, n_unique)
            batch_ids = unique_input_ids[i:end].to(device)
            B = batch_ids.shape[0]

            with torch.no_grad():
                _ = llm(input_ids=batch_ids)
                h = captured["h"]                              # (B, seq_len, d_model)
                flat_h = h.reshape(-1, d_model).float()        # upcast for SAE

                flat_lat_parts = []
                for si in range(0, flat_h.shape[0], sae_batch_size):
                    se = min(si + sae_batch_size, flat_h.shape[0])
                    _, lat_chunk, _ = sae(flat_h[si:se])
                    flat_lat_parts.append(lat_chunk)
                flat_lat = torch.cat(flat_lat_parts, dim=0)
                lat = flat_lat.reshape(B, seq_len, -1)          # (B, seq_len, n_latents)

            lat_cpu = lat.cpu()
            ids_cpu = batch_ids.cpu()

            for b in range(B):
                uidx = i + b
                if uidx not in seq_to_latents:
                    continue
                seq_ids = ids_cpu[b].tolist()
                text = tokenizer.decode(seq_ids, clean_up_tokenization_spaces=False)
                for latent_j, rank in seq_to_latents[uidx]:
                    trajectory = lat_cpu[b, :, latent_j].tolist()
                    final_act = float(topk_vals[rank, latent_j].item())
                    final_results[latent_j][rank] = {
                        "sequence_ids": seq_ids,
                        "sequence_text": text,
                        "final_token_activation": final_act,
                        "trajectory": trajectory,
                    }

            del h, flat_h, flat_lat, lat, lat_cpu
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        handle.remove()

    # Strip empty/None entries, drop dead latents entirely
    cleaned = {}
    for j, records in final_results.items():
        live = [r for r in records if r is not None]
        if live:
            # Sort by final-token activation descending (should already be, but belt-and-suspenders)
            live.sort(key=lambda r: r["final_token_activation"], reverse=True)
            cleaned[j] = live
    return cleaned


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Collect top-activating sequences with trajectories for each SAE latent"
    )
    ap.add_argument("--chunks_dir", type=str, required=True,
                    help="Directory containing chunk_*.pt files (e.g. data/activations/train)")
    ap.add_argument("--activations_root", type=str, default=None,
                    help="Root of activations dir containing metadata.json. "
                         "Defaults to chunks_dir's parent.")
    ap.add_argument("--sae_ckpt", type=str, required=True,
                    help="Path to trained SAE .pth")
    ap.add_argument("--out", type=str, default=None,
                    help="Output .pt path (default: <chunks_dir>/top_activating_examples.pt)")
    ap.add_argument("--top_k", type=int, default=5,
                    help="Number of top sequences to keep per latent")
    ap.add_argument("--device", type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sae_batch_size", type=int, default=2048)
    ap.add_argument("--llm_batch_size", type=int, default=32,
                    help="Batch size for re-running sequences through the LLM in pass 2")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: scan 1 chunk, first 32 latents")
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    activations_root = Path(args.activations_root) if args.activations_root else chunks_dir.parent
    out_path = Path(args.out) if args.out else chunks_dir / "top_activating_examples.pt"

    # ---- Metadata ---------------------------------------------------------
    meta_path = activations_root / "metadata.json"
    with open(meta_path) as f:
        activation_meta = json.load(f)

    seq_len = activation_meta["seq_len"]
    model_name = activation_meta["model_name"]
    layer_idx = activation_meta["layer_idx"]
    tokenizer_path = activation_meta.get("tokenizer_path", model_name)
    d_model = activation_meta["d_model"]

    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"seq_len: {seq_len}, d_model: {d_model}")

    # ---- Tokenizer --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ---- SAE --------------------------------------------------------------
    sae, cfg, sae_meta = load_sae(args.sae_ckpt, args.device)
    n_latents = cfg["n_latents"]
    print(f"SAE: n_latents={n_latents}, k={cfg['k']}, d_model={cfg['d_model']}")

    if cfg["d_model"] != d_model:
        print(f"WARNING: SAE d_model ({cfg['d_model']}) does not match "
              f"activations d_model ({d_model})")

    # Quick-mode: scan 1 chunk, first 32 latents only. Simplest way is to
    # cap the SAE's latent count for the pass-1 tracker by slicing after the
    # fact. To keep the code path consistent we actually keep all tracking
    # but only report results for the first 32 latents.
    chunk_files = sorted(chunks_dir.glob("chunk_*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.pt files in {chunks_dir}")

    if args.quick:
        chunk_files = chunk_files[:1]
        print(f"Quick mode: using 1 chunk, reporting first 32 latents")

    print(f"Chunks to scan: {len(chunk_files)}")

    # ---- Pass 1 -----------------------------------------------------------
    topk_vals, topk_chunk, topk_pos = pass1_find_topk(
        chunk_files, sae, args.top_k, args.device,
        sae_batch_size=args.sae_batch_size,
    )

    if args.quick:
        topk_vals = topk_vals[:, :32]
        topk_chunk = topk_chunk[:, :32]
        topk_pos = topk_pos[:, :32]
        n_latents_effective = 32
    else:
        n_latents_effective = n_latents

    # ---- LLM (only needed for pass 2) -------------------------------------
    print(f"Loading LLM for trajectory computation: {model_name}")
    llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(args.device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    # ---- Pass 2 -----------------------------------------------------------
    results = pass2_trajectories(
        chunk_files, topk_vals, topk_chunk, topk_pos,
        sae, llm, layer_idx, tokenizer, args.device,
        seq_len=seq_len,
        d_model=d_model,
        n_latents=n_latents_effective,
        top_k=args.top_k,
        llm_batch_size=args.llm_batch_size,
        sae_batch_size=args.sae_batch_size,
    )

    # ---- Save -------------------------------------------------------------
    out = {
        "metadata": {
            "model_name": model_name,
            "layer_idx": layer_idx,
            "seq_len": seq_len,
            "d_model": d_model,
            "n_latents": n_latents,
            "n_latents_reported": n_latents_effective,
            "top_k": args.top_k,
            "sae_ckpt": str(args.sae_ckpt),
            "sae_config": cfg,
            "chunks_dir": str(chunks_dir),
            "n_chunks_scanned": len(chunk_files),
            "n_latents_with_examples": len(results),
        },
        "latents": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"\nSaved -> {out_path}")
    print(f"  Latents with at least one example: {len(results):,} / {n_latents_effective:,}")


if __name__ == "__main__":
    main()
