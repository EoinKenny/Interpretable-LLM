"""
Collect residual stream activations AND input_ids from a frozen LLM for SAE
training + later interpretability analysis.

Streams text from Pile .jsonl.zst files, hooks into a specified layer, and
saves (final_token_activation, input_ids) pairs to disk in chunks. Keeping
the input_ids is what lets us later recover the text that produced each
activation — essential for "top activating examples" analysis.

Only sequences with >= seq_len tokens are kept (no padding). We always
extract the final token's activation, but store the full input_ids so that
we can re-run the sequence later to compute activation trajectories.

Splits into 95% train / 5% test, seeded for reproducibility.

Usage:
    # Single GPU
    python collect_activations.py --model meta-llama/Llama-3.2-1B --layer_idx 8

    # Multi-GPU
    accelerate launch --num_processes 4 collect_activations.py \\
        --model meta-llama/Llama-3.2-1B --layer_idx 8
"""

import argparse
import io
import json
import os
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed
import torch.nn as nn
import zstandard as zstd
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


# ---------------------------------------------------------------------------
#                               DATASET
# ---------------------------------------------------------------------------

class StreamingPileDataset(IterableDataset):
    """
    Iterates over .jsonl.zst files in a directory, streaming text.
    Handles sharding for both multi-GPU and multi-worker.

    Only yields sequences with exactly seq_len tokens (no padding).
    Sequences shorter than seq_len are skipped entirely.
    """
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        seq_len: int = 64,
        file_pattern: str = "*.jsonl.zst",
        process_rank: int = 0,
        num_processes: int = 1,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir.absolute()}"
            )
        self.files = sorted(list(self.data_dir.rglob(file_pattern)))
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        if not self.files:
            raise FileNotFoundError(
                f"No files matching '{file_pattern}' in {self.data_dir.absolute()}"
            )

    def __iter__(self) -> Iterator[torch.Tensor]:
        # Shard files across GPU processes
        files_for_this_process = self.files[self.process_rank::self.num_processes]

        # Further shard across DataLoader workers within this process
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files_to_read = files_for_this_process[
                worker_info.id::worker_info.num_workers
            ]
        else:
            files_to_read = files_for_this_process

        for file_path in files_to_read:
            try:
                with open(file_path, "rb") as fh:
                    dctx = zstd.ZstdDecompressor(max_window_size=2_147_483_648)
                    with dctx.stream_reader(fh) as reader:
                        text_reader = io.TextIOWrapper(reader, encoding="utf-8")
                        for line in text_reader:
                            try:
                                record = json.loads(line)
                                text = record.get("text", "")
                                enc = self.tokenizer(
                                    text,
                                    max_length=self.seq_len,
                                    truncation=True,
                                    return_tensors="pt",
                                )
                                input_ids = enc["input_ids"].squeeze(0)
                                if input_ids.shape[0] < self.seq_len:
                                    continue
                                yield input_ids
                            except (json.JSONDecodeError, Exception):
                                continue
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue


# ---------------------------------------------------------------------------
#                         LAYER HOOK
# ---------------------------------------------------------------------------

def get_layer_module(model, layer_idx: int):
    """Get the transformer layer module. Supports LLaMA, GPT-2, Pythia."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = model.gpt_neox.layers
    else:
        raise ValueError(f"Unknown architecture: {type(model)}")
    return layers[layer_idx], len(layers)


class ActivationHook:
    def __init__(self):
        self.output = None
        self.hook = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self.output = output[0].detach()
        else:
            self.output = output.detach()

    def attach(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self.hook:
            self.hook.remove()


# ---------------------------------------------------------------------------
#                               MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect LLM activations + input_ids for SAE training"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--data_dir", default="datasets/pile-uncopyrighted/train")
    parser.add_argument("--output_dir", default="data/activations")
    parser.add_argument(
        "--layer_idx", type=int, default=-1,
        help="Layer to extract from. -1 = middle layer.",
    )
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--n_sequences", type=int, default=50_000_000,
        help="Total sequences to collect (one final-token activation each)",
    )
    parser.add_argument(
        "--seqs_per_chunk", type=int, default=500_000,
        help="Sequences per saved chunk",
    )
    parser.add_argument(
        "--test_fraction", type=float, default=0.05,
        help="Fraction of activations to hold out for testing",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/test split reproducibility",
    )
    args = parser.parse_args()

    n_sequences = args.n_sequences

    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"GPUs: {accelerator.num_processes}, Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model (frozen)
    if accelerator.is_main_process:
        print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Determine layer
    layer_module, n_layers = get_layer_module(model, 0)
    if args.layer_idx == -1:
        args.layer_idx = n_layers // 2
    layer_module, _ = get_layer_module(model, args.layer_idx)

    d_model = model.config.hidden_size

    if accelerator.is_main_process:
        print(f"Hooking layer {args.layer_idx}/{n_layers - 1}, d_model={d_model}")
        print(f"Seq len: {args.seq_len}")
        print(f"Train/test split: {1 - args.test_fraction:.0%} / {args.test_fraction:.0%}")
        print(f"Target: {n_sequences:,} sequences")

    hook = ActivationHook().attach(layer_module)

    # Dataset — sharding is handled manually, so do NOT call
    # accelerator.prepare(dataloader), which would double-shard.
    try:
        dataset = StreamingPileDataset(
            args.data_dir,
            tokenizer,
            seq_len=args.seq_len,
            process_rank=accelerator.process_index,
            num_processes=accelerator.num_processes,
        )
    except FileNotFoundError as e:
        print(f"CRITICAL: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Output directories
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    if accelerator.is_main_process:
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_name": args.model,
            "tokenizer_path": args.model,
            "layer_idx": args.layer_idx,
            "n_layers": n_layers,
            "d_model": d_model,
            "seq_len": args.seq_len,
            "final_token_only": True,
            "stores_input_ids": True,
            "n_sequences_requested": n_sequences,
            "test_fraction": args.test_fraction,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    accelerator.wait_for_everyone()

    # RNG for train/test split (main process, seeded)
    split_rng = torch.Generator().manual_seed(args.seed)

    # Buffers now hold both activations and ids
    train_acts_buf, train_ids_buf = [], []
    test_acts_buf, test_ids_buf = [], []
    total_seqs = 0
    train_chunk_idx = 0
    test_chunk_idx = 0
    t_start = time.time()

    pbar = None
    if accelerator.is_main_process:
        pbar = tqdm(total=n_sequences, desc="Collecting", unit="seq")

    data_iter = iter(dataloader)

    with torch.no_grad():
        while total_seqs < n_sequences:
            # --- Deadlock-safe batch fetch ---
            try:
                batch = next(data_iter)
                has_data = torch.tensor([1], device=device, dtype=torch.long)
            except StopIteration:
                has_data = torch.tensor([0], device=device, dtype=torch.long)
                batch = None

            all_have_data = accelerator.gather(has_data)
            if all_have_data.sum().item() < accelerator.num_processes:
                if accelerator.is_main_process:
                    tqdm.write(
                        f"Data exhausted on "
                        f"{accelerator.num_processes - all_have_data.sum().item()} "
                        f"GPU(s) — stopping collection."
                    )
                break

            input_ids = batch.to(device)  # (B, seq_len)
            _ = model(input_ids=input_ids)
            acts = hook.output  # (B, seq_len, d_model)

            final_acts = acts[:, -1, :]  # (B, d_model)

            # Gather activations AND input_ids from all GPUs
            gathered_acts = accelerator.gather(final_acts)        # (B*num_gpus, d_model)
            gathered_ids = accelerator.gather(input_ids)          # (B*num_gpus, seq_len)

            if accelerator.is_main_process:
                acts_cpu = gathered_acts.cpu().float()
                ids_cpu = gathered_ids.cpu().to(torch.int32)

                # Train/test split (row-level)
                n = acts_cpu.shape[0]
                mask = torch.rand(n, generator=split_rng)
                test_mask = mask < args.test_fraction
                train_mask = ~test_mask

                if train_mask.any():
                    train_acts_buf.append(acts_cpu[train_mask])
                    train_ids_buf.append(ids_cpu[train_mask])
                if test_mask.any():
                    test_acts_buf.append(acts_cpu[test_mask])
                    test_ids_buf.append(ids_cpu[test_mask])

            total_seqs += gathered_acts.shape[0]

            if accelerator.is_main_process:
                pbar.update(gathered_acts.shape[0])

            # --- Synchronized chunk saving ---
            if accelerator.is_main_process:
                train_buf_size = sum(t.shape[0] for t in train_acts_buf)
                test_buf_size = sum(t.shape[0] for t in test_acts_buf)
                need_save = int(
                    train_buf_size >= args.seqs_per_chunk
                    or test_buf_size >= args.seqs_per_chunk
                )
            else:
                need_save = 0

            should_save = torch.tensor([need_save], device=device, dtype=torch.long)
            torch.distributed.broadcast(should_save, src=0)

            if should_save.item():
                if accelerator.is_main_process:
                    # Flush train buffer if large enough
                    if train_buf_size >= args.seqs_per_chunk:
                        acts_tensor = torch.cat(train_acts_buf, dim=0)
                        ids_tensor = torch.cat(train_ids_buf, dim=0)
                        chunk_path = train_dir / f"chunk_{train_chunk_idx:04d}.pt"
                        torch.save({
                            "final_token_activations": acts_tensor,
                            "input_ids": ids_tensor,
                        }, chunk_path)
                        elapsed = time.time() - t_start
                        tqdm.write(
                            f"Saved train/{chunk_path.name}: "
                            f"{acts_tensor.shape[0]:,} activations "
                            f"({total_seqs:,} total, {total_seqs/elapsed:.0f} seq/s)"
                        )
                        train_chunk_idx += 1
                        train_acts_buf, train_ids_buf = [], []

                    # Flush test buffer if large enough
                    if test_buf_size >= args.seqs_per_chunk:
                        acts_tensor = torch.cat(test_acts_buf, dim=0)
                        ids_tensor = torch.cat(test_ids_buf, dim=0)
                        chunk_path = test_dir / f"chunk_{test_chunk_idx:04d}.pt"
                        torch.save({
                            "final_token_activations": acts_tensor,
                            "input_ids": ids_tensor,
                        }, chunk_path)
                        tqdm.write(
                            f"Saved test/{chunk_path.name}: "
                            f"{acts_tensor.shape[0]:,} activations"
                        )
                        test_chunk_idx += 1
                        test_acts_buf, test_ids_buf = [], []

                accelerator.wait_for_everyone()

    # Final flush
    if accelerator.is_main_process:
        if train_acts_buf:
            acts_tensor = torch.cat(train_acts_buf, dim=0)
            ids_tensor = torch.cat(train_ids_buf, dim=0)
            chunk_path = train_dir / f"chunk_{train_chunk_idx:04d}.pt"
            torch.save({
                "final_token_activations": acts_tensor,
                "input_ids": ids_tensor,
            }, chunk_path)
            print(f"Saved final train/{chunk_path.name}: {acts_tensor.shape[0]:,}")
            train_chunk_idx += 1

        if test_acts_buf:
            acts_tensor = torch.cat(test_acts_buf, dim=0)
            ids_tensor = torch.cat(test_ids_buf, dim=0)
            chunk_path = test_dir / f"chunk_{test_chunk_idx:04d}.pt"
            torch.save({
                "final_token_activations": acts_tensor,
                "input_ids": ids_tensor,
            }, chunk_path)
            print(f"Saved final test/{chunk_path.name}: {acts_tensor.shape[0]:,}")
            test_chunk_idx += 1

    hook.remove()

    if accelerator.is_main_process:
        if pbar is not None:
            pbar.close()

        elapsed = time.time() - t_start
        print(f"\nDone: {total_seqs:,} sequences")
        print(f"  Train chunks: {train_chunk_idx}, Test chunks: {test_chunk_idx}")
        print(f"  {elapsed:.1f}s ({total_seqs / elapsed:.0f} seq/s)")
        print(f"  Saved to {output_dir}")

        # Update metadata
        meta["n_sequences"] = total_seqs
        meta["n_train_chunks"] = train_chunk_idx
        meta["n_test_chunks"] = test_chunk_idx
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
