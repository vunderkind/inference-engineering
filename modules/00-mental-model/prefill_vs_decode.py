"""
Module 0 lab: prefill vs decode.

Generation has two phases, and this script times them separately:

  prefill  One forward pass over the whole prompt. Produces the first
           output token and fills the KV cache.
  decode   One forward pass per new token. Each pass feeds in a single
           token and reuses the KV cache from earlier passes.

It sweeps prompt length and reports, for each length:
  - prefill time, and prefill cost per prompt token
  - median decode time per generated token
  - achieved compute during prefill (TFLOPS)
  - achieved memory bandwidth during decode (GB/s)

Run from the repo root:
  uv run python modules/00-mental-model/prefill_vs_decode.py
"""

import argparse
import csv
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Any ordinary English text works. We only need a lot of real tokens.
FILLER = (
    "The river town woke slowly. Fishermen checked their nets, traders "
    "argued over the price of yams, and children ran between the stalls "
    "carrying messages nobody had asked them to carry. By noon the heat "
    "settled over everything, and the market went quiet until evening. "
)


def sync(device: str) -> None:
    # GPU work is asynchronous: Python queues the kernels and moves on.
    # Without this call, the timer would measure how long it took to
    # *queue* the work, not how long the GPU took to *do* it.
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def time_one(model, prompt_ids, new_tokens: int, device: str):
    # ---- Prefill: the whole prompt in one forward pass ----
    sync(device)
    t0 = time.perf_counter()
    # logits_to_keep=1: only compute logits for the last position.
    # Without it, a 4096-token prompt would produce a
    # 4096 x 151,936 logits tensor (~1.2 GB in bf16) that we throw away.
    out = model(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
    next_id = out.logits[:, -1:].argmax(dim=-1)  # greedy pick
    sync(device)
    prefill_s = time.perf_counter() - t0

    # ---- Decode: one token per forward pass ----
    past = out.past_key_values  # the KV cache prefill just built
    step_s = []
    for _ in range(new_tokens):
        t = time.perf_counter()
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(dim=-1)
        sync(device)
        step_s.append(time.perf_counter() - t)

    del past, out
    if device == "mps":
        torch.mps.empty_cache()
    return prefill_s, step_s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(device).eval()

    # model.parameters() counts tied weights once, so this is the
    # number of bytes that actually sit in memory.
    n_params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"device={device}  params={n_params / 1e9:.2f}B  "
          f"weights={weight_bytes / 1e9:.2f} GB")

    # Build a token stream long enough for the longest prompt.
    text = FILLER
    while len(tok(text).input_ids) < max(args.lengths):
        text = text * 2
    ids = tok(text, return_tensors="pt").input_ids[0]

    # Warm-up: the first run on MPS compiles kernels and is slow.
    # We don't want that cost in the measurements.
    time_one(model, ids[:64].unsqueeze(0).to(device), 4, device)

    header = (f"{'prompt':>7} {'prefill_s':>10} {'prefill_ms/tok':>15} "
              f"{'decode_ms/tok':>14} {'ratio':>7} {'prefill_TFLOPS':>15} "
              f"{'decode_GB/s':>12}")
    print(header)
    print("-" * len(header))

    rows = []
    for n in args.lengths:
        prompt = ids[:n].unsqueeze(0).to(device)
        prefill_s, steps = time_one(model, prompt, args.new_tokens, device)
        decode_s = statistics.median(steps)

        prefill_per_tok = prefill_s / n
        row = {
            "prompt_tokens": n,
            "prefill_s": prefill_s,
            "prefill_ms_per_tok": prefill_per_tok * 1e3,
            "decode_ms_per_tok": decode_s * 1e3,
            # How many times more expensive one decode token is than
            # one prefill token.
            "ratio": decode_s / prefill_per_tok,
            # ~2 FLOPs (one multiply, one add) per parameter per token.
            # Ignores attention's own FLOPs, which grow with length.
            "prefill_tflops": 2 * n_params * n / prefill_s / 1e12,
            # Each decode step must read every weight once.
            # Ignores the KV cache reads, which also grow with length.
            "decode_gbps": weight_bytes / decode_s / 1e9,
        }
        rows.append(row)
        print(f"{n:>7} {prefill_s:>10.3f} {row['prefill_ms_per_tok']:>15.2f} "
              f"{row['decode_ms_per_tok']:>14.1f} {row['ratio']:>7.1f} "
              f"{row['prefill_tflops']:>15.2f} {row['decode_gbps']:>12.1f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()