# Based on Punica Project
# Check: https://github.com/efeslab/Atom/blob/main/e2e/punica-atom/benchmarks/bench_textgen.py

import argparse
import dataclasses
import time
import numpy as np
import torch
from tqdm.auto import tqdm

from quest import LlamaForCausalLM


@dataclasses.dataclass
class ModelConfig:
    model_path: str
    dtype: str = dataclasses.field(default="float16")
    device: str = dataclasses.field(default="cuda:0")


MODEL_CFGS = {
    "llama2-7b": ModelConfig(model_path="meta-llama/Llama-2-7b-chat-hf"),
    "llama3-8b": ModelConfig(model_path="meta-llama/Meta-Llama-3-8B-Instruct"),
}


def load_model(model_cfg: ModelConfig):
    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    torch.set_default_dtype(dtype)
    with device:
        model = LlamaForCausalLM.from_pretrained(
            model_cfg.model_path,
            device_map=device,
            torch_dtype=dtype,
        )
    return model


@torch.inference_mode()
def benchmark_quest():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CFGS.keys(), default="llama2-7b")
    parser.add_argument("--context_len", type=int, default=2 * 1024)
    parser.add_argument("--decode_len", type=int, default=256)
    parser.add_argument("--page_size", type=int, default=16)
    parser.add_argument("--token_budget", type=int, default=256)
    parser.add_argument("--iteration", type=int, default=10)
    # MPR options
    parser.add_argument("--use_mpr", action="store_true",
                        help="Enable MPR (Mixed-Precision Recovery) mode.")
    parser.add_argument("--gpu_capacity", type=int, default=None,
                        help="GPU page capacity for MPR (pages). Default: max_seq_len // page_size.")
    parser.add_argument("--tier_high", type=float, default=10.0,
                        help="Score threshold for FP16 recall (MPR).")
    parser.add_argument("--tier_mid", type=float, default=3.0,
                        help="Score threshold for INT8 recall (MPR).")
    parser.add_argument("--tier_low", type=float, default=0.5,
                        help="Score threshold for INT4 recall (MPR).")
    parser.add_argument("--breakdown", action="store_true",
                        help="Print per-step recall/evict latency breakdown (MPR only).")
    args = parser.parse_args()

    assert args.model in MODEL_CFGS, f"Model {args.model} not found in MODEL_CFGS"
    model_cfg = MODEL_CFGS[args.model]

    max_seq_len = args.context_len + args.decode_len + 512
    page_size = args.page_size
    token_budget = args.token_budget
    context_len = args.context_len
    decode_len = args.decode_len

    model = load_model(model_cfg)

    dtype = getattr(torch, model_cfg.dtype)
    device = torch.device(model_cfg.device)

    if args.use_mpr:
        gpu_capacity = args.gpu_capacity or (max_seq_len // page_size)
        model.quest_mpr_init(
            page_size=page_size,
            gpu_capacity=gpu_capacity,
            max_seq_len=max_seq_len,
            token_budget=token_budget,
            tier_high=args.tier_high,
            tier_mid=args.tier_mid,
            tier_low=args.tier_low,
            dtype=dtype,
            device=device,
        )
        print(f"Mode: Quest-MPR  gpu_capacity={gpu_capacity} pages")
    else:
        model.quest_init(
            page_size=page_size,
            max_seq_len=max_seq_len,
            token_budget=token_budget,
            dtype=dtype,
            device=device,
        )
        print("Mode: Quest (standard)")

    hidden_size = model._config.hidden_size

    prefill_latency = []
    decode_latency = []

    for _ in tqdm(range(args.iteration)):
        torch.cuda.empty_cache()

        if args.use_mpr and args.breakdown:
            model.model.iController.clear_timing()

        # Prefill
        ts = time.perf_counter()
        hidden_states = torch.randn(1, context_len, hidden_size, dtype=dtype, device=device)
        model(inputs_embeds=hidden_states)
        te = time.perf_counter()
        prefill_latency.append(te - ts)

        # Decode
        for _ in range(decode_len):
            ts = time.perf_counter()
            hidden_states = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
            model(inputs_embeds=hidden_states)
            te = time.perf_counter()
            decode_latency.append(te - ts)

        model.quest_clear()

    avg_prefill = np.mean(prefill_latency)
    avg_decode = np.mean(decode_latency)
    throughput = 1.0 / avg_decode  # tokens/second

    if args.use_mpr:
        header = ("mode,page_size,token_budget,gpu_capacity,context_len,decode_len,"
                  "avg_prefill_latency,avg_decode_latency,throughput_tok_per_s")
        row = (f"mpr,{page_size},{token_budget},{gpu_capacity},{context_len},{decode_len},"
               f"{avg_prefill:.6f},{avg_decode:.6f},{throughput:.2f}")
    else:
        header = ("mode,page_size,token_budget,context_len,decode_len,"
                  "avg_prefill_latency,avg_decode_latency,throughput_tok_per_s")
        row = (f"quest,{page_size},{token_budget},{context_len},{decode_len},"
               f"{avg_prefill:.6f},{avg_decode:.6f},{throughput:.2f}")

    print(header)
    print(row)

    if args.use_mpr and args.breakdown:
        t = model.model.iController.timing_summary()
        per_step_recall = t["recall_wall_seconds"] / max(1, decode_len * args.iteration)
        per_step_evict = t["evict_wall_seconds"] / max(1, decode_len * args.iteration)
        print(f"\nMPR timing breakdown (per decode step average):")
        print(f"  recall  : {per_step_recall * 1e3:.3f} ms")
        print(f"  evict   : {per_step_evict * 1e3:.3f} ms")
        backup_stats = model.model.iController.backup_store.stats()
        print(f"  CPU backup size: {backup_stats.total_actual_backup_bytes / 1e6:.1f} MB  "
              f"(put_count={backup_stats.put_count})")


if __name__ == "__main__":
    benchmark_quest()

# nsys profile --delay 20 --duration 1 --output "$(env TZ='US/Pacific' date +%Y%m%d-%H%M%S).nsys-rep" python bench_textgen.py
