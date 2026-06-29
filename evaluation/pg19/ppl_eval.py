import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

import argparse
from argparse import ArgumentParser

device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--fixed-length", type=int)
parser.add_argument("--max-tokens", type=int, default=8192)
parser.add_argument("--min-tokens", type=int, default=256)
parser.add_argument("--tokens-step", type=int)
parser.add_argument("--length-step", type=int, default=128)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--output_dir", type=str)

parser.add_argument("--num_eval_tokens", type=int, default=None)

parser.add_argument("--quest", action="store_true", help="Enable quest attention")
parser.add_argument("--token_budget", type=int, default=1024)
parser.add_argument("--chunk_size", type=int, default=16)

# MPR options
parser.add_argument("--use_mpr", action="store_true",
                    help="Enable MPR (Mixed-Precision Recovery) mode using Quest LlamaForCausalLM.")
parser.add_argument("--gpu_capacity", type=int, default=None,
                    help="GPU page capacity for MPR (pages). Default: max_tokens // chunk_size.")
parser.add_argument("--tier_high", type=float, default=10.0,
                    help="FP16 recall score threshold (MPR).")
parser.add_argument("--tier_mid", type=float, default=3.0,
                    help="INT8 recall score threshold (MPR).")
parser.add_argument("--tier_low", type=float, default=0.5,
                    help="INT4 recall score threshold (MPR).")


def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


args = parser.parse_args()

data = load_dataset("emozilla/pg19-test", split="test")

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")

if args.use_mpr:
    # MPR path: use Quest's LlamaForCausalLM with quest_mpr_init.
    # PPL is computed via chunked prefill — the model returns logits for all
    # positions in the chunk, so we don't need the past_key_values loop.
    from quest import LlamaForCausalLM as QuestLlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = QuestLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device,
        torch_dtype=torch.float16,
    )
    page_size = args.chunk_size
    max_seq = args.max_tokens + 512
    gpu_capacity = args.gpu_capacity or (max_seq // page_size)
    model.quest_mpr_init(
        page_size=page_size,
        gpu_capacity=gpu_capacity,
        max_seq_len=max_seq,
        token_budget=args.token_budget,
        tier_high=args.tier_high,
        tier_mid=args.tier_mid,
        tier_low=args.tier_low,
        dtype=torch.float16,
        device=torch.device(device),
    )
    model.eval()
    past_key_values = None
    print(f"MPR mode: gpu_capacity={gpu_capacity} pages, max_seq={max_seq}")

    def ppl_forward_step(input_ids, past_key_values):
        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=True)
        return output.logits, None  # Quest manages KV cache internally

    use_quest_forward = True
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model, tokenizer = load(args.model_name_or_path)
    past_key_values = None
    use_quest_forward = False

    if args.quest:
        print("Enable quest attention")
        from evaluation.quest_attention import enable_quest_attention_eval
        enable_quest_attention_eval(model, args)

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")

num_eval_tokens = 0
for text in data["text"][:1]:
    encodings = tokenizer(text, return_tensors="pt")

    print(encodings.input_ids[:, :10])

    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")

    if use_quest_forward:
        # MPR / Quest path: token-by-token forward reusing Quest's internal KV cache.
        model.quest_clear()
        pbar = tqdm(range(0, seq_len - 1))
        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
            logits, _ = ppl_forward_step(input_ids, None)
            logits = logits.view(-1, model.config.vocab_size)
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
            nlls.append(neg_log_likelihood)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )
            print(neg_log_likelihood.item(), file=f, flush=True)
            num_eval_tokens += 1
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
    else:
        # Standard HF path.
        pbar = tqdm(range(0, seq_len - 1))
        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
            nlls.append(neg_log_likelihood)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )
            print(neg_log_likelihood.item(), file=f, flush=True)
            num_eval_tokens += 1
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break

    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")
