#!/usr/bin/env python3
"""Equivalence test: run Gemma 2 27B via HF transformers on 2 samples.

Compare outputs to vLLM endpoint results to verify they produce identical
predictions at temperature=0.

Usage (on GPU machine):
    python equiv_test_hf.py --data samples.json --prompts prompts.json --output hf_results.json
"""
import argparse
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(system_prompt: str, user_template: str, sample: dict, no_rag_max_tokens: int = 3000) -> list:
    """Build chat messages matching the CLI pipeline."""
    # Format user prompt (no-RAG: evidence is truncated and appended)
    user_prompt = user_template.format(
        source_text=sample["source_text"],
        source_version=sample["source_version"],
        target_version=sample["target_version"],
    )
    # Truncate evidence (approximate token limit via chars, ~4 chars/token)
    evidence = sample.get("evidence", "")
    max_chars = no_rag_max_tokens * 4
    if len(evidence) > max_chars:
        evidence = evidence[:max_chars]
    user_with_evidence = f"{user_prompt}\n\nSupporting evidence:\n{evidence}"

    # Gemma doesn't support system role — merge into user
    full_user = f"{system_prompt.strip()}\n\n{user_with_evidence}"
    return [{"role": "user", "content": full_user}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to samples JSON (list of dicts)")
    parser.add_argument("--prompts", required=True, help="Path to benchmark_prompts.json")
    parser.add_argument("--output", default="hf_results.json", help="Output file")
    parser.add_argument("--model", default="google/gemma-2-27b-it", help="HF model name")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs for device_map")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max generation tokens")
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)
    with open(args.data) as f:
        samples = json.load(f)

    system_prompt = prompts["model_prompt"]["system"]
    user_template = prompts["model_prompt"]["user_template"]

    print(f"Loading {args.model} on {args.gpus} GPUs...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={i: "30GiB" for i in range(args.gpus)},
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    results = []
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}/{len(samples)}: id={sample['id']} nct={sample['instance_id']}")
        messages = build_prompt(system_prompt, user_template, sample)

        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        print(f"  Input tokens: {inputs['input_ids'].shape[1]}")
        t1 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy = deterministic
                temperature=None,
                top_p=None,
            )
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        elapsed = time.time() - t1
        print(f"  Generated {len(gen_tokens)} tokens in {elapsed:.1f}s")
        print(f"  Preview: {prediction[:150]}...")

        results.append({
            "id": sample["id"],
            "instance_id": sample["instance_id"],
            "prediction": prediction,
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": len(gen_tokens),
            "elapsed_s": round(elapsed, 1),
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
