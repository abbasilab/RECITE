#!/usr/bin/env python3
"""1:1 validation: re-run original clintriaLM code on original DB inputs,
compare outputs character-by-character.

Uses the original clintriaLM package (NOT recite) to ensure same code path.
Pulls inputs directly from the original benchmark_results.db.

Usage:
    cd /home/rro/projects/clintriaLM
    .venv/bin/python /home/rro/projects/RECITE/reproduce/validate_1to1.py \
        --model local-qwen-0_5b \
        --num-samples 5

    # Or test multiple small models:
    .venv/bin/python /home/rro/projects/RECITE/reproduce/validate_1to1.py \
        --model local-qwen-0_5b,local-qwen-3b,local-gemma2-2b \
        --num-samples 5
"""

import argparse
import json
import sqlite3
import sys
import os
from difflib import SequenceMatcher
from pathlib import Path

# Original DB config_id mapping (no_rag runs)
CONFIG_MAP = {
    "local-qwen-0_5b": "83525e11a31ca0dabb6e48c7",
    "local-qwen-3b": "1988ec418928af1290853c4a",
    "local-gemma2-2b": "9919b9efea552663a5029a76",
    "local-gemma2-9b": "be5060d19cd1f0e7424afbb0",
    "local-qwen-7b": "5bb7e5de4490ea5d71a57caa",
    "local-longctx-7b": "1fa16bf2ad873d803624dde8",
}

# Model name mapping (model_id -> HF model name from original benchmarks.yaml)
MODEL_HF_MAP = {
    "local-qwen-0_5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "local-qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "local-gemma2-2b": "google/gemma-2-2b-it",
    "local-gemma2-9b": "google/gemma-2-9b-it",
    "local-qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "local-longctx-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

CONTEXT_WINDOWS = {
    "local-qwen-0_5b": 32768,
    "local-qwen-3b": 32768,
    "local-gemma2-2b": 8192,
    "local-gemma2-9b": 8192,
    "local-qwen-7b": 32768,
    "local-longctx-7b": 128000,
}

DEFAULT_DB = "data/results/benchmark_results_cluster1.db"
DEFAULT_PROMPTS = "config/benchmark_prompts.json"


def load_samples(db_path: str, config_id: str, n: int,
                  samples_file: str | None = None) -> list[dict]:
    """Load n original samples with their predictions from DB or JSON file."""
    if samples_file:
        samples = []
        with open(samples_file) as f:
            for line in f:
                samples.append(json.loads(line))
        return samples[:n]
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT instance_id, source_version, target_version, source_text, evidence, prediction "
        f"FROM results_{config_id} "
        f"WHERE split_name='train' AND prediction IS NOT NULL AND prediction != '' "
        f"ORDER BY id LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_inference(model_name: str, context_window: int, prompts_file: str,
                  sample: dict) -> str:
    """Run inference using the original clintriaLM evaluator code."""
    # Import from original clintriaLM package
    from clintriaLM.benchmark.evaluator import (
        _get_python_gpu_model,
        _format_model_prompt,
        BenchmarkPrompts,
    )
    import torch

    # Load prompts — filter to fields BenchmarkPrompts accepts
    with open(prompts_file) as f:
        prompts_data = json.load(f)
    import inspect
    valid_keys = set(inspect.signature(BenchmarkPrompts.__init__).parameters) - {"self"}
    prompts = BenchmarkPrompts(**{k: v for k, v in prompts_data.items() if k in valid_keys})

    hf_model, tokenizer = _get_python_gpu_model(model_name, "cuda")

    ctx_tokens = context_window
    no_rag_max_tokens = max(0, ctx_tokens - 4096)
    if no_rag_max_tokens <= 0:
        no_rag_max_tokens = 512
    no_rag_max_tokens = min(no_rag_max_tokens, max(256, ctx_tokens - 2048 - 512))

    source_text = sample["source_text"]
    evidence = sample["evidence"]
    source_version = sample["source_version"]
    target_version = sample["target_version"]

    has_document = evidence is not None and bool(evidence.strip())
    user_prompt = _format_model_prompt(
        source_text, prompts.model_prompt,
        has_document=has_document,
        source_version=source_version, target_version=target_version,
    )
    system_prompt = prompts.model_prompt.get("system", "")

    if evidence and evidence.strip():
        ev = evidence.strip()
        enc = tokenizer.encode(ev)
        if len(enc) > no_rag_max_tokens:
            ev = tokenizer.decode(enc[:no_rag_max_tokens])
        user_content = f"{user_prompt}\n\nSupporting evidence:\n{ev}"
    else:
        user_content = user_prompt

    system_prompt = (system_prompt or "").strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    max_len = min(tokenizer.model_max_length or 131072, ctx_tokens)

    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", truncation=True, max_length=max_len,
        )
    except Exception as e:
        if "system" in str(e).lower():
            messages_no_sys = [
                {"role": "user", "content": f"{system_prompt}\n\n{user_content}".strip()},
            ]
            prompt_ids = tokenizer.apply_chat_template(
                messages_no_sys, add_generation_prompt=True,
                return_tensors="pt", truncation=True, max_length=max_len,
            )
        else:
            raise

    # Handle BatchEncoding (newer transformers)
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]

    if prompt_ids.shape[1] > max_len:
        prompt_ids = prompt_ids[:, -max_len:]

    prompt_ids = prompt_ids.to(hf_model.device)
    attention_mask = prompt_ids.new_ones(prompt_ids.shape, dtype=torch.long)

    with torch.no_grad():
        output = hf_model.generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, prompt_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="local-qwen-0_5b",
                        help="Comma-separated model IDs")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to validate per model")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    parser.add_argument("--samples-file", default=None,
                        help="JSONL file with pre-exported samples (skip DB)")
    args = parser.parse_args()

    models = args.model.split(",")

    print("=" * 70)
    print("1:1 Validation — Original clintriaLM Code vs Original DB Outputs")
    print("=" * 70)

    import torch
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}, "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print()

    all_results = {}

    for model_id in models:
        model_id = model_id.strip()
        config_id = CONFIG_MAP.get(model_id)
        hf_name = MODEL_HF_MAP.get(model_id)
        ctx = CONTEXT_WINDOWS.get(model_id, 32768)

        if not config_id or not hf_name:
            print(f"[SKIP] {model_id}: unknown model")
            continue

        print(f"--- {model_id} ({hf_name}) ---")
        samples = load_samples(args.db, config_id, args.num_samples,
                               samples_file=args.samples_file)
        print(f"Loaded {len(samples)} samples from DB")

        exact = 0
        sims = []

        for i, sample in enumerate(samples):
            nct = sample["instance_id"]
            vf, vt = sample["source_version"], sample["target_version"]
            orig = sample["prediction"]

            print(f"  [{i+1}/{len(samples)}] {nct} v{vf}→v{vt}...", end=" ", flush=True)
            repro = run_inference(hf_name, ctx, args.prompts, sample)

            if orig.strip() == repro.strip():
                exact += 1
                sim = 1.0
                print("EXACT MATCH")
            else:
                sim = SequenceMatcher(None, orig, repro).ratio()
                print(f"DIFF (sim={sim:.3f})")
                if sim < 0.95:
                    print(f"    ORIG: {orig[:120]}...")
                    print(f"    REPR: {repro[:120]}...")

            sims.append(sim)

        avg_sim = sum(sims) / len(sims) if sims else 0
        all_results[model_id] = {
            "total": len(samples), "exact": exact,
            "exact_pct": exact / len(samples) * 100 if samples else 0,
            "avg_sim": avg_sim,
        }

        print(f"  Result: {exact}/{len(samples)} exact ({exact/len(samples)*100:.0f}%), avg_sim={avg_sim:.3f}")

        # Clear model from GPU before loading next
        from clintriaLM.benchmark.evaluator import clear_python_gpu_cache
        clear_python_gpu_cache()
        torch.cuda.empty_cache()
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print(f"{'Model':<20} {'Total':>6} {'Exact':>6} {'Match%':>8} {'AvgSim':>8}")
    print("-" * 70)
    for m, r in all_results.items():
        print(f"{m:<20} {r['total']:>6} {r['exact']:>6} {r['exact_pct']:>7.1f}% {r['avg_sim']:>7.3f}")
    print("=" * 70)

    all_pass = all(r["exact_pct"] == 100.0 for r in all_results.values())
    if all_pass:
        print("\nALL MODELS PASS — outputs match original DB exactly.")
    else:
        print("\nSOME DIFFERENCES FOUND — see details above.")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
