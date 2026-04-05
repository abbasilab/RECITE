"""
Standalone HuggingFace transformers evaluation script for large models.
Designed for multi-GPU inference without vLLM.

Usage:
    python3 run_hf_eval.py \
        --model Qwen/Qwen2.5-72B-Instruct-AWQ \
        --parquet data/benchmark_splits/benchmark.parquet \
        --output data/rebuttal/qwen25_72b_no_rag.json \
        --checkpoint-dir /tmp/qwen25_72b_ckpt
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Project root and prompts ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROMPTS = json.loads((ROOT / "config" / "benchmark_prompts.json").read_text())
MODEL_SYSTEM = PROMPTS["model_prompt"]["system"]
MODEL_USER_NORAG = PROMPTS["model_prompt"]["user_template"]


# ── Telegram notifications ────────────────────────────────────────────────
TELEGRAM_URL = os.environ.get("TELEGRAM_BRIDGE", "http://localhost:8443/api/send")
TELEGRAM_PREFIX = "[rebuttal-largemodel-gpu2 — clintrialm]"


def notify(msg: str):
    """Best-effort Telegram notification."""
    try:
        import urllib.request
        data = json.dumps({"message": f"{TELEGRAM_PREFIX} {msg}"}).encode()
        req = urllib.request.Request(
            TELEGRAM_URL, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # fire and forget


# ── Prompt building (mirrors rebuttal_eval_endpoint.py) ──────────────────
def build_prompt(row: dict, max_evidence_chars: int = 20000) -> tuple:
    """Build (system, user) prompt for a benchmark sample."""
    source = str(row.get("source_text", "") or "")
    evidence = str(row.get("evidence", "") or "")
    vf = row.get("source_version")
    vt = row.get("target_version")

    try:
        vf = int(float(vf)) if vf is not None and not pd.isna(vf) else None
    except (ValueError, TypeError):
        vf = None
    try:
        vt = int(float(vt)) if vt is not None and not pd.isna(vt) else None
    except (ValueError, TypeError):
        vt = None

    kwargs = {"source_text": source}
    if vf is not None:
        kwargs["source_version"] = vf
    if vt is not None:
        kwargs["target_version"] = vt

    has_doc = bool(evidence.strip())
    if has_doc and len(evidence) > max_evidence_chars:
        evidence = evidence[:max_evidence_chars]

    user_prompt = MODEL_USER_NORAG.format(**kwargs)
    if has_doc:
        user_prompt += f"\n\nSupporting evidence:\n{evidence}"

    return MODEL_SYSTEM, user_prompt


# ── Metrics (self-contained, mirrors evaluator.py) ───────────────────────
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def lcs_length(seq1: list, seq2: list) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(reference: str, candidate: str) -> float:
    ref_words = reference.split()
    cand_words = candidate.split()
    if not ref_words or not cand_words:
        return 0.0
    lcs = lcs_length(ref_words, cand_words)
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand_words)
    recall = lcs / len(ref_words)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(ground_truth: str, prediction: str) -> dict:
    """Compute edit_similarity, BLEU, ROUGE-L."""
    metrics = {}

    # Binary exact match
    metrics["binary_correct"] = 1.0 if ground_truth.strip() == prediction.strip() else 0.0

    # Edit distance
    ed = levenshtein_distance(ground_truth, prediction)
    max_len = max(len(ground_truth), len(prediction))
    metrics["edit_distance"] = ed
    metrics["normalized_edit_distance"] = ed / max_len if max_len > 0 else 1.0
    metrics["edit_similarity"] = 1.0 - metrics["normalized_edit_distance"]

    # BLEU
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        metrics["bleu"] = float(sentence_bleu(
            [ground_truth.split()], prediction.split(), smoothing_function=smooth
        ))
    except Exception:
        metrics["bleu"] = 0.0

    # ROUGE-L
    metrics["rouge_l"] = rouge_l(ground_truth, prediction)

    return metrics


# ── Model loading ─────────────────────────────────────────────────────────
def load_model(model_name: str, cache_dir: Optional[str] = None):
    """Load model with device_map=auto across all GPUs."""
    print(f"Loading model: {model_name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    model.eval()

    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.1f}s")
    print(f"  Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")

    return model, tokenizer


def generate_prediction(
    model, tokenizer, system_prompt: str, user_prompt: str,
    max_new_tokens: int = 2048, temperature: float = 0.0,
) -> str:
    """Generate a single prediction using chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=30000)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Checkpoint I/O ────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: Path) -> list:
    if ckpt_path.exists():
        return json.loads(ckpt_path.read_text())
    return []


def save_checkpoint(results: list, ckpt_path: Path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps(results, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HuggingFace transformers eval")
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--parquet", default=str(ROOT / "data/benchmark_splits/benchmark.parquet"))
    parser.add_argument("--output", default=str(ROOT / "data/rebuttal/qwen25_72b_no_rag.json"))
    parser.add_argument("--checkpoint-dir", default="/tmp/qwen25_72b_ckpt")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-evidence-chars", type=int, default=20000)
    parser.add_argument("--cache-dir", default=None, help="HF model cache dir")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    label = args.model_label or args.model.split("/")[-1]
    ckpt_path = Path(args.checkpoint_dir) / "results_checkpoint.json"

    # Load benchmark
    print(f"Loading benchmark from {args.parquet}")
    df = pd.read_parquet(args.parquet)
    samples = df.to_dict("records")
    if args.limit:
        samples = samples[:args.limit]
    print(f"  {len(samples)} samples")

    # Load checkpoint
    results = load_checkpoint(ckpt_path)
    done_ids = {r["id"] for r in results}
    remaining = [s for s in samples if int(float(s["id"])) not in done_ids]
    print(f"  Checkpoint: {len(results)} done, {len(remaining)} remaining")

    if not remaining:
        print("All samples already completed!")
    else:
        # Load model
        model, tokenizer = load_model(args.model, cache_dir=args.cache_dir)
        notify(f"Model loaded: {label}. Starting inference on {len(remaining)} samples (8x RTX 5000 Ada).")

        t_start = time.time()
        for i, sample in enumerate(remaining):
            sample_id = int(float(sample["id"]))
            instance_id = sample.get("instance_id", "")

            try:
                sys_prompt, user_prompt = build_prompt(sample, max_evidence_chars=args.max_evidence_chars)
                t0 = time.time()
                prediction = generate_prediction(
                    model, tokenizer, sys_prompt, user_prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                gen_time = time.time() - t0
                is_error = False
            except Exception as e:
                prediction = f"[ERROR] {type(e).__name__}: {e}"
                gen_time = 0
                is_error = True

            result = {
                "id": sample_id,
                "instance_id": instance_id,
                "prediction": prediction,
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "ground_truth": str(sample.get("reference_text", "") or ""),
                "is_error": is_error,
            }
            results.append(result)

            # Progress
            total_done = len(results)
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_s = (len(remaining) - i - 1) / rate if rate > 0 else 0
            eta_m = eta_s / 60

            if (i + 1) % 10 == 0:
                print(f"  [{total_done}/{len(samples)}] id={sample_id} "
                      f"gen={gen_time:.1f}s rate={rate:.2f}/s "
                      f"ETA={eta_m:.0f}min err={is_error}")

            # Checkpoint every 100
            if (i + 1) % 100 == 0:
                save_checkpoint(results, ckpt_path)
                print(f"  >> Checkpoint saved ({total_done} results)")

            # Telegram every 500
            if (i + 1) % 500 == 0:
                n_err = sum(1 for r in results if r.get("is_error"))
                notify(
                    f"Progress: {total_done}/{len(samples)} samples done "
                    f"({n_err} errors, {rate:.2f} samples/s, ETA ~{eta_m:.0f}min)"
                )

        # Final checkpoint
        save_checkpoint(results, ckpt_path)

        # Cleanup model to free GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Compute metrics for all results
    print("\nComputing automatic metrics...")
    for r in results:
        if r.get("is_error"):
            r["edit_similarity"] = 0.0
            r["bleu"] = 0.0
            r["rouge_l"] = 0.0
            continue
        m = compute_metrics(r["ground_truth"], r["prediction"])
        r.update(m)

    # Compile final output
    n_errors = sum(1 for r in results if r.get("is_error"))
    valid = [r for r in results if not r.get("is_error")]

    output = {
        "model": args.model,
        "model_label": label,
        "no_rag": True,
        "n_samples": len(results),
        "n_errors": n_errors,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics_summary": {},
        "results": results,
    }

    if valid:
        for metric_key in ["edit_similarity", "bleu", "rouge_l", "binary_correct"]:
            vals = [r[metric_key] for r in valid if metric_key in r]
            if vals:
                output["metrics_summary"][metric_key] = {
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                }

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")

    # Print summary
    ms = output.get("metrics_summary", {})
    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Samples: {len(results)} ({n_errors} errors)")
    if ms:
        for k, v in ms.items():
            print(f"  {k}: {v['mean']:.4f}")
    print(f"{'='*60}")

    # Final telegram
    es = ms.get("edit_similarity", {}).get("mean", 0)
    bleu = ms.get("bleu", {}).get("mean", 0)
    rl = ms.get("rouge_l", {}).get("mean", 0)
    notify(
        f"DONE: {label} eval complete. "
        f"{len(results)} samples, {n_errors} errors. "
        f"EditSim={es:.3f}, BLEU={bleu:.3f}, ROUGE-L={rl:.3f}"
    )


if __name__ == "__main__":
    main()
