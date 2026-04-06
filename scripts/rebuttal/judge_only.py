"""
Judge-only scoring: runs LLM-as-judge (GPT-4o via UCSF Versa) on existing
prediction JSON files that lack judge scores.

Usage:
  uv run python scripts/rebuttal/judge_only.py \
    --input data/rebuttal/gemma2-27b_no_rag.json \
    --judge-batch-size 10 --judge-concurrent 8

Requires env vars: UCSF_API_KEY, UCSF_RESOURCE_ENDPOINT
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent

# Load judge prompts
PROMPTS = json.loads((ROOT / "config" / "benchmark_prompts.json").read_text())
JUDGE_BATCHED_SYSTEM = PROMPTS["judge_prompt_batched"]["system"]
JUDGE_BATCHED_USER = PROMPTS["judge_prompt_batched"]["user_template"]


async def run_judge_batched(
    predictions: list[dict],
    batch_size: int = 10,
    max_concurrent: int = 8,
) -> list[dict]:
    """Run LLM judge on predictions using UCSF Versa GPT-4o in batches."""
    api_key = os.environ.get("UCSF_API_KEY")
    api_ver = os.environ.get("UCSF_API_VER", "2024-10-21")
    resource_ep = os.environ.get("UCSF_RESOURCE_ENDPOINT", "").rstrip("/")

    if not api_key or not resource_ep:
        print("ERROR: UCSF Versa credentials not set (UCSF_API_KEY, UCSF_RESOURCE_ENDPOINT)")
        sys.exit(1)

    judge_endpoint = f"{resource_ep}/openai/deployments/gpt-4o-2024-08-06/chat/completions?api-version={api_ver}"
    headers = {"api-key": api_key, "Content-Type": "application/json"}

    semaphore = asyncio.Semaphore(max_concurrent)

    valid_preds = [p for p in predictions if not p.get("is_error")]
    batches = [valid_preds[i:i+batch_size] for i in range(0, len(valid_preds), batch_size)]
    print(f"  {len(valid_preds)} valid predictions, {len(batches)} batches (size {batch_size})")

    async with httpx.AsyncClient() as client:
        async def judge_batch(batch: list[dict]):
            async with semaphore:
                import re
                pairs = []
                for j, pred in enumerate(batch, 1):
                    pairs.append(
                        f"Pair {j}:\n"
                        f"Target Eligibility Criteria:\n{pred['ground_truth']}\n\n"
                        f"Predicted Eligibility Criteria:\n{pred['prediction']}"
                    )
                pairs_text = "\n\n---\n\n".join(pairs)
                user_msg = JUDGE_BATCHED_USER.format(n=len(batch), pairs=pairs_text)

                messages = [
                    {"role": "system", "content": JUDGE_BATCHED_SYSTEM},
                    {"role": "user", "content": user_msg},
                ]

                for attempt in range(3):
                    try:
                        resp = await client.post(
                            judge_endpoint,
                            headers=headers,
                            json={
                                "messages": messages,
                                "temperature": 0,
                                "max_tokens": 500,
                            },
                            timeout=120.0,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            content = data["choices"][0]["message"]["content"]
                            try:
                                json_match = re.search(r'\{[^{}]*\}', content)
                                if json_match:
                                    scores = json.loads(json_match.group())
                                else:
                                    scores = json.loads(content)
                            except json.JSONDecodeError:
                                scores = {}

                            for j, pred in enumerate(batch, 1):
                                key = str(j)
                                if key in scores:
                                    s = scores[key]
                                    if isinstance(s, list) and len(s) >= 2:
                                        pred["judge_binary"] = float(s[0])
                                        pred["judge_ordinal"] = float(s[1])
                                    else:
                                        pred["judge_binary"] = 0.0
                                        pred["judge_ordinal"] = 2.0
                                else:
                                    pred["judge_binary"] = 0.0
                                    pred["judge_ordinal"] = 2.0
                                pred["judge_raw"] = content
                            return
                        elif resp.status_code == 429:
                            wait = 5 * (attempt + 1)
                            print(f"  Rate limited, waiting {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                        else:
                            print(f"Judge error {resp.status_code}: {resp.text[:200]}")
                            await asyncio.sleep(2 ** attempt)
                            continue
                    except Exception as e:
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        print(f"Judge batch failed: {e}")

                for pred in batch:
                    pred["judge_binary"] = 0.0
                    pred["judge_ordinal"] = 2.0
                    pred["judge_raw"] = "[ERROR] judge failed"

        tasks = [judge_batch(b) for b in batches]
        pbar = tqdm(total=len(batches), desc="Judge (GPT-4o)", unit="batch")
        for coro in asyncio.as_completed(tasks):
            await coro
            pbar.update(1)
        pbar.close()

    for p in predictions:
        if p.get("is_error"):
            p["judge_binary"] = 0.0
            p["judge_ordinal"] = 0.0
            p["judge_raw"] = "[SKIPPED] prediction error"

    return predictions


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Judge-only scoring for existing prediction files")
    parser.add_argument("--input", required=True, help="Path to predictions JSON file")
    parser.add_argument("--judge-concurrent", type=int, default=8)
    parser.add_argument("--judge-batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (for cost testing)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    print(f"Loading predictions from {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    results = data["results"]
    print(f"  {len(results)} total samples, {sum(1 for r in results if r.get('is_error'))} errors")

    # Check if already judged
    already_judged = sum(1 for r in results if "judge_binary" in r)
    if already_judged > 0:
        print(f"  WARNING: {already_judged} samples already have judge scores — will overwrite")

    if args.limit:
        results = results[:args.limit]
        print(f"  Limited to {args.limit} samples")

    t0 = time.time()
    results = await run_judge_batched(
        results,
        batch_size=args.judge_batch_size,
        max_concurrent=args.judge_concurrent,
    )
    elapsed = time.time() - t0
    print(f"\nJudge scoring done in {elapsed:.1f}s")

    # Compute summary stats
    valid = [r for r in results if not r.get("is_error") and "judge_binary" in r]
    if valid:
        binary_equiv = sum(r["judge_binary"] for r in valid) / len(valid)
        mean_ordinal = sum(r["judge_ordinal"] for r in valid) / len(valid)
        print(f"\nResults:")
        print(f"  Binary equivalence: {binary_equiv:.1%}")
        print(f"  Mean ordinal: {mean_ordinal:.3f}/4.0")
        print(f"  Judged: {len(valid)}, Errors: {sum(1 for r in results if r.get('is_error'))}")

        data["binary_equiv"] = binary_equiv
        data["mean_ordinal"] = mean_ordinal
        data["n_judged"] = len(valid)

    # If limited, merge back
    if args.limit:
        with open(input_path) as f:
            full_data = json.load(f)
        for i, r in enumerate(results):
            full_data["results"][i] = r
        # Recompute stats on full set
        all_valid = [r for r in full_data["results"] if not r.get("is_error") and "judge_binary" in r]
        if all_valid:
            full_data["binary_equiv"] = sum(r["judge_binary"] for r in all_valid) / len(all_valid)
            full_data["mean_ordinal"] = sum(r["judge_ordinal"] for r in all_valid) / len(all_valid)
            full_data["n_judged"] = len(all_valid)
        data = full_data

    # Update results in data
    if not args.limit:
        data["results"] = results

    # Save
    with open(input_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved to {input_path}")


if __name__ == "__main__":
    asyncio.run(main())
