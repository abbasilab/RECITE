"""
Concurrent evaluation script for endpoint-served LLMs (vLLM, etc.)
Used for rebuttal: Llama-3.1-70B, Qwen3-32B on large GPU clusters.

Sends parallel requests to maximize vLLM throughput.
Runs predictions first, then batch judges via UCSF Versa.
"""
import asyncio
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent

# Load prompts
PROMPTS = json.loads((ROOT / "config" / "benchmark_prompts.json").read_text())
MODEL_SYSTEM = PROMPTS["model_prompt"]["system"]
MODEL_USER_RAG = PROMPTS["model_prompt"]["user_template_rag"]
MODEL_USER_NORAG = PROMPTS["model_prompt"]["user_template"]
JUDGE_SYSTEM = PROMPTS["judge_prompt"]["system"]
JUDGE_USER = PROMPTS["judge_prompt"]["user_template"]
JUDGE_BATCHED_SYSTEM = PROMPTS["judge_prompt_batched"]["system"]
JUDGE_BATCHED_USER = PROMPTS["judge_prompt_batched"]["user_template"]


def build_prompt(row: dict, no_rag: bool = True, max_evidence_chars: int = 20000, prompt_suffix: str = "") -> tuple[str, str]:
    """Build system + user prompt for a benchmark sample.
    Truncates evidence to max_evidence_chars. Callers should set this based on
    the model's context window: (context_window - 4096) * 4 chars/token.
    prompt_suffix is appended to the user prompt (e.g. model-specific instructions).
    """
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
    # Truncate evidence to keep prompt within context
    if has_doc and len(evidence) > max_evidence_chars:
        evidence = evidence[:max_evidence_chars]

    if no_rag:
        user_prompt = MODEL_USER_NORAG.format(**kwargs)
        if has_doc:
            user_prompt += f"\n\nSupporting evidence:\n{evidence}"
    else:
        if has_doc:
            user_prompt = MODEL_USER_RAG.format(**kwargs)
            user_prompt += f"\n\nSupporting evidence:\n{evidence}"
        else:
            user_prompt = MODEL_USER_NORAG.format(**kwargs)

    if prompt_suffix:
        user_prompt += f"\n\n{prompt_suffix}"

    return MODEL_SYSTEM, user_prompt


def strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> tags from Qwen3 responses."""
    import re
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


async def call_endpoint(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    max_context: int = 8192,
    temperature: float = 0,
    timeout: float = 300.0,
) -> str:
    """Call OpenAI-compatible endpoint."""
    messages = []
    # Gemma models don't support system role — merge into user message
    if "gemma" in model.lower():
        combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        messages.append({"role": "user", "content": combined})
    else:
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

    # Estimate token count (~4 chars/token) and cap max_tokens
    est_input_tokens = sum(len(m["content"]) for m in messages) // 4
    effective_max_tokens = min(max_tokens, max(256, max_context - est_input_tokens - 100))

    for attempt in range(3):
        try:
            payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": effective_max_tokens,
                }
            # Disable thinking for Qwen3 models
            if "qwen3" in model.lower() or "Qwen3" in model:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            resp = await client.post(
                f"{endpoint}/chat/completions",
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return strip_thinking_tags(content)
            elif resp.status_code == 400:
                error = resp.json().get("error", {}).get("message", resp.text[:200])
                # If context too long, truncate and retry
                if "maximum context length" in error and attempt < 2:
                    # Halve the user prompt
                    user_prompt = user_prompt[:len(user_prompt)//2]
                    messages[-1]["content"] = user_prompt
                    est_input_tokens = sum(len(m["content"]) for m in messages) // 4
                    effective_max_tokens = min(max_tokens, max(256, max_context - est_input_tokens - 100))
                    continue
                return f"[ERROR_400] {error}"
            elif resp.status_code in (429, 503):
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                resp.raise_for_status()
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return f"[ERROR_TIMEOUT] {e}"
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return f"[ERROR] {type(e).__name__}: {e}"
    return "[ERROR] Max retries exceeded"


async def call_ollama_native(
    client: httpx.AsyncClient,
    ollama_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    timeout: float = 300.0,
) -> str:
    """Call Ollama native API with think=false for Qwen3 models."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    for attempt in range(3):
        try:
            resp = await client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0, "num_predict": max_tokens},
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("message", {}).get("content", "")
            elif resp.status_code in (429, 503):
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                return f"[ERROR_{resp.status_code}] {resp.text[:200]}"
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadError) as e:
            if attempt < 2:
                await asyncio.sleep(5 * (attempt + 1))
                continue
            return f"[ERROR_TIMEOUT] {e}"
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return f"[ERROR] {type(e).__name__}: {e}"
    return "[ERROR] Max retries exceeded"


async def run_predictions(
    endpoint: str,
    model: str,
    samples: list[dict],
    no_rag: bool = True,
    max_concurrent: int = 16,
    max_tokens: int = 2048,
    use_ollama: bool = False,
    timeout: float = 300.0,
    max_context: int = 8192,
    max_evidence_chars: int = 20000,
    prompt_suffix: str = "",
    output_path: Path | None = None,
    save_every: int = 50,
) -> list[dict]:
    """Run predictions concurrently on all samples.

    If output_path is provided, saves incrementally every save_every samples.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(samples)
    completed = 0

    async with httpx.AsyncClient() as client:
        async def process_one(idx: int, sample: dict):
            async with semaphore:
                try:
                    sys_prompt, user_prompt = build_prompt(sample, no_rag=no_rag, max_evidence_chars=max_evidence_chars, prompt_suffix=prompt_suffix)
                    if use_ollama:
                        prediction = await call_ollama_native(
                            client, endpoint, model, sys_prompt, user_prompt,
                            max_tokens=max_tokens,
                        )
                    else:
                        prediction = await call_endpoint(
                            client, endpoint, model, sys_prompt, user_prompt,
                            max_tokens=max_tokens, timeout=timeout,
                            max_context=max_context,
                        )
                except Exception as e:
                    prediction = f"[ERROR] {type(e).__name__}: {e}"
                results[idx] = {
                    "id": int(float(sample["id"])),
                    "instance_id": sample.get("instance_id", ""),
                    "prediction": prediction,
                    "predicted_at": datetime.now(timezone.utc).isoformat(),
                    "ground_truth": str(sample.get("reference_text", "") or ""),
                    "is_error": prediction.startswith("[ERROR"),
                }

        tasks = [process_one(i, s) for i, s in enumerate(samples)]
        pbar = tqdm(total=len(tasks), desc=f"Predict ({model})", unit="sample")

        # Process with progress updates + incremental saves
        for coro in asyncio.as_completed(tasks):
            await coro
            completed += 1
            pbar.update(1)
            if output_path and completed % save_every == 0:
                partial = [r for r in results if r is not None]
                save_results(partial, output_path, model, no_rag)
                tqdm.write(f"  [checkpoint] saved {len(partial)}/{len(samples)} predictions")
        pbar.close()

    final = [r for r in results if r is not None]
    if output_path:
        save_results(final, output_path, model, no_rag)
    return final


def parse_judge_scores(response: str) -> dict:
    """Parse judge response into binary and ordinal scores."""
    import re
    response = response.strip()

    # Try "binary,ordinal" format
    m = re.match(r"^\s*([01])\s*,\s*([0-4])\s*$", response)
    if m:
        return {"binary": float(m.group(1)), "ordinal": float(m.group(2))}

    # Try extracting from longer text
    m = re.search(r"([01])\s*,\s*([0-4])", response)
    if m:
        return {"binary": float(m.group(1)), "ordinal": float(m.group(2))}

    # Default
    return {"binary": 0.0, "ordinal": 2.0}


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
        print("WARNING: UCSF Versa credentials not set. Skipping judge.")
        return predictions

    judge_endpoint = f"{resource_ep}/openai/deployments/gpt-4o-2024-08-06/chat/completions?api-version={api_ver}"
    headers = {"api-key": api_key, "Content-Type": "application/json"}

    semaphore = asyncio.Semaphore(max_concurrent)

    # Build batches
    valid_preds = [p for p in predictions if not p.get("is_error")]
    batches = [valid_preds[i:i+batch_size] for i in range(0, len(valid_preds), batch_size)]

    async with httpx.AsyncClient() as client:
        async def judge_batch(batch: list[dict]):
            async with semaphore:
                # Build pairs text
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
                            # Parse JSON response
                            try:
                                import re
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
                            await asyncio.sleep(5 * (attempt + 1))
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

                # Fallback
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

    # Mark error predictions
    for p in predictions:
        if p.get("is_error"):
            p["judge_binary"] = 0.0
            p["judge_ordinal"] = 0.0
            p["judge_raw"] = "[SKIPPED] prediction error"

    return predictions


def save_results(results: list[dict], output_path: Path, model_name: str, no_rag: bool):
    """Save results to JSON."""
    output = {
        "model": model_name,
        "no_rag": no_rag,
        "n_samples": len(results),
        "n_errors": sum(1 for r in results if r.get("is_error")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    # Compute summary stats
    valid = [r for r in results if not r.get("is_error") and "judge_binary" in r]
    if valid:
        output["binary_equiv"] = sum(r["judge_binary"] for r in valid) / len(valid)
        output["mean_ordinal"] = sum(r["judge_ordinal"] for r in valid) / len(valid)
        output["n_judged"] = len(valid)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved {len(results)} results to {output_path}")
    if "binary_equiv" in output:
        print(f"  Binary Equiv: {output['binary_equiv']:.4f} ({output['binary_equiv']*100:.1f}%)")
        print(f"  Mean Ordinal: {output['mean_ordinal']:.4f}")
        print(f"  Judged: {output['n_judged']}, Errors: {output['n_errors']}")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Concurrent endpoint evaluation")
    parser.add_argument("--endpoint", required=True, help="vLLM endpoint URL (e.g., http://localhost:8100/v1)")
    parser.add_argument("--model", required=True, help="Model name for API calls")
    parser.add_argument("--model-label", default=None, help="Label for output file (default: derived from model)")
    parser.add_argument("--parquet", default=str(ROOT / "data/benchmark_splits/benchmark.parquet"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/rebuttal"))
    parser.add_argument("--max-concurrent", type=int, default=16, help="Max concurrent prediction requests")
    parser.add_argument("--judge-concurrent", type=int, default=8, help="Max concurrent judge batches")
    parser.add_argument("--judge-batch-size", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for generation")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout per request in seconds")
    parser.add_argument("--max-context", type=int, default=8192, help="Max context window for token capping")
    parser.add_argument("--max-evidence-chars", type=int, default=None,
                        help="Max evidence chars (default: (max_context - 4096) * 4)")
    parser.add_argument("--prompt-suffix", type=str, default="",
                        help="Model-specific text appended to user prompt (e.g. 'Do not use markdown formatting.')")
    parser.add_argument("--no-rag", action="store_true", default=True)
    parser.add_argument("--skip-judge", action="store_true", help="Skip judge evaluation")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama native API (for Qwen3 think=false)")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N predictions (default: 50)")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for testing")
    args = parser.parse_args()

    label = args.model_label or args.model.replace("/", "_").replace("-", "_")
    rag_suffix = "no_rag" if args.no_rag else "rag"

    # Load data
    print(f"Loading benchmark from {args.parquet}")
    df = pd.read_parquet(args.parquet)
    samples = df.to_dict("records")
    if args.limit:
        samples = samples[:args.limit]
    print(f"  {len(samples)} samples loaded")

    # Compute evidence char budget from context window if not specified
    max_evidence_chars = args.max_evidence_chars
    if max_evidence_chars is None:
        # Reserve 4096 tokens for prompt overhead + generation, convert rest to chars (~4 chars/token)
        max_evidence_chars = max(4000, (args.max_context - 4096) * 4)
    print(f"  Evidence budget: {max_evidence_chars} chars (~{max_evidence_chars//4} tokens)")

    # Run predictions (with incremental saves)
    output_path = Path(args.output_dir) / f"{label}_{rag_suffix}.json"
    t0 = time.time()
    results = await run_predictions(
        endpoint=args.endpoint,
        model=args.model,
        samples=samples,
        no_rag=args.no_rag,
        max_concurrent=args.max_concurrent,
        max_tokens=args.max_tokens,
        use_ollama=args.use_ollama,
        timeout=args.timeout,
        max_context=args.max_context,
        max_evidence_chars=max_evidence_chars,
        prompt_suffix=args.prompt_suffix,
        output_path=output_path,
        save_every=args.save_every,
    )
    t_pred = time.time() - t0
    n_errors = sum(1 for r in results if r.get("is_error"))
    print(f"\nPredictions done: {len(results)} samples in {t_pred:.1f}s ({t_pred/len(results):.2f}s/sample)")
    print(f"  Errors: {n_errors}")

    # Run judge
    if not args.skip_judge:
        t1 = time.time()
        results = await run_judge_batched(
            results,
            batch_size=args.judge_batch_size,
            max_concurrent=args.judge_concurrent,
        )
        t_judge = time.time() - t1
        print(f"\nJudge done in {t_judge:.1f}s")

    # Final save (with judge scores if run)
    save_results(results, output_path, args.model, args.no_rag)


if __name__ == "__main__":
    asyncio.run(main())
