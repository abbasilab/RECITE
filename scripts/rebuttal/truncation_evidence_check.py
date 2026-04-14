"""Truncation evidence analysis via gpt-4.1-mini.

Classifies all 3,116 RECITE benchmark samples to determine whether the
Gemma-truncated protocol text (4096 cl100k_base tokens) contains the evidence
needed to answer the RECITE question.

Usage:
    set -a && source /home/rro/projects/phdmanager/.env && set +a
    uv run python scripts/truncation_evidence_check.py [--resume] [--dry-run] [--max-concurrent 20]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import tiktoken

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

# ── Config ──────────────────────────────────────────────────────────────
EVIDENCE_MAX_TOKENS = 4096
DEPLOYMENT_ID = "gpt-4.1-mini-2025-04-14"
MODEL_NAME = "gpt-4.1-mini"

SPLITS_DIR = ROOT / "data" / "benchmark_splits"
DB_PATH = ROOT / "data" / "dev" / "recite.db"
LOG_DIR = ROOT / "logs"
TABLE = "truncation_evidence_41mini"

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_DIR / "truncation_evidence.log", rotation="10 MB", level="DEBUG")

# ── Tokenizer ───────────────────────────────────────────────────────────
enc = tiktoken.get_encoding("cl100k_base")


def truncate_evidence(text: str) -> str:
    """Truncate text to EVIDENCE_MAX_TOKENS cl100k_base tokens."""
    tokens = enc.encode(text)
    if len(tokens) <= EVIDENCE_MAX_TOKENS:
        return text
    return enc.decode(tokens[:EVIDENCE_MAX_TOKENS])


# ── Load samples ────────────────────────────────────────────────────────
def load_all_samples() -> list[dict]:
    """Load all samples from train/val/test parquet files."""
    samples = []
    for split in ("train", "val", "test"):
        path = SPLITS_DIR / f"{split}.parquet"
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            samples.append({
                "id": int(row["id"]),
                "instance_id": row["instance_id"],
                "source_version": int(row["source_version"]),
                "target_version": int(row["target_version"]),
                "evidence": row["evidence"],
                "reference_text": row["reference_text"],
            })
    logger.info(f"Loaded {len(samples)} samples from 3 splits")
    return samples


# ── DB setup ────────────────────────────────────────────────────────────
def ensure_table(conn: sqlite3.Connection):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            recite_id               INTEGER NOT NULL,
            instance_id                  TEXT NOT NULL,
            source_version            INTEGER,
            target_version              INTEGER,
            target_version_mentioned    INTEGER,
            reference_text_present    INTEGER,
            reference_text_confidence REAL,
            reference_text_quote      TEXT,
            model_used              TEXT NOT NULL DEFAULT '{MODEL_NAME}',
            analyzed_at             TEXT NOT NULL,
            raw_response            TEXT,
            PRIMARY KEY (recite_id)
        )
    """)
    conn.commit()


def get_done_ids(conn: sqlite3.Connection) -> set[int]:
    try:
        rows = conn.execute(f"SELECT recite_id FROM {TABLE}").fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()


def save_batch(conn: sqlite3.Connection, rows: list[tuple]):
    conn.executemany(
        f"""INSERT OR REPLACE INTO {TABLE}
            (recite_id, instance_id, source_version, target_version,
             target_version_mentioned, reference_text_present,
             reference_text_confidence, reference_text_quote,
             model_used, analyzed_at, raw_response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows
    )
    conn.commit()


# ── Prompt construction ─────────────────────────────────────────────────
SYSTEM_PROMPT = "You are a clinical trial protocol analyst. Answer precisely in JSON format."

USER_TEMPLATE = """Below is a TRUNCATED excerpt from a clinical trial protocol, and an AMENDED eligibility criterion text.

TRUNCATED PROTOCOL (first 4096 tokens):
{truncated_evidence}

AMENDED ELIGIBILITY CRITERION:
{reference_text}

The version changed from v{source_version} to v{target_version}.

Questions:
1. Does the truncated protocol excerpt contain the version number "v{target_version}" or "version {target_version}" or equivalent? (target_version_mentioned: 0 or 1)
2. Is the substance of the amended eligibility criterion present in the truncated excerpt? Not just keywords — the actual criterion change must be identifiable. (reference_text_present: 0 or 1)
3. How confident are you in #2? (confidence: 0.0 to 1.0)
4. If present, quote the relevant passage (max 200 chars). If not present, say "NOT_FOUND". (quote: string)

Respond ONLY with JSON:
{{"target_version_mentioned": 0, "reference_text_present": 1, "confidence": 0.85, "quote": "..."}}"""


def parse_response(raw: str) -> dict:
    """Parse the JSON response from the model."""
    defaults = {
        "target_version_mentioned": None,
        "reference_text_present": None,
        "reference_text_confidence": None,
        "reference_text_quote": None,
    }
    if not raw or raw.startswith("ERROR"):
        return defaults
    try:
        # Try to extract JSON from response (may have markdown fences)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        data = json.loads(text)
        return {
            "target_version_mentioned": int(data.get("target_version_mentioned", 0)),
            "reference_text_present": int(data.get("reference_text_present", 0)),
            "reference_text_confidence": float(data.get("confidence", 0.0)),
            "reference_text_quote": str(data.get("quote", ""))[:500],
        }
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Parse error: {e} — raw: {raw[:200]}")
        return defaults


# ── Async API ───────────────────────────────────────────────────────────
async def call_versa(session: aiohttp.ClientSession, url: str, headers: dict,
                     system_prompt: str, user_prompt: str,
                     sample_id: int, max_retries: int = 5) -> str:
    """Call Versa API with exponential backoff."""
    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    for attempt in range(max_retries + 1):
        try:
            async with session.post(url, headers=headers, json=body,
                                    timeout=aiohttp.ClientTimeout(total=90)) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", min(2 ** attempt, 60)))
                    logger.warning(f"Sample {sample_id}: 429, retry in {retry_after}s (attempt {attempt+1})")
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                data = await resp.json()
                if "choices" in data and data["choices"]:
                    raw = data["choices"][0]["message"].get("content", "")
                    return raw.strip() if isinstance(raw, str) else str(raw)
                return "ERROR: no choices in response"
        except Exception as e:
            if attempt >= max_retries:
                logger.error(f"Sample {sample_id}: exhausted retries: {e}")
                return f"ERROR: {e}"
            delay = min(2 ** attempt, 60)
            logger.warning(f"Sample {sample_id}: retry {attempt+1} in {delay}s: {e}")
            await asyncio.sleep(delay)
    return "ERROR: unreachable"


# ── Main ────────────────────────────────────────────────────────────────
async def main(args):
    # Versa API setup
    api_key = os.getenv("UCSF_API_KEY")
    api_ver = os.getenv("UCSF_API_VER")
    endpoint = os.getenv("UCSF_RESOURCE_ENDPOINT", "").rstrip("/")
    if not all([api_key, api_ver, endpoint]):
        logger.error("Missing UCSF_API_KEY, UCSF_API_VER, or UCSF_RESOURCE_ENDPOINT")
        sys.exit(1)

    api_url = f"{endpoint}/openai/deployments/{DEPLOYMENT_ID}/chat/completions?api-version={api_ver}"
    api_headers = {"Content-Type": "application/json", "api-key": api_key}

    # Load samples & setup DB
    all_samples = load_all_samples()
    conn = sqlite3.connect(str(DB_PATH))
    ensure_table(conn)
    done_ids = get_done_ids(conn) if args.resume else set()
    todo = [s for s in all_samples if s["id"] not in done_ids]
    logger.info(f"TODO: {len(todo)} samples ({len(done_ids)} already done)")

    if args.dry_run:
        logger.info("Dry run — exiting")
        conn.close()
        return

    if not todo:
        logger.info("Nothing to do — all samples already processed")
        conn.close()
        return

    semaphore = asyncio.Semaphore(args.max_concurrent)
    write_queue: asyncio.Queue = asyncio.Queue()
    write_done = asyncio.Event()
    scored = 0
    errors = 0
    t0 = time.monotonic()
    now_str = datetime.now(timezone.utc).isoformat()

    async def db_writer():
        nonlocal scored
        batch = []
        while True:
            try:
                item = await asyncio.wait_for(write_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if write_done.is_set() and write_queue.empty():
                    break
                continue
            if item is None:
                break
            batch.append(item)
            scored += 1
            if len(batch) >= 50 or (write_done.is_set() and write_queue.empty()):
                save_batch(conn, batch)
                batch = []
                if scored % 200 == 0:
                    elapsed = time.monotonic() - t0
                    rate = scored / elapsed if elapsed > 0 else 0
                    logger.info(f"Progress: {scored}/{len(todo)} ({rate:.1f}/s)")
        if batch:
            save_batch(conn, batch)

    async with aiohttp.ClientSession() as session:
        async def process_one(sample: dict):
            nonlocal errors
            async with semaphore:
                truncated = truncate_evidence(sample["evidence"])
                user_prompt = USER_TEMPLATE.format(
                    truncated_evidence=truncated,
                    reference_text=sample["reference_text"],
                    source_version=sample["source_version"],
                    target_version=sample["target_version"],
                )
                raw = await call_versa(session, api_url, api_headers,
                                       SYSTEM_PROMPT, user_prompt, sample["id"])
                parsed = parse_response(raw)
                if parsed["reference_text_present"] is None:
                    errors += 1
                await write_queue.put((
                    sample["id"],
                    sample["instance_id"],
                    sample["source_version"],
                    sample["target_version"],
                    parsed["target_version_mentioned"],
                    parsed["reference_text_present"],
                    parsed["reference_text_confidence"],
                    parsed["reference_text_quote"],
                    MODEL_NAME,
                    now_str,
                    raw,
                ))

        writer_task = asyncio.create_task(db_writer())
        await asyncio.gather(*[process_one(s) for s in todo])
        write_done.set()
        await writer_task

    conn.close()
    elapsed = time.monotonic() - t0

    # Stats
    stats_conn = sqlite3.connect(str(DB_PATH))
    total = stats_conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    present = stats_conn.execute(
        f"SELECT COUNT(*) FROM {TABLE} WHERE reference_text_present = 1"
    ).fetchone()[0]
    version_mentioned = stats_conn.execute(
        f"SELECT COUNT(*) FROM {TABLE} WHERE target_version_mentioned = 1"
    ).fetchone()[0]
    stats_conn.close()

    pct_present = 100 * present / total if total else 0
    pct_version = 100 * version_mentioned / total if total else 0

    report = (
        f"Truncation evidence check complete.\n"
        f"Total: {total}, Scored this run: {scored}, Errors: {errors}\n"
        f"reference_text_present=1: {present}/{total} ({pct_present:.1f}%)\n"
        f"target_version_mentioned=1: {version_mentioned}/{total} ({pct_version:.1f}%)\n"
        f"Elapsed: {elapsed:.0f}s"
    )
    logger.info(report)

    # Telegram notification
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(
                "http://localhost:8443/api/send",
                json={"chat_id": "7054155159", "text": report},
                timeout=aiohttp.ClientTimeout(total=10),
            )
        logger.info("Telegram notification sent")
    except Exception as e:
        logger.warning(f"Telegram notification failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truncation evidence analysis via gpt-4.1-mini")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already in the table")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show counts without calling API")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Max concurrent API calls (default: 20)")
    args = parser.parse_args()
    asyncio.run(main(args))
