"""Unified extensible judge scoring script.

Scores predictions from DB configs or JSON files using any judge variant + model.
Concurrent via asyncio + aiohttp. Resumes from existing rows.

Usage:
    set -a && source /home/rro/projects/phdmanager/.env && set +a

    # Score DB configs (all canonical 8 models) with v1 + mini:
    uv run python scripts/judge_score.py --variant 1 --judge-model gpt-4.1-mini

    # Score rebuttal JSON files:
    uv run python scripts/judge_score.py --variant 1 --judge-model gpt-4.1-mini \
        --json data/rebuttal/qwen25-72b_no_rag.json \
        --json data/rebuttal/gemma2-27b_no_rag.json

    # Score specific DB configs only:
    uv run python scripts/judge_score.py --variant 1 --judge-model gpt-4.1-mini \
        --config-id 9919b9efea552663a5029a76

    # Dry run:
    uv run python scripts/judge_score.py --variant 1 --judge-model gpt-4.1-mini --dry-run

    # Multiple variants:
    uv run python scripts/judge_score.py --variant 1 --variant 3 --judge-model gpt-4.1-mini
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

# ── Paths ────────────────────────────────────────────────────────────────
DB_PATH = ROOT / "data" / "prod" / "benchmark_results.db"
VARIANTS_DIR = ROOT / "config" / "judge_variants"
LOG_DIR = ROOT / "logs"
SCORE_TABLE = "judge_scores"

# Canonical DB configs: 8 models × 2 RAG configs
CANONICAL_CONFIGS = [
    ("9919b9efea552663a5029a76", "local-gemma2-2b", 1),
    ("783a0330de928e71913c0f05", "local-gemma2-2b", 0),
    ("be5060d19cd1f0e7424afbb0", "local-gemma2-9b", 1),
    ("b602b83dc65b3c677974cbf9", "local-gemma2-9b", 0),
    ("1fa16bf2ad873d803624dde8", "local-longctx-7b", 1),
    ("97ef508facb8b047f3d1b4ae", "local-longctx-7b", 0),
    ("83525e11a31ca0dabb6e48c7", "local-qwen-0_5b", 1),
    ("8ae5df04b909d4c5eff94bdd", "local-qwen-0_5b", 0),
    ("1988ec418928af1290853c4a", "local-qwen-3b", 1),
    ("2ed8bbee8162109e1d7ebb72", "local-qwen-3b", 0),
    ("5bb7e5de4490ea5d71a57caa", "local-qwen-7b", 1),
    ("98d87e89859b6f9d79be8741", "local-qwen-7b", 0),
    ("9fa974044430151d492dc6c3", "versa-4o", 1),
    ("46987023a1086f5f9072bae4", "versa-4o", 0),
    ("c61f176d6d55c87d6318590e", "versa-4o-mini", 1),
    ("f37e1a948e0085d497871277", "versa-4o-mini", 0),
]

# Model name aliases for Versa deployment IDs
VERSA_DEPLOYMENT_MAP = {
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
}

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_DIR / "judge_score.log", rotation="10 MB", level="DEBUG")


# ── Score parsing ────────────────────────────────────────────────────────
def parse_judge_scores(response: str) -> tuple[int, int]:
    """Extract (binary, ordinal) from judge response."""
    if not response or not isinstance(response, str):
        return (0, 0)
    m = re.search(r'SCORES:\s*([01])\s*,\s*([0-4])', response)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r'\b([01])\s*,\s*([0-4])\b', response)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    nums = re.findall(r'\b(\d)\b', response)
    if len(nums) >= 2:
        return (min(int(nums[-2]), 1), min(int(nums[-1]), 4))
    logger.warning(f"Parse fail: {response[:200]}")
    return (0, 0)


# ── Variant loading ──────────────────────────────────────────────────────
def load_variant(variant_id: str | int) -> dict:
    """Load a judge variant config by number or path."""
    path = VARIANTS_DIR / f"variant_{variant_id}.json"
    if not path.exists():
        path = Path(str(variant_id))
    if not path.exists():
        logger.error(f"Variant not found: {variant_id}")
        sys.exit(1)
    with open(path) as f:
        v = json.load(f)
    logger.info(f"Loaded variant: {v['name']} — {v['description']}")
    return v


def resolve_judge_model(model_name: str) -> str:
    """Resolve short model name to full Versa deployment ID."""
    return VERSA_DEPLOYMENT_MAP.get(model_name, model_name)


# ── DB setup ─────────────────────────────────────────────────────────────
def ensure_score_table(conn: sqlite3.Connection):
    """Create judge_scores table with composite PK including variant + judge model."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCORE_TABLE} (
            config_id       TEXT NOT NULL,
            sample_id       INTEGER NOT NULL,
            split_name      TEXT NOT NULL,
            model_id        TEXT NOT NULL,
            no_rag          INTEGER NOT NULL,
            judge_model     TEXT NOT NULL,
            judge_variant   TEXT NOT NULL,
            binary_score    INTEGER,
            ordinal_score   INTEGER,
            normalized_score REAL,
            raw_response    TEXT,
            scored_at       TEXT NOT NULL,
            PRIMARY KEY (config_id, sample_id, split_name, judge_model, judge_variant)
        )
    """)
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{SCORE_TABLE}_model
        ON {SCORE_TABLE}(model_id, no_rag)
    """)
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{SCORE_TABLE}_judge
        ON {SCORE_TABLE}(judge_model, judge_variant)
    """)
    conn.commit()


def get_done_keys(conn: sqlite3.Connection, judge_model: str,
                  judge_variant: str, split: str) -> set[tuple[str, int]]:
    """Return set of (config_id, sample_id) already scored for this judge+variant."""
    try:
        rows = conn.execute(
            f"""SELECT config_id, sample_id FROM {SCORE_TABLE}
                WHERE split_name = ? AND judge_model = ? AND judge_variant = ?""",
            (split, judge_model, judge_variant)
        ).fetchall()
        return {(r[0], r[1]) for r in rows}
    except sqlite3.OperationalError:
        return set()


def get_db_samples(conn: sqlite3.Connection, config_id: str,
                   split: str) -> list[dict]:
    """Get samples from a DB results table."""
    table = f"results_{config_id}"
    try:
        rows = conn.execute(
            f"SELECT id, prediction, reference_text FROM [{table}] WHERE split_name = ?",
            (split,)
        ).fetchall()
    except sqlite3.OperationalError as e:
        logger.error(f"Table {table} not found: {e}")
        return []
    return [{"id": r[0], "prediction": r[1], "ground_truth": r[2]} for r in rows]


def load_json_samples(path: Path) -> tuple[list[dict], str, bool]:
    """Load prediction samples from a JSON file. Returns (samples, model_id, no_rag)."""
    with open(path) as f:
        data = json.load(f)
    model = data.get("model", path.stem)
    no_rag = data.get("no_rag", True)
    results = data.get("results", [])
    samples = [
        {"id": r["id"], "prediction": r.get("prediction", ""), "ground_truth": r.get("ground_truth", "")}
        for r in results
        if not r.get("is_error", False)
    ]
    return samples, model, no_rag


def save_scores_batch(conn: sqlite3.Connection, rows: list[tuple]):
    """Batch insert scores."""
    conn.executemany(
        f"""INSERT OR REPLACE INTO {SCORE_TABLE}
            (config_id, sample_id, split_name, model_id, no_rag, judge_model, judge_variant,
             binary_score, ordinal_score, normalized_score, raw_response, scored_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows
    )
    conn.commit()


# ── Async API calls ──────────────────────────────────────────────────────
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


async def score_batch(samples: list[dict], config_id: str, model_id: str,
                      no_rag: bool, variant: dict, judge_model: str,
                      split: str, semaphore: asyncio.Semaphore,
                      api_url: str, headers: dict,
                      done_keys: set[tuple[str, int]],
                      dry_run: bool = False) -> tuple[int, int]:
    """Score a batch of samples for one config+variant combo."""
    todo = [s for s in samples if (config_id, s["id"]) not in done_keys]
    already = len(samples) - len(todo)
    logger.info(f"{model_id} norag={int(no_rag)} [{variant['name']}]: "
                f"{len(todo)} to score ({already} done)")

    if not todo or dry_run:
        return already, 0

    system_prompt = variant["system"]
    user_template = variant["user_template"]
    conn = sqlite3.connect(str(DB_PATH))
    write_queue: asyncio.Queue = asyncio.Queue()
    scored = 0
    write_done = asyncio.Event()
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
                save_scores_batch(conn, batch)
                batch = []
                if scored % 100 == 0:
                    logger.info(f"  {model_id} [{variant['name']}]: {scored}/{len(todo)}")
        if batch:
            save_scores_batch(conn, batch)

    async with aiohttp.ClientSession() as session:
        async def score_one(sample: dict):
            async with semaphore:
                user_prompt = (user_template
                               .replace("{ground_truth}", sample["ground_truth"] or "")
                               .replace("{prediction}", sample["prediction"] or ""))
                raw = await call_versa(session, api_url, headers, system_prompt,
                                       user_prompt, sample["id"])
                binary, ordinal = parse_judge_scores(raw)
                await write_queue.put((
                    config_id, sample["id"], split, model_id, int(no_rag),
                    judge_model, variant["name"], binary, ordinal,
                    ordinal / 4.0, raw, now_str
                ))

        writer_task = asyncio.create_task(db_writer())
        await asyncio.gather(*[score_one(s) for s in todo])
        write_done.set()
        await writer_task

    conn.close()
    return already, scored


# ── Main ─────────────────────────────────────────────────────────────────
async def main(args):
    # Load variants
    variants = [load_variant(v) for v in args.variant]

    # Resolve judge model
    judge_model = resolve_judge_model(args.judge_model)
    logger.info(f"Judge model: {judge_model}")

    # Versa API setup
    api_key = os.getenv("UCSF_API_KEY")
    api_ver = os.getenv("UCSF_API_VER")
    endpoint = os.getenv("UCSF_RESOURCE_ENDPOINT", "").rstrip("/")
    if not all([api_key, api_ver, endpoint]):
        logger.error("Missing UCSF_API_KEY, UCSF_API_VER, or UCSF_RESOURCE_ENDPOINT")
        sys.exit(1)

    api_url = f"{endpoint}/openai/deployments/{judge_model}/chat/completions?api-version={api_ver}"
    api_headers = {"Content-Type": "application/json", "api-key": api_key}

    # Ensure table
    conn = sqlite3.connect(str(DB_PATH))
    ensure_score_table(conn)
    conn.close()

    semaphore = asyncio.Semaphore(args.max_concurrent)
    split = args.split
    t0 = time.monotonic()
    total_already = 0
    total_scored = 0

    # Build work items: list of (samples, config_id, model_id, no_rag)
    work_items = []

    # JSON sources
    for json_path in (args.json or []):
        p = Path(json_path)
        if not p.is_absolute():
            p = ROOT / p
        samples, model_id, no_rag = load_json_samples(p)
        # Use filename stem as config_id for JSON sources
        config_id = f"json_{p.stem}"
        work_items.append((samples, config_id, model_id, no_rag))
        logger.info(f"JSON source: {p.name} → {len(samples)} samples, model={model_id}")

    # DB sources
    if not args.json or args.config_id:
        db_conn = sqlite3.connect(str(DB_PATH))
        if args.config_id:
            # Specific config IDs
            for cid in args.config_id:
                match = next((c for c in CANONICAL_CONFIGS if c[0] == cid), None)
                if match:
                    _, model_id, no_rag = match
                else:
                    # Try to get model info from configs table
                    row = db_conn.execute(
                        "SELECT model_id, no_rag FROM configs WHERE id = ?", (cid,)
                    ).fetchone()
                    if row:
                        model_id, no_rag = row
                    else:
                        logger.warning(f"Config {cid} not found, skipping")
                        continue
                samples = get_db_samples(db_conn, cid, split)
                if samples:
                    work_items.append((samples, cid, model_id, no_rag))
        elif not args.json:
            # All canonical configs (default when no --json specified)
            for cid, model_id, no_rag in CANONICAL_CONFIGS:
                samples = get_db_samples(db_conn, cid, split)
                if samples:
                    work_items.append((samples, cid, model_id, no_rag))
        db_conn.close()

    if not work_items:
        logger.error("No work items found")
        sys.exit(1)

    logger.info(f"Scoring {len(work_items)} configs × {len(variants)} variants = "
                f"{len(work_items) * len(variants)} runs")

    # Score each config × variant
    for variant in variants:
        conn = sqlite3.connect(str(DB_PATH))
        done_keys = get_done_keys(conn, judge_model, variant["name"], split)
        conn.close()

        for samples, config_id, model_id, no_rag in work_items:
            already, scored = await score_batch(
                samples, config_id, model_id, no_rag, variant, judge_model,
                split, semaphore, api_url, api_headers, done_keys, args.dry_run
            )
            total_already += already
            total_scored += scored
            # Update done_keys with newly scored
            done_keys.update((config_id, s["id"]) for s in samples)

    elapsed = time.monotonic() - t0
    logger.info(f"Done. Scored {total_scored} new, {total_already} pre-existing. {elapsed:.0f}s")
    print(f"\nComplete: {total_scored} scored, {total_already} pre-existing, {elapsed:.0f}s elapsed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified judge scoring — any model, variant, prediction source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # All canonical models, v1, mini:
  uv run python scripts/judge_score.py --variant 1 --judge-model gpt-4.1-mini

  # Rebuttal JSONs only:
  uv run python scripts/judge_score.py --variant 1 --judge-model gpt-4.1-mini \\
      --json data/rebuttal/qwen25-72b_no_rag.json \\
      --json data/rebuttal/gemma2-27b_no_rag.json

  # Multiple variants at once:
  uv run python scripts/judge_score.py --variant 1 --variant 3 --judge-model gpt-4.1-mini
""")
    parser.add_argument("--variant", type=str, action="append", required=True,
                        help="Variant number (1-10) or path to variant JSON. Repeatable.")
    parser.add_argument("--judge-model", type=str, required=True,
                        help="Judge model (e.g. gpt-4.1-mini, gpt-4.1)")
    parser.add_argument("--json", type=str, action="append",
                        help="Path to prediction JSON file. Repeatable. If set, only scores these.")
    parser.add_argument("--config-id", type=str, action="append",
                        help="Specific DB config ID(s) to score. Repeatable.")
    parser.add_argument("--split", type=str, default="test",
                        help="Split to score (default: test)")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Max concurrent API calls (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be scored without calling API")
    args = parser.parse_args()
    asyncio.run(main(args))
