"""Stratified re-verification of truncation evidence with stricter prompt.

Draws ~150 stratified subsample from 3,116 RECITE samples, then scores with
both gpt-4.1 and gpt-4.1-mini for 1:1 comparison. Stricter prompt requires
verbatim quote of the specific amended EC text.

Stratification:
  - EC change type (inclusion/exclusion/both/unknown)
  - Version position (early 0-5 / mid 6-15 / late 16+)
  - 4.1-mini label balance (oversample absent to ~30%)
  - Max 1 sample per trial (instance_id)

Usage:
    set -a && source /home/rro/projects/phdmanager/.env && set +a
    uv run python scripts/truncation_stratified_verify.py [--sample-only] [--max-concurrent 10]
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
DB_PATH = ROOT / "data" / "dev" / "recite.db"
SPLITS_DIR = ROOT / "data" / "benchmark_splits"
LOG_DIR = ROOT / "logs"
OUTPUT_TABLE_41 = "truncation_verify_gpt41"
OUTPUT_TABLE_MINI = "truncation_verify_gpt41mini"
SAMPLE_SIZE = 150

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_DIR / "truncation_stratified_verify.log", rotation="10 MB", level="DEBUG")

enc = tiktoken.get_encoding("cl100k_base")

MODELS = {
    "gpt-4.1": {
        "deployment": "gpt-4.1-2025-04-14",
        "table": OUTPUT_TABLE_41,
    },
    "gpt-4.1-mini": {
        "deployment": "gpt-4.1-mini-2025-04-14",
        "table": OUTPUT_TABLE_MINI,
    },
}

# ── Stricter prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical trial protocol analyst performing precise evidence verification.
You must determine whether a SPECIFIC amended eligibility criterion is present in a truncated protocol excerpt.
Be strict: generic EC mentions do NOT count. Only the specific criterion change described matters."""

USER_TEMPLATE = """Below is a TRUNCATED excerpt (first 4096 tokens) from a clinical trial protocol, and the SPECIFIC AMENDED eligibility criterion text that should be found.

TRUNCATED PROTOCOL:
{truncated_evidence}

SPECIFIC AMENDED ELIGIBILITY CRITERION:
{reference_text}

The protocol changed from version {source_version} to version {target_version}.

INSTRUCTIONS — be strict:
1. Search the truncated excerpt for the SPECIFIC amended criterion described above.
2. Generic eligibility criteria mentions do NOT count — you must find text that substantively matches the specific amendment.
3. If present, you MUST quote the exact passage (verbatim, max 300 chars) that contains or describes this specific criterion change.
4. If you cannot find a verbatim passage matching this specific amendment, answer "NOT_FOUND" — do not guess.

Respond ONLY with this JSON (no markdown fences):
{{"present": 0 or 1, "confidence": 0.0-1.0, "reasoning": "1-2 sentence explanation of your judgment", "verbatim_quote": "exact text from excerpt or NOT_FOUND"}}"""


def truncate_evidence(text: str) -> str:
    tokens = enc.encode(text)
    if len(tokens) <= EVIDENCE_MAX_TOKENS:
        return text
    return enc.decode(tokens[:EVIDENCE_MAX_TOKENS])


# ── Load & stratify ───────────────────────────────────────────────────
def load_all_samples() -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        path = SPLITS_DIR / f"{split}.parquet"
        df = pd.read_parquet(path)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def draw_stratified_sample(n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Draw stratified subsample with label oversampling and trial diversity."""
    conn = sqlite3.connect(str(DB_PATH))

    # Load all samples with their mini labels
    samples = load_all_samples()
    logger.info(f"Loaded {len(samples)} total samples")

    # Join with mini labels
    mini_df = pd.read_sql_query(
        "SELECT recite_id, reference_text_present AS mini_label, reference_text_confidence AS mini_conf "
        "FROM truncation_evidence_41mini",
        conn,
    )
    samples = samples.merge(mini_df, left_on="id", right_on="recite_id", how="inner")

    # Join with ec_changes for change type (take first match per sample)
    ec_df = pd.read_sql_query(
        "SELECT DISTINCT instance_id, source_version, target_version, ec_change_type FROM ec_changes",
        conn,
    )
    # Deduplicate: one row per (instance_id, source_version, target_version)
    ec_df = ec_df.drop_duplicates(subset=["instance_id", "source_version", "target_version"], keep="first")
    samples = samples.merge(ec_df, on=["instance_id", "source_version", "target_version"], how="left")
    samples["ec_change_type"] = samples["ec_change_type"].fillna("unknown")

    conn.close()

    # Stratification bins
    samples["version_group"] = pd.cut(
        samples["source_version"],
        bins=[-1, 5, 15, 999],
        labels=["early(0-5)", "mid(6-15)", "late(16+)"],
    )

    # Split into absent vs present pools
    absent_pool = samples[samples["mini_label"] == 0].copy()
    present_pool = samples[samples["mini_label"] == 1].copy()

    logger.info(f"Absent pool: {len(absent_pool)}, Present pool: {len(present_pool)}")

    # Target: ~30% absent, ~70% present
    n_absent = min(int(n * 0.30), len(absent_pool))
    n_present = n - n_absent

    # Sample absent: stratified by version_group × ec_change_type, max 1 per trial
    absent_sample = _stratified_draw(absent_pool, n_absent)
    present_sample = _stratified_draw(present_pool, n_present, exclude_ncts=set(absent_sample["instance_id"]))

    result = pd.concat([absent_sample, present_sample], ignore_index=True)
    logger.info(
        f"Final sample: {len(result)} ({len(absent_sample)} absent, {len(present_sample)} present)"
    )
    return result


def _stratified_draw(pool: pd.DataFrame, n: int, exclude_ncts: set | None = None) -> pd.DataFrame:
    """Draw n samples stratified by version_group × ec_change_type, max 1 per trial."""
    if exclude_ncts:
        pool = pool[~pool["instance_id"].isin(exclude_ncts)]

    # Deduplicate by instance_id (keep first = random-ish)
    pool = pool.drop_duplicates(subset=["instance_id"], keep="first").copy()

    if len(pool) <= n:
        return pool

    # Stratify by version_group × ec_change_type
    pool["stratum"] = pool["version_group"].astype(str) + "|" + pool["ec_change_type"].astype(str)
    strata_counts = pool["stratum"].value_counts()

    # Proportional allocation
    allocated = {}
    remaining = n
    for stratum, count in strata_counts.items():
        alloc = max(1, int(round(n * count / len(pool))))
        allocated[stratum] = min(alloc, count)
        remaining -= allocated[stratum]

    # Distribute remainder
    for stratum in strata_counts.index:
        if remaining <= 0:
            break
        space = strata_counts[stratum] - allocated[stratum]
        if space > 0:
            add = min(remaining, space)
            allocated[stratum] += add
            remaining -= add

    # Draw from each stratum
    drawn = []
    for stratum, alloc in allocated.items():
        stratum_pool = pool[pool["stratum"] == stratum]
        drawn.append(stratum_pool.sample(n=min(alloc, len(stratum_pool)), random_state=42))

    result = pd.concat(drawn, ignore_index=True)
    # Trim to exact n if over-allocated
    if len(result) > n:
        result = result.sample(n=n, random_state=42)
    return result


# ── DB setup ──────────────────────────────────────────────────────────
def ensure_table(conn: sqlite3.Connection, table: str):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            recite_id               INTEGER NOT NULL,
            instance_id                  TEXT NOT NULL,
            source_version            INTEGER,
            target_version              INTEGER,
            ec_change_type          TEXT,
            version_group           TEXT,
            mini_label              INTEGER,
            present                 INTEGER,
            confidence              REAL,
            reasoning               TEXT,
            verbatim_quote          TEXT,
            model_used              TEXT NOT NULL,
            analyzed_at             TEXT NOT NULL,
            raw_response            TEXT,
            PRIMARY KEY (recite_id)
        )
    """)
    conn.commit()


def parse_response(raw: str) -> dict:
    defaults = {"present": None, "confidence": None, "reasoning": None, "verbatim_quote": None}
    if not raw or raw.startswith("ERROR"):
        return defaults
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        data = json.loads(text)
        return {
            "present": int(data.get("present", 0)),
            "confidence": float(data.get("confidence", 0.0)),
            "reasoning": str(data.get("reasoning", ""))[:500],
            "verbatim_quote": str(data.get("verbatim_quote", ""))[:500],
        }
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Parse error: {e} — raw: {raw[:200]}")
        return defaults


# ── Async API ─────────────────────────────────────────────────────────
async def call_versa(session: aiohttp.ClientSession, url: str, headers: dict,
                     system_prompt: str, user_prompt: str,
                     sample_id: int, max_retries: int = 5) -> str:
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
                                    timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", min(2 ** attempt, 60)))
                    logger.warning(f"Sample {sample_id}: 429, retry in {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                data = await resp.json()
                if "choices" in data and data["choices"]:
                    raw = data["choices"][0]["message"].get("content", "")
                    return raw.strip() if isinstance(raw, str) else str(raw)
                return "ERROR: no choices"
        except Exception as e:
            if attempt >= max_retries:
                logger.error(f"Sample {sample_id}: exhausted retries: {e}")
                return f"ERROR: {e}"
            delay = min(2 ** attempt, 60)
            logger.warning(f"Sample {sample_id}: retry {attempt+1} in {delay}s: {e}")
            await asyncio.sleep(delay)
    return "ERROR: unreachable"


async def score_model(samples_df: pd.DataFrame, model_name: str, model_cfg: dict,
                      api_url_base: str, api_headers: dict, max_concurrent: int):
    """Score all samples with one model, write to its table."""
    deployment = model_cfg["deployment"]
    table = model_cfg["table"]
    api_url = f"{api_url_base}/openai/deployments/{deployment}/chat/completions?api-version={os.getenv('UCSF_API_VER')}"

    conn = sqlite3.connect(str(DB_PATH))
    ensure_table(conn, table)

    # Check already done
    try:
        done = {r[0] for r in conn.execute(f"SELECT recite_id FROM {table}").fetchall()}
    except sqlite3.OperationalError:
        done = set()

    todo = samples_df[~samples_df["id"].isin(done)]
    logger.info(f"[{model_name}] TODO: {len(todo)} samples ({len(done)} already done)")

    if todo.empty:
        conn.close()
        return

    semaphore = asyncio.Semaphore(max_concurrent)
    now_str = datetime.now(timezone.utc).isoformat()
    scored = 0
    errors = 0
    t0 = time.monotonic()

    async with aiohttp.ClientSession() as session:
        async def process_one(row):
            nonlocal scored, errors
            async with semaphore:
                truncated = truncate_evidence(row["evidence"])
                user_prompt = USER_TEMPLATE.format(
                    truncated_evidence=truncated,
                    reference_text=row["reference_text"],
                    source_version=row["source_version"],
                    target_version=row["target_version"],
                )
                raw = await call_versa(session, api_url, api_headers,
                                       SYSTEM_PROMPT, user_prompt, row["id"])
                parsed = parse_response(raw)
                if parsed["present"] is None:
                    errors += 1

                conn.execute(
                    f"""INSERT OR REPLACE INTO {table}
                        (recite_id, instance_id, source_version, target_version, ec_change_type,
                         version_group, mini_label, present, confidence, reasoning,
                         verbatim_quote, model_used, analyzed_at, raw_response)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        int(row["id"]), row["instance_id"], int(row["source_version"]),
                        int(row["target_version"]),
                        row.get("ec_change_type", "unknown"),
                        str(row.get("version_group", "")),
                        int(row.get("mini_label", -1)),
                        parsed["present"], parsed["confidence"],
                        parsed["reasoning"], parsed["verbatim_quote"],
                        model_name, now_str, raw,
                    ),
                )
                scored += 1
                if scored % 25 == 0:
                    conn.commit()
                    elapsed = time.monotonic() - t0
                    logger.info(f"[{model_name}] {scored}/{len(todo)} ({scored/elapsed:.1f}/s)")

        await asyncio.gather(*[process_one(row) for _, row in todo.iterrows()])

    conn.commit()

    # Stats
    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    present = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE present = 1").fetchone()[0]
    pct = 100 * present / total if total else 0
    conn.close()

    elapsed = time.monotonic() - t0
    report = (
        f"[{model_name}] Done: {total} scored, {present}/{total} present ({pct:.1f}%), "
        f"{errors} errors, {elapsed:.0f}s"
    )
    logger.info(report)
    return report


async def main(args):
    # Draw sample
    sample_df = draw_stratified_sample(SAMPLE_SIZE)

    # Save sample manifest
    manifest_path = ROOT / "data" / "rebuttal" / "stratified_verify_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df[["id", "instance_id", "source_version", "target_version", "ec_change_type",
               "version_group", "mini_label", "split"]].to_csv(manifest_path, index=False)
    logger.info(f"Saved manifest: {manifest_path} ({len(sample_df)} samples)")

    # Print stratification summary
    print(f"\n{'='*60}")
    print(f"STRATIFIED SAMPLE: {len(sample_df)} samples")
    print(f"{'='*60}")
    print(f"\nMini label distribution:")
    print(sample_df["mini_label"].value_counts().to_string())
    print(f"\nVersion group distribution:")
    print(sample_df["version_group"].value_counts().to_string())
    print(f"\nEC change type distribution:")
    print(sample_df["ec_change_type"].value_counts().to_string())
    print(f"\nUnique trials: {sample_df['instance_id'].nunique()}")
    print(f"{'='*60}\n")

    if args.sample_only:
        logger.info("--sample-only: exiting before API calls")
        return

    # API setup
    api_key = os.getenv("UCSF_API_KEY")
    api_ver = os.getenv("UCSF_API_VER")
    endpoint = os.getenv("UCSF_RESOURCE_ENDPOINT", "").rstrip("/")
    if not all([api_key, api_ver, endpoint]):
        logger.error("Missing UCSF_API_KEY, UCSF_API_VER, or UCSF_RESOURCE_ENDPOINT")
        sys.exit(1)

    api_headers = {"Content-Type": "application/json", "api-key": api_key}

    # Score with both models
    reports = []
    for model_name, model_cfg in MODELS.items():
        report = await score_model(sample_df, model_name, model_cfg, endpoint, api_headers, args.max_concurrent)
        reports.append(report)

    # Cross-comparison
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cross = pd.read_sql_query(f"""
            SELECT g.recite_id, g.present AS gpt41_present, g.confidence AS gpt41_conf,
                   m.present AS mini_present, m.confidence AS mini_conf,
                   g.mini_label AS original_mini_label
            FROM {OUTPUT_TABLE_41} g
            JOIN {OUTPUT_TABLE_MINI} m ON g.recite_id = m.recite_id
        """, conn)

        if not cross.empty:
            agree = (cross["gpt41_present"] == cross["mini_present"]).sum()
            total = len(cross)
            print(f"\n{'='*60}")
            print(f"CROSS-COMPARISON: gpt-4.1 vs gpt-4.1-mini ({total} samples)")
            print(f"Agreement: {agree}/{total} ({100*agree/total:.1f}%)")
            print(f"\n4.1 present rate: {cross['gpt41_present'].mean():.1%}")
            print(f"Mini present rate: {cross['mini_present'].mean():.1%}")
            print(f"Original mini label present rate: {cross['original_mini_label'].mean():.1%}")

            # Confusion matrix
            print(f"\n            | mini=absent | mini=present |")
            for gpt_val, gpt_label in [(0, "4.1=absent "), (1, "4.1=present")]:
                row = cross[cross["gpt41_present"] == gpt_val]
                mini_0 = (row["mini_present"] == 0).sum()
                mini_1 = (row["mini_present"] == 1).sum()
                print(f"  {gpt_label} |     {mini_0:3d}     |      {mini_1:3d}      |")
            print(f"{'='*60}\n")
    except Exception as e:
        logger.warning(f"Cross-comparison failed: {e}")
    finally:
        conn.close()

    # Telegram notification
    summary = "\n".join(r for r in reports if r)
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(
                "http://localhost:8443/api/send",
                json={"chat_id": "7054155159", "text": f"Stratified verification done:\n{summary}"},
                timeout=aiohttp.ClientTimeout(total=10),
            )
    except Exception as e:
        logger.warning(f"Telegram notification failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified truncation evidence re-verification")
    parser.add_argument("--sample-only", action="store_true", help="Draw sample and save manifest, no API calls")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent API calls per model")
    args = parser.parse_args()
    asyncio.run(main(args))
