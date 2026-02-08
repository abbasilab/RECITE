"""
Generate markdown summary tables from benchmark_predictions/ run directories.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunRecord:
    """One run: model_id, run dir name, optional top_k, no_rag, run_started_at, splits metrics."""

    model_id: str
    run_id: str  # e.g. run_2025-01-29T20-45-00Z
    top_k: Optional[int] = None
    no_rag: Optional[bool] = None
    run_started_at: Optional[str] = None
    splits: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)  # split -> metrics -> {mean, std, min, max}


def _load_json_with_nan(path: Path) -> Any:
    """Load JSON file; replace literal NaN with null so json.load works."""
    text = path.read_text()
    # Allow NaN in JSON (written by some dumpers)
    text = re.sub(r"\bNaN\b", "null", text)
    return json.loads(text)


def _field_explanations_md(include_run_config: bool = True) -> str:
    """Return markdown describing table columns, metrics (range + meaning), and prompts."""
    table_cols = """### Table columns
| Column | Description |
|--------|-------------|
| **Model** | Model ID from config (e.g. versa-4o, versa-4o-mini). |
| **Run** | Run directory name (timestamp and no_rag / topk{N}). |
"""
    if include_run_config:
        table_cols += """| **top_k** | RAG retrieval top_k for this run (e.g. 10). |
| **no_rag** | Whether this run used no retrieval (evidence truncated only). |
"""
    table_cols += """
### Metrics (ranges and meaning)
| Metric | Range | Description |
|--------|--------|-------------|
| **binary_correct** | 0 or 1 | Exact string match: 1 iff prediction.strip() == reference.strip(). For revision tasks almost always 0; use llm_judge_binary for acceptability. |
| **bleu** | 0–1 | BLEU (n-gram overlap). Higher = more n-gram overlap with reference. |
| **edit_distance** | ≥ 0 | Levenshtein (character) edit distance; lower = closer. |
| **normalized_edit_distance** | 0–1 | Edit distance / max(len(ref), len(pred)); lower = closer. |
| **edit_similarity** | 0–1 | 1 − normalized_edit_distance; higher = closer. |
| **rouge_l** | 0–1 | ROUGE-L F1 (longest common subsequence); higher = more overlap. |
| **llm_judge_binary** | 0 or 1 | Judge: 0 = not acceptable, 1 = acceptable. **Primary acceptability metric.** |
| **llm_judge_score** | 0–4 | Judge ordinal: 0 = no match, 1 = poor, 2 = partial, 3 = good, 4 = excellent. |
| **llm_judge_normalized** | 0–1 | llm_judge_score / 4 (scale 0–4). |

### Prompts
- **Predictor:** System + user template from `config/benchmark_prompts.json` → `model_prompt.system`, `model_prompt.user_template` (or `user_template_rag` with evidence). Task: given source EC and evidence, output amended EC for target_version.
- **Judge:** `judge_prompt` in same file. Input: target (reference) and predicted EC. Output: two numbers — binary (0/1 acceptable) and ordinal (0–4 quality). Scale is `judge_prompt.score_scale` (default 0–4).

"""
    return "## Field explanations\n\n" + table_cols


def _fmt_cell(value: Any) -> str:
    """Format a metric value for table cell; use — for None/NaN."""
    if value is None:
        return "—"
    try:
        if isinstance(value, (int, float)):
            import math
            if math.isnan(value):
                return "—"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)
    except (TypeError, ValueError):
        pass
    return str(value) if value is not None else "—"


def collect_run_records(
    root_dir: Path,
    include_run_config: bool = True,
) -> List[RunRecord]:
    """
    Discover run dirs under root_dir (model_id/run_*/), load evaluation_summary.json
    and optionally run_config.yaml; return list of RunRecord.
    Skips run dirs that lack evaluation_summary.json.
    """
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        return []

    records: List[RunRecord] = []
    for model_dir in sorted(root_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            summary_path = run_dir / "evaluation_summary.json"
            if not summary_path.exists():
                continue
            try:
                summary = _load_json_with_nan(summary_path)
            except (json.JSONDecodeError, OSError):
                continue
            splits = summary.get("splits") or {}
            top_k: Optional[int] = None
            no_rag: Optional[bool] = None
            run_started_at: Optional[str] = None
            if include_run_config:
                config_path = run_dir / "run_config.yaml"
                if config_path.exists():
                    try:
                        import yaml
                        with open(config_path) as f:
                            cfg = yaml.safe_load(f)
                        if cfg:
                            v = cfg.get("top_k")
                            if isinstance(v, (int, float)):
                                top_k = int(v)
                            no_rag = cfg.get("no_rag")
                            if not isinstance(no_rag, bool):
                                no_rag = None
                            run_started_at = cfg.get("run_started_at")
                    except Exception:
                        pass
            records.append(
                RunRecord(
                    model_id=model_id,
                    run_id=run_dir.name,
                    top_k=top_k,
                    no_rag=no_rag,
                    run_started_at=run_started_at,
                    splits=splits,
                )
            )
    return records


def generate_benchmark_summary_md(
    root_dir: Path,
    output_path: Optional[Path] = None,
    include_run_config: bool = True,
) -> str:
    """
    Build markdown summary tables from benchmark_predictions/ under root_dir.
    One table per split; columns: Model ID, Run, [top_k], [no_rag], then metric means.
    Returns markdown string; if output_path is set, also writes to file.
    """
    root_dir = Path(root_dir)
    records = collect_run_records(root_dir, include_run_config=include_run_config)

    if not records:
        md = "# Benchmark summary\n\nNo runs found.\n"
        if output_path:
            Path(output_path).write_text(md)
        return md

    # Collect all splits and metric names (splits[split] = { count, metrics: { name: { mean, std, ... } } })
    all_splits: List[str] = []
    seen_splits = set()
    metric_names: Dict[str, List[str]] = {}
    for r in records:
        for split_name, split_data in r.splits.items():
            if split_name not in seen_splits:
                seen_splits.add(split_name)
                all_splits.append(split_name)
            m = (split_data.get("metrics") or {}) if isinstance(split_data, dict) else {}
            if isinstance(m, dict):
                names = [k for k in m if isinstance(m.get(k), dict) and "mean" in (m.get(k) or {})]
                if split_name not in metric_names:
                    metric_names[split_name] = []
                for n in names:
                    if n not in metric_names[split_name]:
                        metric_names[split_name].append(n)
    for k in metric_names:
        metric_names[k] = sorted(metric_names[k])
    all_splits = sorted(all_splits)

    lines: List[str] = [
        "# Benchmark summary",
        "",
        "**Note:** `binary_correct` is exact string match (prediction == reference); for revision tasks it is almost always 0. Use **`llm_judge_binary_mean`** (fraction judged acceptable) as the primary acceptability metric.",
        "",
        _field_explanations_md(include_run_config=include_run_config),
    ]
    for split_name in all_splits:
        metrics_list = metric_names.get(split_name, [])
        # Move binary_correct to the end (exact match is a weak metric for revision tasks)
        if "binary_correct" in metrics_list:
            metrics_list = [m for m in metrics_list if m != "binary_correct"] + ["binary_correct"]
        header = ["Model", "Run"]
        if include_run_config:
            header.append("top_k")
            header.append("no_rag")
        header.extend(f"{m}_mean" for m in metrics_list)
        lines.append(f"## {split_name}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in records:
            splits_data = r.splits.get(split_name)
            if not isinstance(splits_data, dict):
                continue
            m = (splits_data.get("metrics") or {}) if isinstance(splits_data, dict) else {}
            row = [r.model_id, r.run_id]
            if include_run_config:
                row.append(_fmt_cell(r.top_k))
                row.append("yes" if r.no_rag else ("no" if r.no_rag is False else "—"))
            for mn in metrics_list:
                mean_val = None
                if isinstance(m.get(mn), dict):
                    mean_val = (m.get(mn) or {}).get("mean")
                row.append(_fmt_cell(mean_val))
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")
    md = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(md)
    return md
