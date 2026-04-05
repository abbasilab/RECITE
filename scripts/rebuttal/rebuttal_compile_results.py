"""
Compile results from large model evaluations for rebuttal.
Builds scaling trend table and comparison with paper results.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REBUTTAL_DIR = ROOT / "data" / "rebuttal"

# Paper results (from CLAUDE.md and RECITE paper)
PAPER_RESULTS = {
    "GPT-4o-mini": {"binary_equiv": 0.858, "params": "~8B", "type": "proprietary"},
    "Gemma 2 9B": {"binary_equiv": 0.847, "params": "9B", "type": "open"},
    "Qwen2.5-7B": {"binary_equiv": None, "params": "7B", "type": "open"},
    "Qwen2.5-3B": {"binary_equiv": None, "params": "3B", "type": "open"},
    "Qwen2.5-0.5B": {"binary_equiv": None, "params": "0.5B", "type": "open"},
    "Gemma 2 2B": {"binary_equiv": None, "params": "2B", "type": "open"},
    "DeepSeek-R1-7B": {"binary_equiv": None, "params": "7B", "type": "open"},
}


def load_rebuttal_results():
    """Load results from rebuttal JSON files."""
    results = {}
    for f in REBUTTAL_DIR.glob("*_no_rag.json"):
        data = json.loads(f.read_text())
        model = data["model"]
        label = f.stem.replace("_no_rag", "")
        results[label] = {
            "model": model,
            "binary_equiv": data.get("binary_equiv"),
            "mean_ordinal": data.get("mean_ordinal"),
            "n_samples": data.get("n_samples"),
            "n_errors": data.get("n_errors"),
            "n_judged": data.get("n_judged"),
        }
    return results


def format_table(results: dict):
    """Format results as markdown table."""
    lines = []
    lines.append("| Model | Params | Binary Equiv (%) | Mean Ordinal | Samples | Errors |")
    lines.append("|-------|--------|-------------------|--------------|---------|--------|")

    for label, data in sorted(results.items()):
        be = f"{data['binary_equiv']*100:.1f}" if data.get("binary_equiv") is not None else "—"
        mo = f"{data['mean_ordinal']:.2f}" if data.get("mean_ordinal") is not None else "—"
        n = data.get("n_samples", "—")
        err = data.get("n_errors", "—")
        lines.append(f"| {data.get('model', label)} | — | {be} | {mo} | {n} | {err} |")

    return "\n".join(lines)


def main():
    results = load_rebuttal_results()
    if not results:
        print("No results found in", REBUTTAL_DIR)
        return

    print("=" * 60)
    print("REBUTTAL: Large Model Evaluation Results")
    print("=" * 60)
    print()

    for label, data in results.items():
        print(f"--- {label} ---")
        if data.get("binary_equiv") is not None:
            print(f"  Binary Equiv: {data['binary_equiv']*100:.1f}%")
            print(f"  Mean Ordinal: {data['mean_ordinal']:.2f}")
        print(f"  Samples: {data['n_samples']}, Errors: {data['n_errors']}, Judged: {data['n_judged']}")
        print()

    print("Markdown table:")
    print(format_table(results))


if __name__ == "__main__":
    main()
