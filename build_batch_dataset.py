#!/usr/bin/env python3
"""
Build a unified batch-only dataset from all four providers.

Reads the consolidated/individual batch result files, filters to batch data
only, renumbers iterations per (model, image) to avoid collisions from
multi-part submissions, and writes a single analysis-ready JSON.

Usage:
    python3 build_batch_dataset.py

Output:
    results/batch_dataset_all_models.json
"""
import json
import glob
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Mapping of model names to their batch result file patterns.
# Gemini files use a generic "results_batch_" prefix (pre-refactor naming).
MODEL_FILES = {
    "gpt-5.4": "results_openai_batch_consolidated_final.json",
    "claude-sonnet-4-6": "results_anthropic_batch_*.json",
    "gemini-2.5-pro": "results_batch_20260411_113610.json",
    "gemini-3.1-pro-preview": "results_batch_20260411_113740.json",
}


def load_model_results(model: str, pattern: str) -> list[dict]:
    """Load all results for a model from matching files."""
    results = []
    if "*" not in pattern:
        # Single file
        path = RESULTS_DIR / pattern
        if path.exists():
            d = json.loads(path.read_text())
            results.extend(r for r in d["results"] if r.get("success"))
    else:
        for f in sorted(glob.glob(str(RESULTS_DIR / pattern))):
            d = json.loads(Path(f).read_text())
            results.extend(r for r in d["results"] if r.get("success"))
    return results


def main():
    all_results = []
    all_images = set()
    model_counts = {}

    for model, pattern in MODEL_FILES.items():
        results = load_model_results(model, pattern)

        # Renumber iterations per image to avoid multi-part collisions
        seen = defaultdict(int)
        for r in sorted(results, key=lambda x: (x["image_file"], x["iteration"])):
            seen[r["image_file"]] += 1
            r["iteration"] = seen[r["image_file"]]
            all_images.add(r["image_file"])

        model_counts[model] = len(results)
        all_results.extend(results)
        print(f"  {model}: {len(results)} results")

    print(f"\nTotal: {len(all_results)} results across {len(model_counts)} models")
    print(f"Images: {len(all_images)}")

    output = {
        "meta": {
            "description": "Batch-only dataset for academic analysis",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "models": list(model_counts.keys()),
            "model_counts": model_counts,
            "n_images": len(all_images),
            "images": sorted(all_images),
            "total_results": len(all_results),
        },
        "results": all_results,
    }

    out_path = RESULTS_DIR / "batch_dataset_all_models.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
