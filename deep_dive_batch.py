#!/usr/bin/env python3
"""
Deep statistical analysis of the batch-only dataset for the Diabetologia
submission. Four models × 13 images × ~500–560 iterations each.

Primary outcome: within-image consistency / reproducibility of carbohydrate
estimates. Secondary: accuracy vs USDA/user reference values.

Usage:
    python3 deep_dive_batch.py

Reads:
    results/batch_dataset_all_models.json   (built by build_batch_dataset.py)
    usda_reference.json                     (ground-truth carb values)

Writes:
    results/batch_analysis_results.json     (full machine-readable output)
    Prints a human-readable summary to stdout.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
DATASET_PATH = RESULTS_DIR / "batch_dataset_all_models.json"
REFERENCE_PATH = SCRIPT_DIR / "usda_reference.json"

# ---------------------------------------------------------------------------
# Clinical constants
# ---------------------------------------------------------------------------
ICR = 10  # insulin-to-carb ratio (1 unit per 10g carbs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def total_carbs_g(food_items: list[dict]) -> float:
    """Sum of (carbs_per_100 * portion_estimate_size / 100) across items."""
    return sum(
        (fi.get("carbs_per_100") or 0) * (fi.get("portion_estimate_size") or 0) / 100
        for fi in (food_items or [])
    )


def ci_95(data: list[float]) -> tuple[float, float]:
    """95% confidence interval via t-distribution."""
    n = len(data)
    if n < 2:
        return (float("nan"), float("nan"))
    m = np.mean(data)
    se = sp_stats.sem(data)
    lo, hi = sp_stats.t.interval(0.95, n - 1, loc=m, scale=se)
    return (float(lo), float(hi))


def cohens_d(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load():
    dataset = json.loads(DATASET_PATH.read_text())
    reference = json.loads(REFERENCE_PATH.read_text())
    # Strip comment keys
    reference = {k: v for k, v in reference.items() if not k.startswith("_")}
    return dataset, reference


# ---------------------------------------------------------------------------
# Section 1: Dataset overview & reliability
# ---------------------------------------------------------------------------

def analyse_reliability(results: list[dict]) -> dict:
    by_model = defaultdict(lambda: {"total": 0, "success": 0, "fail": 0, "errors": defaultdict(int)})
    for r in results:
        m = r["model"]
        by_model[m]["total"] += 1
        if r["success"]:
            by_model[m]["success"] += 1
        else:
            by_model[m]["fail"] += 1
            ec = r.get("error_class") or "unknown"
            by_model[m]["errors"][ec] += 1
    out = {}
    for model, d in sorted(by_model.items()):
        out[model] = {
            "n_total": d["total"],
            "n_success": d["success"],
            "n_fail": d["fail"],
            "success_rate_pct": round(100 * d["success"] / d["total"], 2) if d["total"] else 0,
            "error_types": dict(d["errors"]),
        }
    return out


# ---------------------------------------------------------------------------
# Section 2: Consistency — within-image variation (PRIMARY OUTCOME)
# ---------------------------------------------------------------------------

def analyse_consistency(results: list[dict]) -> dict:
    # Group successful results by (model, image)
    groups = defaultdict(list)
    for r in results:
        if r["success"]:
            groups[(r["model"], r["image_file"])].append(total_carbs_g(r.get("food_items")))

    per_model = defaultdict(lambda: {"per_image": {}, "summary": {}})

    for (model, image), vals in sorted(groups.items()):
        if len(vals) < 10:
            continue
        arr = np.array(vals)
        m = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))
        cv = 100 * sd / m if m > 0 else 0
        rng = float(np.max(arr) - np.min(arr))
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        p5, p95 = float(np.percentile(arr, 5)), float(np.percentile(arr, 95))
        iqr = q75 - q25

        # Shapiro-Wilk normality test (sample if n > 5000)
        test_arr = arr if len(arr) <= 5000 else np.random.default_rng(42).choice(arr, 5000, replace=False)
        try:
            sw_stat, sw_p = sp_stats.shapiro(test_arr)
        except Exception:
            sw_stat, sw_p = float("nan"), float("nan")

        per_model[model]["per_image"][image] = {
            "n": len(vals),
            "mean": round(m, 2),
            "median": round(float(np.median(arr)), 2),
            "sd": round(sd, 2),
            "cv_pct": round(cv, 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "range": round(rng, 2),
            "iqr": round(iqr, 2),
            "p5": round(p5, 2),
            "p95": round(p95, 2),
            "p5_p95_range": round(p95 - p5, 2),
            "skewness": round(float(sp_stats.skew(arr)), 3),
            "kurtosis": round(float(sp_stats.kurtosis(arr)), 3),
            "shapiro_W": round(float(sw_stat), 4),
            "shapiro_p": float(sw_p),
            "insulin_range_U": round(rng / ICR, 2),
            "insulin_iqr_U": round(iqr / ICR, 2),
            "insulin_p5_p95_U": round((p95 - p5) / ICR, 2),
        }

    # Summaries per model
    for model, data in per_model.items():
        imgs = data["per_image"]
        if not imgs:
            continue
        cvs = [v["cv_pct"] for v in imgs.values()]
        ranges = [v["range"] for v in imgs.values()]
        iqrs = [v["iqr"] for v in imgs.values()]
        p5p95s = [v["p5_p95_range"] for v in imgs.values()]
        ins_ranges = [v["insulin_range_U"] for v in imgs.values()]
        non_normal = sum(1 for v in imgs.values() if v["shapiro_p"] < 0.05)

        data["summary"] = {
            "n_images_analysed": len(imgs),
            "cv_median": round(float(np.median(cvs)), 2),
            "cv_mean": round(float(np.mean(cvs)), 2),
            "cv_max": round(float(np.max(cvs)), 2),
            "cv_min": round(float(np.min(cvs)), 2),
            "range_median": round(float(np.median(ranges)), 2),
            "range_mean": round(float(np.mean(ranges)), 2),
            "range_max": round(float(np.max(ranges)), 2),
            "iqr_median": round(float(np.median(iqrs)), 2),
            "iqr_mean": round(float(np.mean(iqrs)), 2),
            "p5_p95_median": round(float(np.median(p5p95s)), 2),
            "p5_p95_mean": round(float(np.mean(p5p95s)), 2),
            "insulin_range_median_U": round(float(np.median(ins_ranges)), 2),
            "insulin_range_max_U": round(float(np.max(ins_ranges)), 2),
            "images_over_2U_insulin": sum(1 for r in ins_ranges if r > 2),
            "images_over_5U_insulin": sum(1 for r in ins_ranges if r > 5),
            "n_non_normal_distributions": non_normal,
        }

    return dict(per_model)


# ---------------------------------------------------------------------------
# Section 3: Accuracy vs reference values (SECONDARY OUTCOME)
# ---------------------------------------------------------------------------

QUALITY_LABELS = {1: "packet/label", 2: "weighed/measured", 3: "portioned", 4: "visual estimate"}
STRONG_QUALITY = {1, 2}   # tiers used in stratified analysis
WEAKER_QUALITY = {3, 4}


def _accuracy_stats(errs: list[dict]) -> dict:
    """Compute accuracy statistics from a list of error dicts."""
    if not errs:
        return {"n": 0}
    abs_errs = [e["abs_error"] for e in errs]
    signed_errs = [e["error"] for e in errs]
    abs_pcts = [e["abs_pct_error"] for e in errs]
    n = len(errs)

    mae = float(np.mean(abs_errs))
    mae_ci = ci_95(abs_errs)
    mape = float(np.mean(abs_pcts))
    mape_ci = ci_95(abs_pcts)
    bias = float(np.mean(signed_errs))
    bias_ci = ci_95(signed_errs)

    within_10 = 100 * sum(1 for e in abs_errs if e <= 10) / n
    within_20 = 100 * sum(1 for e in abs_errs if e <= 20) / n
    danger_over = 100 * sum(1 for e in errs if e["pct_error"] > 20) / n
    danger_under = 100 * sum(1 for e in errs if e["pct_error"] < -20) / n
    insulin_2u = 100 * sum(1 for e in abs_errs if e > 20) / n
    insulin_5u = 100 * sum(1 for e in abs_errs if e > 50) / n

    return {
        "n": n,
        "n_images": len(set(e["image"] for e in errs)),
        "mae": round(mae, 2),
        "mae_ci_low": round(mae_ci[0], 2),
        "mae_ci_high": round(mae_ci[1], 2),
        "mape": round(mape, 2),
        "mape_ci_low": round(mape_ci[0], 2),
        "mape_ci_high": round(mape_ci[1], 2),
        "bias": round(bias, 2),
        "bias_ci_low": round(bias_ci[0], 2),
        "bias_ci_high": round(bias_ci[1], 2),
        "within_10g_pct": round(within_10, 1),
        "within_20g_pct": round(within_20, 1),
        "dangerous_over_20pct": round(danger_over, 1),
        "dangerous_under_20pct": round(danger_under, 1),
        "insulin_error_over_2U_pct": round(insulin_2u, 1),
        "insulin_error_over_5U_pct": round(insulin_5u, 1),
    }


def analyse_accuracy(results: list[dict], reference: dict) -> dict:
    # Build per-model error arrays, tagged with reference quality
    by_model = defaultdict(list)
    for r in results:
        if not r["success"]:
            continue
        ref = reference.get(r["image_file"])
        if not ref or ref.get("total_portion_carbs_g") is None:
            continue
        pred = total_carbs_g(r.get("food_items"))
        ref_val = ref["total_portion_carbs_g"]
        error = pred - ref_val
        by_model[r["model"]].append({
            "image": r["image_file"],
            "pred": pred,
            "ref": ref_val,
            "ref_quality": ref.get("reference_quality", 4),
            "error": error,
            "abs_error": abs(error),
            "pct_error": 100 * error / ref_val if ref_val else 0,
            "abs_pct_error": 100 * abs(error) / ref_val if ref_val else 0,
        })

    out = {}
    for model, errs in sorted(by_model.items()):
        strong = [e for e in errs if e["ref_quality"] in STRONG_QUALITY]
        weaker = [e for e in errs if e["ref_quality"] in WEAKER_QUALITY]

        out[model] = {
            "all": _accuracy_stats(errs),
            "strong_reference": _accuracy_stats(strong),
            "weaker_reference": _accuracy_stats(weaker),
        }

    return out


# ---------------------------------------------------------------------------
# Section 4: Per-image accuracy detail
# ---------------------------------------------------------------------------

def analyse_per_image_accuracy(results: list[dict], reference: dict) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r["success"]:
            continue
        groups[r["image_file"]][r["model"]].append(total_carbs_g(r.get("food_items")))

    out = {}
    for image in sorted(groups):
        ref = reference.get(image, {})
        ref_val = ref.get("total_portion_carbs_g")
        ref_range = ref.get("total_portion_carbs_range")
        ref_qual = ref.get("reference_quality")
        img_data = {
            "description": ref.get("description", ""),
            "reference_carbs_g": ref_val,
            "reference_range": ref_range,
            "reference_quality": ref_qual,
            "reference_quality_label": QUALITY_LABELS.get(ref_qual, "none") if ref_qual else "consistency only",
            "models": {},
        }
        for model in sorted(groups[image]):
            vals = groups[image][model]
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))
            img_data["models"][model] = {
                "n": len(vals),
                "mean": round(m, 2),
                "median": round(float(np.median(vals)), 2),
                "sd": round(sd, 2),
                "cv_pct": round(100 * sd / m, 2) if m > 0 else 0,
                "range": round(float(np.max(vals) - np.min(vals)), 2),
                "mae": round(float(np.mean(np.abs(np.array(vals) - ref_val))), 2) if ref_val is not None else None,
            }
        out[image] = img_data
    return out


# ---------------------------------------------------------------------------
# Section 5: Pairwise model comparisons
# ---------------------------------------------------------------------------

def analyse_pairwise(results: list[dict], reference: dict) -> dict:
    """Welch's t-test, Mann-Whitney U, and Cohen's d on absolute errors
    for every pair of models. Computed for all references AND for strong
    references only (quality 1–2)."""
    by_model_all = defaultdict(list)
    by_model_strong = defaultdict(list)
    for r in results:
        if not r["success"]:
            continue
        ref = reference.get(r["image_file"])
        if not ref or ref.get("total_portion_carbs_g") is None:
            continue
        pred = total_carbs_g(r.get("food_items"))
        ae = abs(pred - ref["total_portion_carbs_g"])
        by_model_all[r["model"]].append(ae)
        if ref.get("reference_quality", 4) in STRONG_QUALITY:
            by_model_strong[r["model"]].append(ae)

    def _pw(by_model_dict, label):
        models = sorted(by_model_dict.keys())
        comps = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = by_model_dict[models[i]], by_model_dict[models[j]]
                if len(a) < 2 or len(b) < 2:
                    continue
                t_stat, t_p = sp_stats.ttest_ind(a, b, equal_var=False)
                u_stat, u_p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
                d = cohens_d(a, b)
                comps.append({
                    "subset": label,
                    "model_a": models[i],
                    "model_b": models[j],
                    "n_a": len(a),
                    "n_b": len(b),
                    "mean_abs_error_a": round(float(np.mean(a)), 2),
                    "mean_abs_error_b": round(float(np.mean(b)), 2),
                    "welch_t": round(float(t_stat), 4),
                    "welch_p": float(t_p),
                    "mann_whitney_U": float(u_stat),
                    "mann_whitney_p": float(u_p),
                    "cohens_d": round(d, 4),
                })
        return comps

    return {
        "all": _pw(by_model_all, "all"),
        "strong_reference": _pw(by_model_strong, "strong_reference"),
    }


# ---------------------------------------------------------------------------
# Section 6: Pairwise consistency comparisons (CV-based)
# ---------------------------------------------------------------------------

def analyse_pairwise_consistency(consistency: dict) -> dict:
    """Compare within-image CVs between every pair of models using paired
    tests across the 13 images (same images for all models)."""
    models = sorted(consistency.keys())
    images = set()
    for m in models:
        images.update(consistency[m]["per_image"].keys())
    images = sorted(images)

    comparisons = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            cvs_a = [consistency[models[i]]["per_image"].get(img, {}).get("cv_pct") for img in images]
            cvs_b = [consistency[models[j]]["per_image"].get(img, {}).get("cv_pct") for img in images]
            # Only include images present in both
            paired = [(a, b) for a, b in zip(cvs_a, cvs_b) if a is not None and b is not None]
            if len(paired) < 3:
                continue
            a_arr = np.array([p[0] for p in paired])
            b_arr = np.array([p[1] for p in paired])
            # Wilcoxon signed-rank test (paired, non-parametric)
            try:
                w_stat, w_p = sp_stats.wilcoxon(a_arr, b_arr)
            except Exception:
                w_stat, w_p = float("nan"), float("nan")
            comparisons.append({
                "model_a": models[i],
                "model_b": models[j],
                "n_images": len(paired),
                "median_cv_a": round(float(np.median(a_arr)), 2),
                "median_cv_b": round(float(np.median(b_arr)), 2),
                "mean_cv_a": round(float(np.mean(a_arr)), 2),
                "mean_cv_b": round(float(np.mean(b_arr)), 2),
                "wilcoxon_W": float(w_stat),
                "wilcoxon_p": float(w_p),
            })
    return {"comparisons": comparisons}


# ---------------------------------------------------------------------------
# Section 7: Food item identification consistency
# ---------------------------------------------------------------------------

def analyse_food_items(results: list[dict]) -> dict:
    by_model = defaultdict(list)
    for r in results:
        if not r["success"]:
            continue
        items = r.get("food_items") or []
        by_model[r["model"]].append(len(items))

    out = {}
    for model, counts in sorted(by_model.items()):
        arr = np.array(counts)
        out[model] = {
            "n": len(counts),
            "mean_items_per_query": round(float(np.mean(arr)), 2),
            "median_items_per_query": round(float(np.median(arr)), 2),
            "sd": round(float(np.std(arr, ddof=1)), 2),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
        }
    return out


# ---------------------------------------------------------------------------
# Section 8: Token usage
# ---------------------------------------------------------------------------

def analyse_tokens(results: list[dict]) -> dict:
    by_model = defaultdict(lambda: {"input": [], "output": []})
    for r in results:
        if not r["success"]:
            continue
        inp = r.get("input_tokens")
        out = r.get("output_tokens")
        if inp is not None:
            by_model[r["model"]]["input"].append(inp)
        if out is not None:
            by_model[r["model"]]["output"].append(out)

    out = {}
    for model, tokens in sorted(by_model.items()):
        d = {}
        for kind in ("input", "output"):
            arr = tokens[kind]
            if arr:
                d[f"{kind}_mean"] = round(float(np.mean(arr)), 0)
                d[f"{kind}_median"] = round(float(np.median(arr)), 0)
                d[f"{kind}_sd"] = round(float(np.std(arr, ddof=1)), 0)
                d[f"{kind}_n"] = len(arr)
            else:
                d[f"{kind}_n"] = 0
        out[model] = d
    return out


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_summary(analysis: dict):
    print("=" * 80)
    print("BATCH DATASET ANALYSIS — DIABETOLOGIA SUBMISSION")
    print("=" * 80)

    # --- Reliability ---
    print("\n1. RELIABILITY")
    print("-" * 60)
    rel = analysis["reliability"]
    print(f"{'Model':<28} {'N':>6} {'Success':>8} {'Rate':>7}")
    for model, d in sorted(rel.items()):
        print(f"{model:<28} {d['n_total']:>6} {d['n_success']:>8} {d['success_rate_pct']:>6.1f}%")

    # --- Consistency (PRIMARY) ---
    print("\n2. CONSISTENCY — WITHIN-IMAGE VARIATION (PRIMARY OUTCOME)")
    print("-" * 60)
    con = analysis["consistency"]
    print(f"{'Model':<28} {'Med CV%':>8} {'Mean CV%':>9} {'Max CV%':>8} {'Med Range':>10} {'Max Range':>10} {'Non-Normal':>11}")
    for model in sorted(con):
        s = con[model]["summary"]
        print(f"{model:<28} {s['cv_median']:>7.1f}% {s['cv_mean']:>8.1f}% {s['cv_max']:>7.1f}% "
              f"{s['range_median']:>9.1f}g {s['range_max']:>9.1f}g {s['n_non_normal_distributions']:>7}/{s['n_images_analysed']}")

    print(f"\n  Clinical translation (ICR 1:{ICR}):")
    print(f"  {'Model':<28} {'Med ΔU':>8} {'Max ΔU':>8} {'>2U imgs':>9} {'>5U imgs':>9}")
    for model in sorted(con):
        s = con[model]["summary"]
        print(f"  {model:<28} {s['insulin_range_median_U']:>7.1f}U {s['insulin_range_max_U']:>7.1f}U "
              f"{s['images_over_2U_insulin']:>5}/{s['n_images_analysed']}    {s['images_over_5U_insulin']:>5}/{s['n_images_analysed']}")

    # Per-image CV detail
    print(f"\n  Per-image CV% by model:")
    images = sorted(set(img for m in con for img in con[m]["per_image"]))
    header = f"  {'Image':<32}" + "".join(f"{m[:15]:>16}" for m in sorted(con))
    print(header)
    for img in images:
        row = f"  {img:<32}"
        for model in sorted(con):
            v = con[model]["per_image"].get(img)
            if v:
                row += f"{v['cv_pct']:>15.1f}%"
            else:
                row += f"{'—':>16}"
        print(row)

    # --- Accuracy (SECONDARY) ---
    print("\n3. ACCURACY vs REFERENCE VALUES (SECONDARY OUTCOME)")
    print("-" * 60)
    acc = analysis["accuracy"]

    for subset_key, subset_label in [("all", "All 9 reference images"),
                                      ("strong_reference", "Strong reference only (quality 1-2: packet/label, weighed/measured) — 5 images"),
                                      ("weaker_reference", "Weaker reference (quality 3-4: portioned, visual estimate) — 4 images")]:
        print(f"\n  {subset_label}:")
        print(f"  {'Model':<28} {'N':>6} {'imgs':>5} {'MAE':>7} {'MAE 95% CI':>16} {'MAPE':>7} {'Bias':>7} {'≤10g':>6} {'≤20g':>6}")
        for model in sorted(acc):
            d = acc[model].get(subset_key, {})
            if not d or d.get("n", 0) == 0:
                continue
            print(f"  {model:<28} {d['n']:>6} {d.get('n_images',0):>5} {d['mae']:>6.1f}g ({d['mae_ci_low']:.1f}–{d['mae_ci_high']:.1f}) "
                  f"{d['mape']:>6.1f}% {d['bias']:>+6.1f}g {d['within_10g_pct']:>5.1f}% {d['within_20g_pct']:>5.1f}%")

    print(f"\n  Clinical danger zone (all 9 images):")
    print(f"  {'Model':<28} {'Over >20%':>10} {'Under >20%':>11} {'Error >2U':>10} {'Error >5U':>10}")
    for model in sorted(acc):
        d = acc[model]["all"]
        print(f"  {model:<28} {d['dangerous_over_20pct']:>9.1f}% {d['dangerous_under_20pct']:>10.1f}% "
              f"{d['insulin_error_over_2U_pct']:>9.1f}% {d['insulin_error_over_5U_pct']:>9.1f}%")

    # --- Pairwise comparisons ---
    print("\n4. PAIRWISE COMPARISONS — ABSOLUTE ERROR")
    print("-" * 60)
    pw = analysis["pairwise_accuracy"]
    for subset_key, subset_label in [("all", "All 9 reference images"), ("strong_reference", "Strong reference only (5 images)")]:
        print(f"\n  {subset_label}:")
        for c in pw.get(subset_key, []):
            sig = "***" if c["welch_p"] < 0.001 else "**" if c["welch_p"] < 0.01 else "*" if c["welch_p"] < 0.05 else "ns"
            print(f"    {c['model_a'][:20]} vs {c['model_b'][:20]}:  "
                  f"MAE {c['mean_abs_error_a']:.1f} vs {c['mean_abs_error_b']:.1f}g  "
                  f"t={c['welch_t']:.2f}  p={c['welch_p']:.2e}  d={c['cohens_d']:.3f}  {sig}")

    print("\n5. PAIRWISE COMPARISONS — CONSISTENCY (Wilcoxon on CVs)")
    print("-" * 60)
    pwc = analysis["pairwise_consistency"]
    for c in pwc["comparisons"]:
        sig = "***" if c["wilcoxon_p"] < 0.001 else "**" if c["wilcoxon_p"] < 0.01 else "*" if c["wilcoxon_p"] < 0.05 else "ns"
        print(f"  {c['model_a'][:20]} vs {c['model_b'][:20]}:  "
              f"Med CV {c['median_cv_a']:.1f}% vs {c['median_cv_b']:.1f}%  "
              f"W={c['wilcoxon_W']:.0f}  p={c['wilcoxon_p']:.4f}  {sig}")

    # --- Per-image accuracy ---
    print("\n6. PER-IMAGE ACCURACY (mean estimate vs reference)")
    print("-" * 60)
    pia = analysis["per_image_accuracy"]
    for image, data in sorted(pia.items()):
        ref = data["reference_carbs_g"]
        qual = data.get("reference_quality_label", "")
        ref_str = f"{ref}g [{qual}]" if ref is not None else f"no ref [{qual}]"
        print(f"\n  {image} ({ref_str}): {data['description'][:60]}")
        for model, md in sorted(data["models"].items()):
            mae_str = f"MAE {md['mae']:.1f}g" if md["mae"] is not None else "n/a"
            print(f"    {model:<26} mean={md['mean']:>6.1f}g  sd={md['sd']:>5.1f}  CV={md['cv_pct']:>5.1f}%  range={md['range']:>6.1f}g  {mae_str}")

    # --- Food items ---
    print("\n7. FOOD ITEM IDENTIFICATION")
    print("-" * 60)
    fi = analysis["food_items"]
    print(f"{'Model':<28} {'Mean items':>11} {'Median':>8} {'SD':>6} {'Min':>5} {'Max':>5}")
    for model, d in sorted(fi.items()):
        print(f"{model:<28} {d['mean_items_per_query']:>10.2f} {d['median_items_per_query']:>8.1f} "
              f"{d['sd']:>6.2f} {d['min']:>5} {d['max']:>5}")

    # --- Tokens ---
    print("\n8. TOKEN USAGE")
    print("-" * 60)
    tok = analysis["token_usage"]
    for model, d in sorted(tok.items()):
        if d.get("input_n", 0) > 0:
            print(f"  {model:<26} input={d['input_mean']:.0f}±{d['input_sd']:.0f}  "
                  f"output={d['output_mean']:.0f}±{d['output_sd']:.0f}  (n={d['input_n']})")
        else:
            print(f"  {model:<26} no token data")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset, reference = load()
    results = dataset["results"]

    print(f"Dataset: {len(results)} results, {len(dataset['meta']['models'])} models")
    print(f"Reference values: {sum(1 for v in reference.values() if v.get('total_portion_carbs_g') is not None)} images\n")

    analysis = {
        "meta": dataset["meta"],
        "reliability": analyse_reliability(results),
        "consistency": analyse_consistency(results),
        "accuracy": analyse_accuracy(results, reference),
        "per_image_accuracy": analyse_per_image_accuracy(results, reference),
        "pairwise_accuracy": analyse_pairwise(results, reference),
        "pairwise_consistency": {},
        "food_items": analyse_food_items(results),
        "token_usage": analyse_tokens(results),
    }

    # Pairwise consistency needs the consistency section
    analysis["pairwise_consistency"] = analyse_pairwise_consistency(analysis["consistency"])

    # Write JSON
    out_path = RESULTS_DIR / "batch_analysis_results.json"
    out_path.write_text(json.dumps(analysis, indent=2, default=str))
    print(f"Full results written to: {out_path}\n")

    # Print human-readable summary
    print_summary(analysis)


if __name__ == "__main__":
    main()
