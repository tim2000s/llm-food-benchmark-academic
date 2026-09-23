#!/usr/bin/env python3
"""Analyses for the revised manuscript: photograph-level inference, threshold
sensitivity, parse failures and repeated-query aggregation.

The first version of the manuscript treated each of the 26,904 queries as an
independent observation. They are 13 photographs measured repeatedly, so every
between-model comparison here uses the photograph as the unit: paired across
photographs for tests, and bootstrapped over photographs (5,000 resamples) for
intervals. Query-level intervals are reported alongside in one table to show
the difference.

Sections, in the order the manuscript uses them:
 1. Per-photograph reproducibility by model, and paired comparisons of CV.
 2. Accuracy at photograph level: paired MAE differences with intervals, and
    the number of photographs a given difference would need.
 3. Sensitivity of the dosing-error rates to the insulin-to-carbohydrate ratio
    (1 U per 5 g, 10 g and 20 g).
 4. Parse failures by model and photograph.
 5. Repeated-query aggregation: the median of k queries for k = 1, 3, 5 and 20,
    resampled with replacement from the queries already collected, measured
    against the model's own typical answer and against the reference.

    python3 jamia_revision_analysis.py

Writes figures/Figure_3_aggregation.png, results/jamia_revision.json and
results/JAMIA_REVISION_TABLES.md. No API calls.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
SEED = 20260923
N_BOOT = 5000
DRAWS = 10000
KS = (1, 3, 5, 20)
ICR_RATIOS = (5, 10, 20)          # grams of carbohydrate per unit of insulin
STRONG_TIERS = {1, 2}

FILES = {
    "Claude Sonnet 4.6": "results_anthropic_batch_consolidated.json",
    "GPT-5.4": "results_openai_batch_consolidated_final.json",
    "Gemini 3.1 Pro": "results_gemini31pro_batch.json",
    "Gemini 2.5 Pro": "results_gemini25pro_batch.json",
}
MODELS = list(FILES)
MEALS = {
    "IMG_20200412_154249.jpg": "Roast beef dinner", "IMG_20250915_112359.jpg": "Bakewell tart",
    "IMG_20260209_192318.jpg": "Soup with bread", "IMG_20260210_142258.jpg": "Cheese sandwich",
    "MVIMG_20260222_204918.jpg": "Churros with ice cream", "MVIMG_20260303_200132.jpg": "Chilli con carne on rice",
    "MVIMG_20260304_120022.jpg": "Bakery cookie", "MVIMG_20260308_142023.jpg": "Stuffed pork loin",
    "IMG-20260410-WA0016.jpg": "Breakfast burrito", "IMG-20260410-WA0017.jpg": "Pizza capricciosa",
    "IMG-20260410-WA0018.jpg": "Eggs benedict", "IMG-20260410-WA0019.jpg": "Crema catalana",
    "IMG-20260410-WA0020.jpg": "Paella",
}


def total_carbs_g(food_items) -> float:
    return sum((fi.get("carbs_per_100") or 0) * (fi.get("portion_estimate_size") or 0) / 100
               for fi in (food_items or []))


def load() -> tuple[dict, dict, dict]:
    """Returns per-model {photograph: [estimates]}, per-model rows, and the references."""
    reference = json.load(open(BASE / "usda_reference.json"))
    totals, rows = {}, {}
    for model, fname in FILES.items():
        R = json.load(open(RESULTS / fname))["results"]
        rows[model] = R
        by = collections.defaultdict(dict)
        for r in R:
            if r["success"]:
                by[r["image_file"]].setdefault(r["iteration"], total_carbs_g(r["food_items"]))
        totals[model] = {img: [v for _, v in sorted(d.items())] for img, d in by.items()}
    return totals, rows, reference


def boot_ci(values: list[float], stat=np.mean, rng=None) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(N_BOOT)])
    return float(stat(v)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_photograph(totals: dict) -> dict:
    out = {}
    for model, by in totals.items():
        out[model] = {}
        for img, vals in by.items():
            v = np.asarray(vals)
            p5, p95 = np.percentile(v, [5, 95])
            out[model][img] = {
                "n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
                "cv_pct": float(100 * v.std(ddof=1) / v.mean()),
                "range_g": float(v.max() - v.min()), "p5_p95_g": float(p95 - p5),
            }
    return out


def reproducibility(pp: dict, rng) -> dict:
    out = {"by_model": {}, "paired_cv": []}
    for model, imgs in pp.items():
        cvs = [d["cv_pct"] for d in imgs.values()]
        ranges = [d["range_g"] for d in imgs.values()]
        out["by_model"][model] = {
            "n_photographs": len(cvs),
            "cv_median": boot_ci(cvs, np.median, rng),
            "cv_mean": float(np.mean(cvs)), "cv_max": float(max(cvs)),
            "insulin_range_median_u": boot_ci([r / 10 for r in ranges], np.median, rng),
            "photographs_range_over_2u": int(sum(r > 20 for r in ranges)),
            "photographs_range_over_5u": int(sum(r > 50 for r in ranges)),
        }
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            imgs = sorted(set(pp[a]) & set(pp[b]))
            da = [pp[a][x]["cv_pct"] for x in imgs]
            db = [pp[b][x]["cv_pct"] for x in imgs]
            diff = np.array(da) - np.array(db)
            out["paired_cv"].append({
                "a": a, "b": b, "n": len(imgs),
                "median_diff_points": float(np.median(diff)),
                "a_higher_in": int((diff > 0).sum()),
                "wilcoxon_p": float(stats.wilcoxon(da, db).pvalue),
            })
    return out


def accuracy(totals: dict, reference: dict, rng) -> dict:
    def subset(tiers):
        return [img for img, r in reference.items()
                if isinstance(r, dict) and r.get("reference_quality") in tiers]

    out = {"by_model": {}, "paired_mae": [], "photographs_needed": {}}
    sets = {"strong": subset(STRONG_TIERS), "all_nine": subset({1, 2, 3, 4})}
    for name, photos in sets.items():
        for model, by in totals.items():
            per_img_mae, errs = [], []
            for img in photos:
                ref = reference[img]["total_portion_carbs_g"]
                e = np.asarray(by[img]) - ref
                per_img_mae.append(float(np.mean(np.abs(e))))
                errs.append(e)
            allerr = np.concatenate(errs)
            se = allerr.std(ddof=1) / np.sqrt(len(allerr))  # query-level, for comparison only
            qmae = float(np.mean(np.abs(allerr)))
            qse = np.abs(allerr).std(ddof=1) / np.sqrt(len(allerr))
            out["by_model"].setdefault(model, {})[name] = {
                "n_photographs": len(photos), "n_queries": int(len(allerr)),
                "mae_photograph_level": boot_ci(per_img_mae, np.mean, rng),
                "mae_query_level_ci": (qmae, float(qmae - 1.96 * qse), float(qmae + 1.96 * qse)),
                "bias_g": float(np.mean(allerr)),
                "within_20g_pct": float(np.mean(np.abs(allerr) <= 20) * 100),
            }
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            diffs = []
            for img in sets["all_nine"]:
                ref = reference[img]["total_portion_carbs_g"]
                diffs.append(float(np.mean(np.abs(np.asarray(totals[a][img]) - ref))
                                   - np.mean(np.abs(np.asarray(totals[b][img]) - ref))))
            est, lo, hi = boot_ci(diffs, np.mean, rng)
            sd = float(np.std(diffs, ddof=1))
            out["paired_mae"].append({
                "a": a, "b": b, "n": len(diffs), "mean_diff_g": est, "ci": (lo, hi),
                "a_lower_in": int(sum(d < 0 for d in diffs)),
                "wilcoxon_p": float(stats.wilcoxon(diffs).pvalue),
                "sd_g": sd, "n_for_5g": int(np.ceil((z * sd / 5) ** 2)),
                "n_for_observed": int(np.ceil((z * sd / abs(est)) ** 2)) if abs(est) > 0.01 else None,
            })
    return out


def dosing_thresholds(totals: dict, reference: dict, rng) -> dict:
    """Share of queries implying an overdose above 2 U and 5 U at each ratio,
    on the strong-reference photographs, for one query and for a 20-query median."""
    photos = [img for img, r in reference.items()
              if isinstance(r, dict) and r.get("reference_quality") in STRONG_TIERS]
    out = {}
    for model, by in totals.items():
        out[model] = {}
        for g in ICR_RATIOS:
            one_2u, one_5u, twenty_2u = [], [], []
            for img in photos:
                ref = reference[img]["total_portion_carbs_g"]
                v = np.asarray(by[img], dtype=float)
                e = v - ref
                med = np.median(v[rng.integers(0, len(v), size=(DRAWS, 20))], axis=1) - ref
                one_2u.append(float(np.mean(e > 2 * g)))
                one_5u.append(float(np.mean(e > 5 * g)))
                twenty_2u.append(float(np.mean(med > 2 * g)))
            out[model][g] = {"one_query_2u": float(np.mean(one_2u)),
                             "one_query_5u": float(np.mean(one_5u)),
                             "twenty_query_2u": float(np.mean(twenty_2u))}
    return out


def failures(rows: dict) -> dict:
    out = {}
    for model, R in rows.items():
        bad = [r for r in R if not r["success"]]
        per = collections.Counter(r["image_file"] for r in bad)
        counts = collections.Counter(r["image_file"] for r in R)
        out[model] = {
            "submitted": len(R), "parsed": len(R) - len(bad), "failed": len(bad),
            "failed_pct": 100 * len(bad) / len(R),
            "per_photograph": {MEALS.get(img, img): {"failed": n, "of": counts[img],
                                                     "pct": 100 * n / counts[img]}
                               for img, n in per.items()},
            "messages": sorted({(r.get("error") or "")[:60] for r in bad}),
        }
    return out


def aggregation(totals: dict, reference: dict, rng) -> dict:
    """Median of k queries, resampled with replacement. Variability risk is the
    share of k-query medians more than 10 g from the model's own median for the
    photograph; accuracy risk is the share implying an overdose above 2 U at
    1 U per 10 g on the strong-reference photographs."""
    strong = [img for img, r in reference.items()
              if isinstance(r, dict) and r.get("reference_quality") in STRONG_TIERS]
    out = {}
    for model, by in totals.items():
        out[model] = {}
        for k in KS:
            off, width, over, mae, per_photo = [], [], [], [], {}
            for img, vals in by.items():
                v = np.asarray(vals, dtype=float)
                est = np.median(v[rng.integers(0, len(v), size=(DRAWS, k))], axis=1)
                own = float(np.median(v))
                p5, p95 = np.percentile(est, [5, 95])
                off.append(float(np.mean(np.abs(est - own) > 10)))
                width.append(float(p95 - p5))
                if img in strong:
                    ref = reference[img]["total_portion_carbs_g"]
                    over.append(float(np.mean(est - ref > 20)))
                    mae.append(float(np.mean(np.abs(est - ref))))
                    per_photo[MEALS.get(img, img)] = {
                        "typical_error_g": own - ref, "over_2u": float(np.mean(est - ref > 20))}
            out[model][k] = {
                "off_own_10g_pct": 100 * float(np.mean(off)),
                "width_median_g": float(np.median(width)),
                "over_2u_strong_pct": 100 * float(np.mean(over)),
                "mae_strong_g": float(np.mean(mae)),
                "per_photograph_strong": per_photo,
            }
    return out


def figure_aggregation(agg: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3dd", "#ffffff"
    colours = {"Claude Sonnet 4.6": "#2a78d6", "GPT-5.4": "#eb6834",
               "Gemini 3.1 Pro": "#1baf7a", "Gemini 2.5 Pro": "#eda100"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=300)
    fig.patch.set_facecolor(surface)
    panels = (("off_own_10g_pct", "(a) Estimates more than 1 U from the\nmodel's own typical answer (13 photographs)"),
              ("over_2u_strong_pct", "(b) Estimates implying an overdose above\n2 U (5 strong-reference photographs)"))
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(surface)
        for model, col in colours.items():
            ks = sorted(agg[model])
            ax.plot(ks, [agg[model][k][key] for k in ks], color=col, lw=2, marker="o", ms=4, label=model)
        ax.set_xscale("log")
        ax.set_xticks(list(KS), [str(k) for k in KS])
        ax.minorticks_off()
        ax.set_xlim(0.85, 24)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Queries aggregated (median)", fontsize=8, color=muted)
        ax.set_ylabel("% of estimates", fontsize=8, color=muted)
        ax.set_title(title, fontsize=8, color=ink, loc="left")
        ax.tick_params(labelsize=7.5, colors=muted)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, fontsize=7.5, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, facecolor=surface)
    fig.savefig(path.with_suffix(".pdf"), facecolor=surface)
    plt.close(fig)


def write_tables(res: dict, path: Path) -> None:
    L = ["# Tables for the revised manuscript", "",
         "Generated by jamia_revision_analysis.py. Intervals are 95% bootstrap intervals over "
         "photographs unless stated otherwise.", "",
         "## Reproducibility by model (photograph level)", "",
         "| Model | Median CV | Mean CV | Max CV | Median insulin range (U) | Photographs > 2 U | > 5 U |",
         "|---|---|---|---|---|---|---|"]
    for m, d in res["reproducibility"]["by_model"].items():
        c, i = d["cv_median"], d["insulin_range_median_u"]
        L.append(f"| {m} | {c[0]:.1f}% ({c[1]:.1f}–{c[2]:.1f}) | {d['cv_mean']:.1f}% | {d['cv_max']:.1f}% | "
                 f"{i[0]:.1f} ({i[1]:.1f}–{i[2]:.1f}) | {d['photographs_range_over_2u']}/13 | "
                 f"{d['photographs_range_over_5u']}/13 |")

    L += ["", "## Paired comparisons of within-photograph CV (Wilcoxon, 13 photographs)", "",
          "| Comparison | Median difference (points) | First higher in | p |", "|---|---|---|---|"]
    for c in res["reproducibility"]["paired_cv"]:
        L.append(f"| {c['a']} vs {c['b']} | {c['median_diff_points']:+.1f} | {c['a_higher_in']}/13 | {c['wilcoxon_p']:.4f} |")

    for name, label in (("strong", "five strong-reference photographs"), ("all_nine", "all nine reference photographs")):
        L += ["", f"## Accuracy, {label}", "",
              "| Model | MAE (photograph-level 95% CI) | MAE (query-level 95% CI) | Bias (g) | Within 20 g |",
              "|---|---|---|---|---|"]
        for m, d in res["accuracy"]["by_model"].items():
            a = d[name]
            p, q = a["mae_photograph_level"], a["mae_query_level_ci"]
            L.append(f"| {m} | {p[0]:.1f} ({p[1]:.1f}–{p[2]:.1f}) | {q[0]:.1f} ({q[1]:.1f}–{q[2]:.1f}) | "
                     f"{a['bias_g']:+.1f} | {a['within_20g_pct']:.1f}% |")

    L += ["", "## Paired MAE differences over the nine reference photographs", "",
          "| Comparison | Mean difference (g, 95% CI) | First lower in | Wilcoxon p | Photographs for 5 g | For the observed difference |",
          "|---|---|---|---|---|---|"]
    for c in res["accuracy"]["paired_mae"]:
        n_obs = c["n_for_observed"] if c["n_for_observed"] else "n/a"
        L.append(f"| {c['a']} vs {c['b']} | {c['mean_diff_g']:+.1f} ({c['ci'][0]:+.1f} to {c['ci'][1]:+.1f}) | "
                 f"{c['a_lower_in']}/9 | {c['wilcoxon_p']:.3f} | {c['n_for_5g']} | {n_obs} |")

    L += ["", "## Dosing-error rates by insulin-to-carbohydrate ratio (strong-reference photographs)", "",
          "| Model | 1 U/5 g: > 2 U | > 5 U | 1 U/10 g: > 2 U | > 5 U | 1 U/20 g: > 2 U | > 5 U |",
          "|---|---|---|---|---|---|---|"]
    for m, d in res["dosing_thresholds"].items():
        L.append(f"| {m} | " + " | ".join(f"{100*d[g]['one_query_2u']:.1f}% | {100*d[g]['one_query_5u']:.1f}%"
                                          for g in ICR_RATIOS) + " |")

    L += ["", "## Parse failures", "", "| Model | Submitted | Parsed | Failed | Failure rate | Photographs affected |",
          "|---|---|---|---|---|---|"]
    for m, d in res["failures"].items():
        per = "; ".join(f"{k} {v['failed']}/{v['of']}" for k, v in d["per_photograph"].items()) or "none"
        L.append(f"| {m} | {d['submitted']} | {d['parsed']} | {d['failed']} | {d['failed_pct']:.2f}% | {per} |")

    L += ["", "## Aggregating repeated queries (median of k, resampled with replacement)", "",
          "| Model | k | More than 1 U off own answer | P5–P95 width (g) | > 2 U overdose (strong) | MAE (strong, g) |",
          "|---|---|---|---|---|---|"]
    for m, d in res["aggregation"].items():
        for k in KS:
            a = d[k]
            L.append(f"| {m} | {k} | {a['off_own_10g_pct']:.1f}% | {a['width_median_g']:.1f} | "
                     f"{a['over_2u_strong_pct']:.1f}% | {a['mae_strong_g']:.1f} |")

    L += ["", "### Per-photograph overdose rates, one query against 20 (strong reference)", "",
          "| Model | Photograph | Typical error (g) | 1 query | 20 queries |", "|---|---|---|---|---|"]
    for m, d in res["aggregation"].items():
        for meal, p in d[1]["per_photograph_strong"].items():
            p20 = d[20]["per_photograph_strong"][meal]["over_2u"]
            if p["over_2u"] > 0 or p20 > 0:
                L.append(f"| {m} | {meal} | {p['typical_error_g']:+.1f} | {100*p['over_2u']:.1f}% | {100*p20:.1f}% |")
    path.write_text("\n".join(L) + "\n")


def main():
    rng = np.random.default_rng(SEED)
    totals, rows, reference = load()
    pp = per_photograph(totals)
    res = {
        "seed": SEED, "n_boot": N_BOOT, "draws": DRAWS,
        "per_photograph": pp,
        "reproducibility": reproducibility(pp, rng),
        "accuracy": accuracy(totals, reference, rng),
        "dosing_thresholds": dosing_thresholds(totals, reference, rng),
        "failures": failures(rows),
        "aggregation": aggregation(totals, reference, rng),
    }
    FIGURES.mkdir(exist_ok=True)
    figure_aggregation(res["aggregation"], FIGURES / "Figure_3_aggregation.png")
    (RESULTS / "jamia_revision.json").write_text(json.dumps(res, indent=1, default=float))
    write_tables(res, RESULTS / "JAMIA_REVISION_TABLES.md")
    print((RESULTS / "JAMIA_REVISION_TABLES.md").read_text())


if __name__ == "__main__":
    main()
