#!/usr/bin/env python3
"""
Graphical abstract for DTT submission.
Single-panel summary: photo in → 4 models → CV spread → insulin risk.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "results" / "batch_dataset_all_models.json"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_DISPLAY = {
    "claude-sonnet-4-6": "Claude\nSonnet 4.6",
    "gpt-5.4": "GPT-5.4",
    "gemini-3.1-pro-preview": "Gemini\n3.1 Pro",
    "gemini-2.5-pro": "Gemini\n2.5 Pro",
}
MODEL_ORDER_RAW = ["claude-sonnet-4-6", "gpt-5.4", "gemini-3.1-pro-preview", "gemini-2.5-pro"]
MODEL_COLORS = {
    "claude-sonnet-4-6": "#5B6ABF",
    "gpt-5.4": "#2CA58D",
    "gemini-3.1-pro-preview": "#E8A838",
    "gemini-2.5-pro": "#E05555",
}

# Key stats from the paper
STATS = {
    "claude-sonnet-4-6":      {"cv": 2.4,  "insulin_u": 0.9, "overdose_2u": 0.0,  "mae": 8.7},
    "gpt-5.4":                {"cv": 8.4,  "insulin_u": 2.3, "overdose_2u": 36.8, "mae": 17.4},
    "gemini-3.1-pro-preview": {"cv": 10.3, "insulin_u": 2.9, "overdose_2u": 12.2, "mae": 12.5},
    "gemini-2.5-pro":         {"cv": 11.0, "insulin_u": 4.7, "overdose_2u": 19.2, "mae": 16.6},
}


def create_graphical_abstract():
    fig = plt.figure(figsize=(12, 7))

    # Title
    fig.text(0.5, 0.95, "LLM Vision APIs for Carbohydrate Estimation: Reproducibility and Clinical Risk",
             fontsize=14, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.91, "26,904 parsed queries  |  13 food photographs  |  4 models  |  500+ repeats per image per model",
             fontsize=10, ha="center", va="top", color="#555555")

    # ── Left panel: CV comparison (bar chart) ──
    ax1 = fig.add_axes([0.06, 0.12, 0.28, 0.70])

    models = MODEL_ORDER_RAW
    cvs = [STATS[m]["cv"] for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    labels = [MODEL_DISPLAY[m] for m in models]

    bars = ax1.barh(range(len(models)), cvs, color=colors, alpha=0.85, height=0.6, zorder=2)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Median within-image CV (%)", fontsize=10)
    ax1.set_title("Reproducibility", fontsize=11, fontweight="bold", pad=10)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 14)
    ax1.axvline(x=10, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    # Threshold labels sit just above the x-axis inside the panel, clear of the title and the footer.
    ax1.text(10.2, 0.02, "10%", fontsize=7, color="grey", va="bottom",
             transform=blended_transform_factory(ax1.transData, ax1.transAxes))

    # Add value labels on bars
    for i, (bar, cv) in enumerate(zip(bars, cvs)):
        ax1.text(cv + 0.3, i, f"{cv}%", va="center", fontsize=9, fontweight="bold")

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Middle panel: Insulin dosing uncertainty ──
    ax2 = fig.add_axes([0.40, 0.12, 0.25, 0.70])

    insulin = [STATS[m]["insulin_u"] for m in models]

    bars2 = ax2.barh(range(len(models)), insulin, color=colors, alpha=0.85, height=0.6, zorder=2)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels([""] * len(models))  # no duplicate labels
    ax2.set_xlabel("Median insulin uncertainty (U)", fontsize=10)
    ax2.set_title("Dosing Uncertainty", fontsize=11, fontweight="bold", pad=10)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 6.5)

    # Clinical thresholds
    ax2.axvspan(2, 6.5, alpha=0.06, color="orange", zorder=0)
    ax2.axvspan(5, 6.5, alpha=0.08, color="red", zorder=0)
    ax2.axvline(x=2, color="orange", linestyle=":", linewidth=1, alpha=0.6)
    ax2.axvline(x=5, color="red", linestyle=":", linewidth=1, alpha=0.6)
    tr2 = blended_transform_factory(ax2.transData, ax2.transAxes)
    ax2.text(2.1, 0.02, ">2 U", fontsize=7, color="#c8701a", va="bottom", alpha=0.9, transform=tr2)
    ax2.text(5.1, 0.02, ">5 U", fontsize=7, color="#c0392b", va="bottom", alpha=0.9, transform=tr2)

    for i, (bar, u) in enumerate(zip(bars2, insulin)):
        ax2.text(u + 0.15, i, f"{u} U", va="center", fontsize=9, fontweight="bold")

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Right panel: Clinical risk (overdose %) ──
    ax3 = fig.add_axes([0.72, 0.12, 0.25, 0.70])

    overdose = [STATS[m]["overdose_2u"] for m in models]

    # Color bars by risk level
    risk_colors = []
    for o in overdose:
        if o == 0:
            risk_colors.append("#2CA02C")  # green
        elif o < 20:
            risk_colors.append("#FF8C00")  # amber
        else:
            risk_colors.append("#DC143C")  # red

    bars3 = ax3.barh(range(len(models)), overdose, color=risk_colors, alpha=0.85, height=0.6, zorder=2)
    ax3.set_yticks(range(len(models)))
    ax3.set_yticklabels([""] * len(models))
    ax3.set_xlabel("Queries causing >2 U overdose (%)", fontsize=10)
    ax3.set_title("Clinical Risk", fontsize=11, fontweight="bold", pad=10)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 45)

    for i, (bar, pct) in enumerate(zip(bars3, overdose)):
        ax3.text(pct + 1, i, f"{pct}%", va="center", fontsize=9, fontweight="bold")

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ── Bottom annotation ──
    fig.text(0.5, 0.02,
             "Strong-reference data (packet label or weighed)  |  ICR 1:10  |  Street 2026  |  Data: github.com/tim2000s/llm-food-benchmark-academic",
             fontsize=8, ha="center", va="bottom", color="#888888")

    for ext in ("png", "pdf"):
        p = OUTPUT_DIR / f"graphical_abstract.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    create_graphical_abstract()
    print("Done.")
