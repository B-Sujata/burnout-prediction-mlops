"""
Stage 9: Visualization — Real Dataset Edition
Generates 5 plots:
  1. Burnout Score Distribution + Risk pie
  2. Correlation Heatmap
  3. Feature Importance (top models)
  4. Model Performance Comparison
  5. Burnout Score vs Original Stress Level (validation)
"""

import os, sys, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger, log_separator
from src.utils.config_loader import load_config

logger = get_logger(__name__)
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def plot_burnout_distribution(df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["burnout_score"], bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(df["burnout_score"].mean(), color="red", linestyle="--",
                    label=f"Mean = {df['burnout_score'].mean():.1f}")
    axes[0].axvline(df["burnout_score"].median(), color="orange", linestyle="--",
                    label=f"Median = {df['burnout_score'].median():.1f}")
    axes[0].set_title("Burnout Score Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Burnout Score (0–100)")
    axes[0].set_ylabel("Number of Students")
    axes[0].legend()

    low  = (df["burnout_score"] <= 33).sum()
    mod  = ((df["burnout_score"] > 33) & (df["burnout_score"] <= 66)).sum()
    high = (df["burnout_score"] > 66).sum()
    axes[1].pie(
        [low, mod, high],
        labels=[f"Low\n({low})", f"Moderate\n({mod})", f"High\n({high})"],
        autopct="%1.1f%%",
        colors=["#55A868", "#DD8452", "#C44E52"],
        startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    axes[1].set_title("Risk Category Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Burnout distribution → {save_path}")


def plot_correlation_heatmap(df, save_path):
    key_cols = [
        "anxiety_level", "depression", "self_esteem", "mental_health_history",
        "headache", "blood_pressure", "sleep_quality", "breathing_problem",
        "study_load", "future_career_concerns", "academic_performance",
        "peer_pressure", "bullying", "social_support", "noise_level",
        "emotional_strain", "physical_stress", "academic_pressure",
        "social_stress", "recovery_index", "burnout_score",
    ]
    cols = [c for c in key_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(18, 15))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot_kws={"size": 7.5}, ax=ax)
    ax.set_title("Feature Correlation Heatmap — Real Student Data", fontsize=15, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Correlation heatmap → {save_path}")


def plot_feature_importance(model_dir, feature_names, save_path):
    candidates = ["randomforest", "gradientboosting", "xgboost"]
    found = []
    for name in candidates:
        p = os.path.join(model_dir, f"{name}_model.pkl")
        if os.path.exists(p):
            found.append((name, p))
        if len(found) == 2:
            break

    n = len(found)
    if n == 0:
        logger.warning("No tree-based model found for feature importance plot.")
        return

    fig, axes = plt.subplots(1, n, figsize=(10 * n, 8))
    if n == 1:
        axes = [axes]

    for ax, (name, path), color in zip(axes, found, PALETTE):
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        estimator = pipeline.named_steps["model"]
        if not hasattr(estimator, "feature_importances_"):
            continue
        importances = estimator.feature_importances_
        top_idx = np.argsort(importances)[::-1][:18]
        y_pos = range(len(top_idx))
        ax.barh(list(y_pos), importances[top_idx][::-1], color=color, edgecolor="white", alpha=0.85)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([feature_names[i] for i in top_idx[::-1]], fontsize=9)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Feature Importance — {name.title()}", fontsize=13, fontweight="bold")

    plt.suptitle("Top Feature Importances (Real Dataset)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance → {save_path}")


def plot_model_comparison(metrics_path, save_path):
    with open(metrics_path) as f:
        summary = json.load(f)
    best_model = summary.pop("best_model", None)
    names = list(summary.keys())
    mae_vals  = [summary[m]["mae"]  for m in names]
    rmse_vals = [summary[m]["rmse"] for m in names]
    r2_vals   = [summary[m]["r2"]   for m in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, vals, label, color in zip(
        axes, [mae_vals, rmse_vals, r2_vals],
        ["MAE (↓ better)", "RMSE (↓ better)", "R² Score (↑ better)"],
        PALETTE
    ):
        bars = ax.bar(names, vals, color=color, edgecolor="white", alpha=0.85)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.bar_label(bars, fmt="%.3f", fontsize=9)
        if best_model in names:
            ax.get_xticklabels()[names.index(best_model)].set_color("red")
            ax.get_xticklabels()[names.index(best_model)].set_fontweight("bold")

    fig.suptitle("Model Performance Comparison — Real Student Data", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Model comparison → {save_path}")


def plot_stress_level_validation(df, save_path):
    """
    Validate our engineered burnout_score against the original
    categorical stress_level labels (0=Low, 1=Moderate, 2=High).
    """
    if "stress_level" not in df.columns:
        logger.warning("stress_level column not found — skipping validation plot.")
        return

    label_map = {0: "Low (0)", 1: "Moderate (1)", 2: "High (2)"}
    df = df.copy()
    df["stress_label"] = df["stress_level"].map(label_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    order = ["Low (0)", "Moderate (1)", "High (2)"]
    colors = ["#55A868", "#DD8452", "#C44E52"]
    data_by_group = [df[df["stress_label"] == g]["burnout_score"].values for g in order]
    bp = axes[0].boxplot(data_by_group, labels=order, patch_artist=True, notch=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[0].set_title("Burnout Score vs Original Stress Level\n(Model Validation)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Original Stress Level (from dataset)")
    axes[0].set_ylabel("Engineered Burnout Score (0–100)")

    # Mean burnout per stress level
    means = df.groupby("stress_label")["burnout_score"].mean().reindex(order)
    axes[1].bar(order, means.values, color=colors, edgecolor="white", alpha=0.85)
    for i, v in enumerate(means.values):
        axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_title("Mean Burnout Score per Stress Level", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Original Stress Level")
    axes[1].set_ylabel("Mean Burnout Score")
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Validation plot → {save_path}")


def visualize_results(config_path: str = "configs/config.yaml") -> None:
    log_separator(logger, "Stage 9: Visualization")
    config = load_config(config_path)

    engineered_path = config.get("paths", "engineered_data")
    model_dir       = config.get("paths", "model_dir")
    results_dir     = config.get("paths", "results_dir")
    metrics_path    = os.path.join(results_dir, "metrics_summary.json")

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(engineered_path)
    logger.info(f"Loaded dataset: {df.shape}")

    with open(os.path.join(model_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    plot_burnout_distribution(df, os.path.join(results_dir, "burnout_distribution.png"))
    plot_correlation_heatmap(df,  os.path.join(results_dir, "correlation_heatmap.png"))
    plot_feature_importance(model_dir, feature_names, os.path.join(results_dir, "feature_importance.png"))
    if os.path.exists(metrics_path):
        plot_model_comparison(metrics_path, os.path.join(results_dir, "model_comparison.png"))
    plot_stress_level_validation(df, os.path.join(results_dir, "stress_level_validation.png"))

    logger.info(f"All 5 plots saved to: {results_dir}")

if __name__ == "__main__":
    visualize_results()
