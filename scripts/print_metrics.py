"""
Simple script to print metrics summary.
Called from CI/CD instead of inline Python to avoid YAML escape issues.
"""
import json
import sys

try:
    with open("results/metrics_summary.json") as f:
        summary = json.load(f)

    best = summary.pop("best_model", "unknown")

    print("\n" + "=" * 50)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 50)

    for name, metrics in summary.items():
        r2  = metrics.get("r2",   "N/A")
        mae = metrics.get("mae",  "N/A")
        rmse= metrics.get("rmse", "N/A")
        print(f"  {name:<22s}  R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    print(f"\n  Best model: {best}")
    print("=" * 50)

except FileNotFoundError:
    print("metrics_summary.json not found - pipeline may not have completed")
    sys.exit(1)
