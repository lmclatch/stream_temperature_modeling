"""
Plot NeuralHydrology training results for the stream temperature LSTM.

Produces a multi-panel figure summarizing:
  1. Training/validation loss curves over epochs
  2. Validation NSE/KGE/RMSE per epoch (median)
  3. Distribution of per-basin NSE at the best epoch
  4. Map of per-basin NSE (if basin coordinates are available)
  5. Time series example: best, median, and worst basin (predicted vs observed)

Usage:
    python plot_results.py --run-dir /path/to/run_dir [--epoch N] [--out-dir /path/]

If --epoch is not specified, uses the epoch with the highest median NSE.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-dir",
    type=Path,
    required=True,
    help="NH run directory (contains model_epochNNN.pt, validation/, output.log)",
)
parser.add_argument(
    "--epoch",
    type=int,
    default=None,
    help="Epoch to use for per-basin and time-series plots. Default: best validation NSE.",
)
parser.add_argument(
    "--out-dir",
    type=Path,
    default=None,
    help="Where to write figures. Default: <run-dir>/figures/",
)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path(
        "/uufs/chpc.utah.edu/common/home/johnsonrc-group1/hourly_streamtemp/nh_data/time_series"
    ),
    help="Directory of preprocessed per-basin NetCDFs (for time series plots).",
)
args = parser.parse_args()

run_dir = args.run_dir
out_dir = args.out_dir or (run_dir / "figures")
out_dir.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
# 1. Parse output.log for per-epoch metrics
# ---------------------------------------------------------------------------
log_path = run_dir / "output.log"
train_loss_per_epoch = {}
val_metrics_per_epoch = {}  # epoch -> dict of NSE, KGE, RMSE

with open(log_path) as f:
    for line in f:
        # Training loss: "Epoch N average loss: avg_loss: 0.01505, ..."
        if "average loss:" in line and "validation" not in line:
            parts = line.split("Epoch ")[1]
            ep = int(parts.split(" ")[0])
            loss = float(parts.split("avg_loss: ")[1].split(",")[0])
            train_loss_per_epoch[ep] = loss
        # Validation: "Epoch N average validation loss: ... -- Median validation metrics: avg_loss: ..., NSE: ..., KGE: ..., RMSE: ..."
        elif "validation loss" in line:
            parts = line.split("Epoch ")[1]
            ep = int(parts.split(" ")[0])
            tail = line.split("Median validation metrics:")[1]
            metrics = {}
            for kv in tail.split(","):
                k, v = kv.strip().split(":")
                try:
                    metrics[k.strip()] = float(v.strip())
                except ValueError:
                    pass
            val_metrics_per_epoch[ep] = metrics

epochs = sorted(train_loss_per_epoch.keys())
print(f"Found {len(epochs)} epochs in log: {epochs}")

# Pick best epoch by median NSE
best_epoch = (
    args.epoch
    if args.epoch is not None
    else max(val_metrics_per_epoch, key=lambda e: val_metrics_per_epoch[e].get("NSE", -np.inf))
)
print(f"Using epoch {best_epoch} for per-basin plots (best by median NSE)")

# ---------------------------------------------------------------------------
# 2. Load per-basin metrics for the chosen epoch
# ---------------------------------------------------------------------------
metrics_csv = run_dir / "validation" / f"model_epoch{best_epoch:03d}" / "validation_metrics.csv"
df = pd.read_csv(metrics_csv)
df["basin"] = df["basin"].astype(str).str.zfill(8)  # USGS gauge IDs are 8-digit strings
print(f"Loaded {len(df)} basins from {metrics_csv}")
print(f"  NaN basins: {df['NSE'].isna().sum()}")

# ---------------------------------------------------------------------------
# 3. Load validation_results.p for time-series example plots
# ---------------------------------------------------------------------------
results_pkl = run_dir / "validation" / f"model_epoch{best_epoch:03d}" / "validation_results.p"
with open(results_pkl, "rb") as f:
    results = pickle.load(f)
# results structure: {basin_id: {'1H': xr.Dataset with y_obs and y_sim}}
# (the freq key may differ — adjust below if needed)

# ---------------------------------------------------------------------------
# Figure 1: Training curves + median validation metrics over epochs
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.plot(epochs, [train_loss_per_epoch[e] for e in epochs], "o-", label="Train loss", color="C0")
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (MSE in normalized units)", color="C0")
ax.tick_params(axis="y", labelcolor="C0")
ax.set_title("Training loss over epochs")
ax.grid(alpha=0.3)

ax = axes[1]
nse = [val_metrics_per_epoch[e]["NSE"] for e in epochs]
kge = [val_metrics_per_epoch[e]["KGE"] for e in epochs]
rmse = [val_metrics_per_epoch[e]["RMSE"] for e in epochs]
ax.plot(epochs, nse, "o-", label="Median NSE", color="C2")
ax.plot(epochs, kge, "s-", label="Median KGE", color="C1")
ax.set_xlabel("Epoch")
ax.set_ylabel("Median validation metric (across basins)")
ax2 = ax.twinx()
ax2.plot(epochs, rmse, "^--", label="Median RMSE (°C)", color="C3", alpha=0.7)
ax2.set_ylabel("Median RMSE (°C)", color="C3")
ax2.tick_params(axis="y", labelcolor="C3")
ax.legend(loc="lower right")
ax.set_title("Validation metrics (median across basins)")
ax.grid(alpha=0.3)
ax.axvline(best_epoch, color="k", linestyle=":", alpha=0.5, label=f"Best epoch ({best_epoch})")

fig.tight_layout()
fig.savefig(out_dir / "01_training_curves.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_dir / '01_training_curves.png'}")

# ---------------------------------------------------------------------------
# Figure 2: Per-basin metric distributions at best epoch
# ---------------------------------------------------------------------------
clean = df.dropna(subset=["NSE", "KGE"])
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# NSE histogram, clipped at 0 for visibility (negative outliers crush the distribution)
ax = axes[0]
nse_clipped = clean["NSE"].clip(lower=0)
ax.hist(nse_clipped, bins=40, color="C2", alpha=0.7, edgecolor="black")
ax.axvline(clean["NSE"].median(), color="k", linestyle="--", label=f"Median = {clean['NSE'].median():.3f}")
ax.set_xlabel("NSE (clipped at 0)")
ax.set_ylabel("Count of basins")
ax.set_title(f"NSE distribution (n={len(clean)} non-NaN basins)")
ax.legend()
ax.grid(alpha=0.3)

# KGE histogram
ax = axes[1]
ax.hist(clean["KGE"], bins=40, color="C1", alpha=0.7, edgecolor="black")
ax.axvline(clean["KGE"].median(), color="k", linestyle="--", label=f"Median = {clean['KGE'].median():.3f}")
ax.set_xlabel("KGE")
ax.set_ylabel("Count of basins")
ax.set_title("KGE distribution")
ax.legend()
ax.grid(alpha=0.3)

# RMSE histogram
ax = axes[2]
ax.hist(clean["RMSE"], bins=40, color="C3", alpha=0.7, edgecolor="black")
ax.axvline(clean["RMSE"].median(), color="k", linestyle="--", label=f"Median = {clean['RMSE'].median():.3f}°C")
ax.set_xlabel("RMSE (°C)")
ax.set_ylabel("Count of basins")
ax.set_title("RMSE distribution")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle(f"Per-basin validation metrics @ epoch {best_epoch}", y=1.02)
fig.tight_layout()
fig.savefig(out_dir / "02_metric_distributions.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_dir / '02_metric_distributions.png'}")

# ---------------------------------------------------------------------------
# Figure 3: Time series for best, median, and worst basins
# ---------------------------------------------------------------------------
# Pick three exemplar basins from non-NaN ones
sorted_clean = clean.sort_values("NSE")
worst = sorted_clean.iloc[0]
median_basin = sorted_clean.iloc[len(sorted_clean) // 2]
best = sorted_clean.iloc[-1]
exemplars = [("Best NSE", best), ("Median NSE", median_basin), ("Worst NSE", worst)]

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)

for ax, (label, row) in zip(axes, exemplars):
    bid = row["basin"]
    if bid not in results:
        # Try int key in case results uses int basin IDs
        try:
            res_key = int(bid)
        except ValueError:
            res_key = bid
        if res_key not in results:
            ax.text(0.5, 0.5, f"Basin {bid}: no results found", transform=ax.transAxes, ha="center")
            continue
    else:
        res_key = bid

    # Find the frequency key (likely '1H' or 'h')
    freq_keys = list(results[res_key].keys())
    if not freq_keys:
        ax.text(0.5, 0.5, f"Basin {bid}: no freq key", transform=ax.transAxes, ha="center")
        continue
    freq = freq_keys[0]
    ds = results[res_key][freq]["xr"]

    # Get observed and simulated series
    target_var = [v for v in ds.data_vars if "obs" in v][0]
    sim_var = target_var.replace("_obs", "_sim")

    obs = ds[target_var].values.squeeze()
    sim = ds[sim_var].values.squeeze()
    time = ds["date"].values if "date" in ds.coords else ds[list(ds.coords)[0]].values

    ax.plot(time, obs, label="Observed", color="black", linewidth=0.8, alpha=0.8)
    ax.plot(time, sim, label="Predicted", color="C0", linewidth=0.8, alpha=0.8)
    ax.set_ylabel("Stream T (°C)")
    ax.set_title(
        f"{label}: basin {bid} | NSE={row['NSE']:.3f}, KGE={row['KGE']:.3f}, RMSE={row['RMSE']:.2f}°C"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Date")
fig.tight_layout()
fig.savefig(out_dir / "03_timeseries_examples.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_dir / '03_timeseries_examples.png'}")

# ---------------------------------------------------------------------------
# Figure 4: Scatter — predicted vs observed for the median basin
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (label, row) in zip(axes, exemplars):
    bid = row["basin"]
    res_key = bid if bid in results else (int(bid) if bid.isdigit() and int(bid) in results else None)
    if res_key is None:
        ax.text(0.5, 0.5, f"Basin {bid}: no data", transform=ax.transAxes, ha="center")
        continue

    freq = list(results[res_key].keys())[0]
    ds = results[res_key][freq]["xr"]
    target_var = [v for v in ds.data_vars if "obs" in v][0]
    sim_var = target_var.replace("_obs", "_sim")

    obs = ds[target_var].values.squeeze()
    sim = ds[sim_var].values.squeeze()
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_m, sim_m = obs[mask], sim[mask]

    ax.scatter(obs_m, sim_m, s=1, alpha=0.3, color="C0")
    lo = min(obs_m.min(), sim_m.min())
    hi = max(obs_m.max(), sim_m.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="1:1")
    ax.set_xlabel("Observed T (°C)")
    ax.set_ylabel("Predicted T (°C)")
    ax.set_title(f"{label}: basin {bid}\nNSE={row['NSE']:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

fig.tight_layout()
fig.savefig(out_dir / "04_obs_vs_pred_scatter.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_dir / '04_obs_vs_pred_scatter.png'}")

# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print(f"Summary @ epoch {best_epoch}")
print("=" * 60)
print(f"  Median NSE:  {clean['NSE'].median():.4f}")
print(f"  Median KGE:  {clean['KGE'].median():.4f}")
print(f"  Median RMSE: {clean['RMSE'].median():.3f} °C")
print(f"  N basins (non-NaN): {len(clean)} / {len(df)}")
print(f"  NaN basins: {df[df['NSE'].isna()]['basin'].tolist()}")
print()
print(f"All figures written to: {out_dir}")
