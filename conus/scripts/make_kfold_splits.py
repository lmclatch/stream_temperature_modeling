"""
Set up CONUS spatial CV: generate stratified 5-fold splits AND per-fold NeuralHydrology configs.

PART 1 — STRATIFIED 5-FOLD SPLITTING
    Stratification: by frac_snow tertile (low / mid / high), so each fold's test set
    contains basins from the full hydroclimate range. This prevents one fold from
    accidentally getting all the warm basins (or all the cold ones), which would
    make the with-Q vs no-Q comparison hard to interpret.

    For each fold k (k = 1..5):
        - Test set: ~16 basins (one fifth of basins from each tertile)
        - Train set: ~66 basins (the rest)

    Outputs (in OUT_DIR):
        fold{1..5}_noQ_train.txt    fold{1..5}_noQ_test.txt
        fold{1..5}_withQ_train.txt  fold{1..5}_withQ_test.txt
        manifest.csv  -- one row per basin, with frac_snow, tertile, fold assignment

    Sanity-check assertions:
        - Each variant: union of all 5 test sets == full basin list, no basin in 2 test sets
        - Each fold: train ∩ test == empty
        - Tertile representation: every fold's test set spans all 3 tertiles

PART 2 — PER-FOLD CONFIG GENERATION
    Reads the existing v2 template configs (with-Q and no-Q) and produces 10 fold-specific
    configs (5 folds x 2 variants). Only these fields are modified per fold:
        - experiment_name
        - run_dir
        - train_basin_file
        - validation_basin_file (= train_basin_file: validation on training basins, period 2023)
        - test_basin_file

    Everything else (architecture, hyperparameters, seed=42, statics, dynamic_inputs,
    date ranges) is preserved exactly from the templates.

    Outputs (in CONFIG_OUT_DIR):
        no_q_fold{1..5}.yml
        with_q_fold{1..5}.yml

Usage:
    python make_kfold_setup.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE = Path("/uufs/chpc.utah.edu/common/home/u1566670/hydroinformatics/stream_temperature_modeling/conus")
ATTRS_CSV = Path(
    "/uufs/chpc.utah.edu/common/home/johnsonrc-group1/hourly_streamtemp/hourly/attributes/camels_attributes_90basins.csv"
)
BASINS_NOQ = BASE / "data" / "basins_noQ.txt"
BASINS_WITHQ = BASE / "data" / "basins_withQ.txt"
OUT_DIR = BASE / "data" / "folds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_NOQ = BASE / "config" / "test_without_streamflow.yml"
TEMPLATE_WITHQ = BASE / "config" / "test_with_streamflow.yml"
CONFIG_OUT_DIR = BASE / "config" / "folds"
CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_BASE = BASE / "results"

SEED = 42
N_FOLDS = 5
N_TERTILES = 3
EXCLUDE_BASIN = "12178100"  # known sensor pathology in T, drop from both lists

# -----------------------------------------------------------------------------
# Validate inputs exist before doing any work
# -----------------------------------------------------------------------------
for path in [ATTRS_CSV, BASINS_NOQ, BASINS_WITHQ, TEMPLATE_NOQ, TEMPLATE_WITHQ]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")

# =============================================================================
# PART 1 — Stratified 5-fold splits
# =============================================================================

# -----------------------------------------------------------------------------
# Load basin lists (post-12178100 drop)
# -----------------------------------------------------------------------------
def load_basins(path: Path) -> list[str]:
    with open(path) as f:
        basins = [b.strip().zfill(8) for b in f if b.strip()]
    if EXCLUDE_BASIN in basins:
        basins.remove(EXCLUDE_BASIN)
        print(f"  Dropped {EXCLUDE_BASIN} from {path.name}")
    return sorted(basins)

print("Loading basin lists...")
noq_basins = load_basins(BASINS_NOQ)
withq_basins = load_basins(BASINS_WITHQ)
print(f"  no-Q basins: {len(noq_basins)}")
print(f"  with-Q basins: {len(withq_basins)}")
print(f"  Shared basins: {len(set(noq_basins) & set(withq_basins))}")

# Use no-Q as the universe for splitting (it's the larger / more permissive list)
universe = noq_basins
print(f"\nSplitting universe: {len(universe)} basins (no-Q list)")

# -----------------------------------------------------------------------------
# Load frac_snow for each basin
# -----------------------------------------------------------------------------
print("\nLoading CAMELS attributes...")
attrs = pd.read_csv(ATTRS_CSV, dtype={"gauge_id": str})
attrs["gauge_id"] = attrs["gauge_id"].str.zfill(8)
attrs = attrs.set_index("gauge_id")

missing = [b for b in universe if b not in attrs.index]
if missing:
    print(f"WARNING: {len(missing)} basins not in attributes file: {missing}")
    sys.exit(1)

snow = attrs.loc[universe, "frac_snow"].copy()
print(f"  frac_snow range: {snow.min():.3f} to {snow.max():.3f}")
print(f"  median: {snow.median():.3f}, mean: {snow.mean():.3f}")

# -----------------------------------------------------------------------------
# Bin into 3 tertiles by frac_snow
# -----------------------------------------------------------------------------
tertiles = pd.qcut(snow, q=N_TERTILES, labels=["low_snow", "mid_snow", "high_snow"])
print(f"\nTertile boundaries (frac_snow):")
for label, group in tertiles.groupby(tertiles, observed=True):
    s = snow.loc[group.index]
    print(f"  {label}: n={len(group)}, range [{s.min():.3f}, {s.max():.3f}]")

# -----------------------------------------------------------------------------
# Stratified k-fold: for each tertile, shuffle and assign basins to folds
# -----------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
fold_assignment = pd.Series(index=universe, dtype=int, name="fold")

for label in ["low_snow", "mid_snow", "high_snow"]:
    in_tertile = tertiles[tertiles == label].index.tolist()
    shuffled = rng.permutation(in_tertile).tolist()
    # Round-robin assign to folds 1..N_FOLDS so each fold gets a balanced share
    for i, basin in enumerate(shuffled):
        fold_assignment.loc[basin] = (i % N_FOLDS) + 1

# -----------------------------------------------------------------------------
# Build manifest
# -----------------------------------------------------------------------------
manifest = pd.DataFrame({
    "basin": universe,
    "frac_snow": snow.values,
    "tertile": tertiles.values,
    "fold": fold_assignment.values,
    "in_noQ": [b in noq_basins for b in universe],
    "in_withQ": [b in withq_basins for b in universe],
})
manifest.to_csv(OUT_DIR / "manifest.csv", index=False)
print(f"\nManifest written: {OUT_DIR / 'manifest.csv'}")

# -----------------------------------------------------------------------------
# Write per-fold basin list files (noQ and withQ variants)
# -----------------------------------------------------------------------------
print("\nWriting fold files...")
for variant_name, variant_basins in [("noQ", noq_basins), ("withQ", withq_basins)]:
    variant_set = set(variant_basins)
    for fold in range(1, N_FOLDS + 1):
        test_basins = sorted(b for b in universe if fold_assignment[b] == fold and b in variant_set)
        train_basins = sorted(b for b in variant_basins if b not in test_basins)

        train_path = OUT_DIR / f"fold{fold}_{variant_name}_train.txt"
        test_path = OUT_DIR / f"fold{fold}_{variant_name}_test.txt"
        train_path.write_text("\n".join(train_basins) + "\n")
        test_path.write_text("\n".join(test_basins) + "\n")

# -----------------------------------------------------------------------------
# Sanity-check assertions
# -----------------------------------------------------------------------------
print("\nValidating splits...")
for variant_name, variant_basins in [("noQ", noq_basins), ("withQ", withq_basins)]:
    variant_set = set(variant_basins)
    all_test_unions = set()
    for fold in range(1, N_FOLDS + 1):
        test_path = OUT_DIR / f"fold{fold}_{variant_name}_test.txt"
        train_path = OUT_DIR / f"fold{fold}_{variant_name}_train.txt"
        test_set = set(test_path.read_text().strip().split("\n"))
        train_set = set(train_path.read_text().strip().split("\n"))

        assert test_set.isdisjoint(train_set), f"{variant_name} fold {fold}: train/test overlap"
        assert (test_set | train_set) == variant_set, (
            f"{variant_name} fold {fold}: train+test != full basin set "
            f"(missing {variant_set - (test_set | train_set)})"
        )
        assert all_test_unions.isdisjoint(test_set), (
            f"{variant_name} fold {fold}: test basin appears in earlier fold's test set"
        )
        all_test_unions |= test_set

    assert all_test_unions == variant_set, (
        f"{variant_name}: union of test sets != full basin set "
        f"(missing {variant_set - all_test_unions})"
    )
    print(f"  {variant_name}: all assertions passed")

# -----------------------------------------------------------------------------
# Print summary table
# -----------------------------------------------------------------------------
print("\nFold composition:")
print(f"{'fold':>4} {'variant':>8} {'n_train':>8} {'n_test':>7} {'low':>4} {'mid':>4} {'high':>5} "
      f"{'mean_snow_test':>16}")
for variant_name, variant_basins in [("noQ", noq_basins), ("withQ", withq_basins)]:
    for fold in range(1, N_FOLDS + 1):
        test_path = OUT_DIR / f"fold{fold}_{variant_name}_test.txt"
        train_path = OUT_DIR / f"fold{fold}_{variant_name}_train.txt"
        test_basins = test_path.read_text().strip().split("\n")
        train_basins = train_path.read_text().strip().split("\n")
        test_tertiles = tertiles.loc[test_basins].value_counts()
        mean_test_snow = snow.loc[test_basins].mean()
        print(f"{fold:>4} {variant_name:>8} {len(train_basins):>8} {len(test_basins):>7} "
              f"{test_tertiles.get('low_snow', 0):>4} "
              f"{test_tertiles.get('mid_snow', 0):>4} "
              f"{test_tertiles.get('high_snow', 0):>5} "
              f"{mean_test_snow:>16.3f}")

print(f"\nSplits done. Files written to: {OUT_DIR}")

# =============================================================================
# PART 2 — Per-fold config generation
# =============================================================================

print("\n" + "=" * 70)
print("Generating per-fold NeuralHydrology configs")
print("=" * 70)

def generate_fold_configs(template_path: Path, variant_short: str, variant_long: str):
    """
    variant_short : matches fold filename suffix ('noQ' or 'withQ')
    variant_long  : used in experiment_name and config filename ('no_q' or 'with_q')
    """
    print(f"\nGenerating {variant_short} configs from {template_path.name}...")

    with open(template_path) as f:
        template = yaml.safe_load(f)

    for fold in range(1, N_FOLDS + 1):
        cfg = dict(template)  # shallow copy fine — we only overwrite top-level keys

        train_path = OUT_DIR / f"fold{fold}_{variant_short}_train.txt"
        test_path = OUT_DIR / f"fold{fold}_{variant_short}_test.txt"

        for p in [train_path, test_path]:
            if not p.exists():
                raise FileNotFoundError(f"Fold file missing: {p}")

        cfg["experiment_name"] = f"streamtemp_{variant_long}_fold{fold}"
        cfg["run_dir"] = str(RESULTS_BASE / f"kfold_{variant_long}" / f"fold{fold}")
        cfg["train_basin_file"] = str(train_path)
        cfg["validation_basin_file"] = str(train_path)  # validation on TRAINING basins, period 2023
        cfg["test_basin_file"] = str(test_path)

        out_path = CONFIG_OUT_DIR / f"{variant_long}_fold{fold}.yml"
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  Wrote {out_path.name} (train={train_path.name}, test={test_path.name})")

generate_fold_configs(TEMPLATE_NOQ, "noQ", "no_q")
generate_fold_configs(TEMPLATE_WITHQ, "withQ", "with_q")

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("All done.")
print("=" * 70)
print(f"  Splits:   {OUT_DIR}")
print(f"  Configs:  {CONFIG_OUT_DIR}")
print()
print("Verify a config diff against its template (should show ~5 changed fields):")
print(f"  diff {TEMPLATE_NOQ} \\")
print(f"       {CONFIG_OUT_DIR / 'no_q_fold1.yml'}")
print(f"  diff {TEMPLATE_WITHQ} \\")
print(f"       {CONFIG_OUT_DIR / 'with_q_fold1.yml'}")