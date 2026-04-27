"""
Full pipeline: 90 CAMELS gauges -> hydrofabric catchments -> static attributes.

Input:  camels_temperature_availability_hourly.csv (90-basin filter)
        hydrofabric v2.2 (hydrolocations.gpkg, conus_nextgen.gpkg)
Output: attributes_hydrofabric_90basins.csv
        gauge_to_divide.csv (the crosswalk, kept for provenance)
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path

# ==================== Paths ====================
HF_DIR = Path("/uufs/chpc.utah.edu/common/home/johnsonrc-group1/hydrofabric/v2.2")
HL_GPKG = HF_DIR / "hydrolocations.gpkg"
CONUS_GPKG = HF_DIR / "conus_nextgen.gpkg"

BASIN_LIST = Path(
    "/uufs/chpc.utah.edu/common/home/u1566670/hydroinformatics/"
    "stream_temperature_modeling/alaska/camels_temperature_availability_hourly.csv"
)

OUT_DIR = Path(
    "/uufs/chpc.utah.edu/common/home/u1566670/hydroinformatics/"
    "stream_temperature_modeling/conus"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 1. Load 90 filtered basins ====================
df_avail = pd.read_csv(BASIN_LIST, dtype={"site": str})
filtered = df_avail[
    (df_avail["has_temp"] == True) &
    (df_avail["n_years"] >= 4.0) &
    (df_avail["pct_missing"] < 10.0)
]
gauge_ids = filtered["site"].str.zfill(8).unique().tolist()
print(f"[1/4] Loaded {len(gauge_ids)} filtered gauges")

# ==================== 2. Gauge -> POI via hydrolocations ====================
print("[2/4] Reading hydrolocations ...")
hl = gpd.read_file(HL_GPKG, layer="hydrolocations", ignore_geometry=True)

gauges_hl = hl[
    (hl["hl_reference"] == "gages") &
    (hl["hl_link"].astype(str).str.zfill(8).isin(gauge_ids))
].copy()
gauges_hl["hl_link_padded"] = gauges_hl["hl_link"].astype(str).str.zfill(8)

print(f"      Matched {len(gauges_hl)} / {len(gauge_ids)} gauges to POIs")
unmatched_poi = sorted(set(gauge_ids) - set(gauges_hl["hl_link_padded"]))
if unmatched_poi:
    print(f"      UNMATCHED at POI step: {unmatched_poi}")

# ==================== 3. POI -> catchment via network ====================
print("[3/4] Reading network layer (~3.5M rows, takes a minute) ...")
network = gpd.read_file(CONUS_GPKG, layer="network", ignore_geometry=True)

# Match on hl_uri (string-to-string, avoids numeric NaN issues)
net_with_hl = network[network["hl_uri"].notna()]
merged = net_with_hl.merge(
    gauges_hl[["hl_link_padded", "hl_uri"]],
    on="hl_uri",
    how="inner"
)

# A POI can appear on multiple flowpath rows — keep one per gauge
gauge_to_divide = (
    merged[["hl_link_padded", "divide_id", "id", "hf_id",
            "vpuid", "tot_drainage_areasqkm"]]
    .drop_duplicates(subset=["hl_link_padded"], keep="first")
    .reset_index(drop=True)
)
print(f"      Unique gauge -> divide: {len(gauge_to_divide)}")

unmatched_div = sorted(set(gauge_ids) - set(gauge_to_divide["hl_link_padded"]))
if unmatched_div:
    print(f"      UNMATCHED at divide step: {unmatched_div}")

crosswalk_path = OUT_DIR / "gauge_to_divide.csv"
gauge_to_divide.to_csv(crosswalk_path, index=False)
print(f"      Saved crosswalk -> {crosswalk_path}")

# ==================== 4. Pull static attributes ====================
print("[4/4] Reading divide-attributes ...")
attrs = gpd.read_file(CONUS_GPKG, layer="divide-attributes", ignore_geometry=True)
print(f"      Total divides with attributes: {len(attrs)}")
print(f"      Attribute columns: {attrs.shape[1]}")

# Find the divide ID column in this layer (usually 'divide_id' or 'id')
id_col = "divide_id" if "divide_id" in attrs.columns else "id"
print(f"      Using ID column: {id_col}")

basin_attrs = attrs[attrs[id_col].isin(gauge_to_divide["divide_id"])].copy()
basin_attrs = basin_attrs.merge(
    gauge_to_divide[["hl_link_padded", "divide_id"]],
    left_on=id_col,
    right_on="divide_id",
    how="left"
)

# Put gauge_id first for readability
basin_attrs = basin_attrs.rename(columns={"hl_link_padded": "gauge_id"})
front = ["gauge_id", "divide_id"]
basin_attrs = basin_attrs[front + [c for c in basin_attrs.columns if c not in front]]

out_path = OUT_DIR / "attributes_hydrofabric_90basins.csv"
basin_attrs.to_csv(out_path, index=False)
print(f"      Saved -> {out_path}")
print(f"      Shape: {basin_attrs.shape} (should be ~90 rows × many attrs)")

# ==================== Summary ====================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Started with: {len(gauge_ids)} filtered gauges")
print(f"Matched in hydrolocations: {len(gauges_hl)}")
print(f"Matched to divides: {len(gauge_to_divide)}")
print(f"Got attributes for: {len(basin_attrs)}")
print(f"Number of attributes: {basin_attrs.shape[1] - 2}")  # minus gauge_id, divide_id
print(f"\nSample attributes: {basin_attrs.columns[:10].tolist()}")