"""
Downloads AORC hourly forcings for all 671 CAMELS basins from HydroShare
and organizes them into the folder structure expected by NeuralHydrology.

Source:
    Frame, J. et al. (2025). AORC atmospheric forcing data across CAMELS US
    basins, 1980-2024. HydroShare.
    https://doi.org/10.4211/hs.c738c05278a34bc9848dd14d61cffab9

NeuralHydrology folder structure after this script:
    CAMELS_US/
        hourly/
            aorc_hourly/
                01013500_hourly_aorc.csv
                01022500_hourly_aorc.csv
                ... (671 basins)

Requirements:
    pip install requests tqdm
"""

import time
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# USER PARAMETERS
# =============================================================================

GROUP_DATA_DIR = Path('/uufs/chpc.utah.edu/common/home/johnsonrc-group1/CAMELS_US')
HS_RESOURCE_ID = 'c738c05278a34bc9848dd14d61cffab9'
MAX_RETRIES    = 3
RETRY_DELAY    = 5

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

aorc_dir = GROUP_DATA_DIR / 'hourly' / 'aorc_hourly'
tar_dir  = GROUP_DATA_DIR / 'hourly' / 'aorc_hourly' / 'tar_downloads'
aorc_dir.mkdir(parents=True, exist_ok=True)
tar_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  CAMELS AORC Hourly Forcing Download")
print("=" * 60)
print(f"  Basin CSVs  : {aorc_dir}")
print(f"  Tar cache   : {tar_dir}")
print(f"  Resource    : https://www.hydroshare.org/resource/{HS_RESOURCE_ID}/")
print()

# =============================================================================
# STEP 1: GET FILE LIST FROM HYDROSHARE API
# =============================================================================

def get_hydroshare_file_list(resource_id):
    url = f"https://www.hydroshare.org/hsapi/resource/{resource_id}/files/"
    files = []
    while url:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for f in data.get('results', []):
            files.append((f['file_name'], f['url']))
        url = data.get('next')
    return files


print("Querying HydroShare for file list ...")
try:
    all_files = get_hydroshare_file_list(HS_RESOURCE_ID)
except requests.exceptions.RequestException as e:
    print(f"ERROR: Could not reach HydroShare API: {e}")
    raise

tar_files = [(fname, url) for fname, url in all_files if fname.endswith('.tar.gz')]
print(f"  Found {len(tar_files)} water year archives (1980-2024)")
print()

# =============================================================================
# STEP 2: DOWNLOAD TAR.GZ FILES
# =============================================================================

def download_file(url, dest_path):
    """Download with retry logic. Skip if already exists."""
    if dest_path.exists():
        return False
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            tmp_path = dest_path.with_suffix('.tmp')
            total = int(resp.headers.get('content-length', 0))
            with open(tmp_path, 'wb') as f, tqdm(
                total=total, unit='B', unit_scale=True,
                desc=dest_path.name, leave=False
            ) as bar:
                for chunk in resp.iter_content(chunk_size=512 * 1024):
                    f.write(chunk)
                    bar.update(len(chunk))
            tmp_path.rename(dest_path)
            return True
        except (requests.exceptions.RequestException, OSError) as e:
            if attempt < MAX_RETRIES:
                print(f"\n  Retry {attempt}/{MAX_RETRIES} for {dest_path.name}: {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {dest_path.name}: {e}")


already_done = sum(1 for fname, _ in tar_files if (tar_dir / fname).exists())
print(f"  Already downloaded : {already_done} / {len(tar_files)} archives")
print(f"  Remaining          : {len(tar_files) - already_done} archives")
print()

failed_downloads = []
for fname, furl in tqdm(tar_files, desc="Downloading archives", unit='year'):
    try:
        download_file(furl, tar_dir / fname)
    except RuntimeError as e:
        failed_downloads.append((fname, str(e)))

if failed_downloads:
    print(f"\nWARNING: {len(failed_downloads)} archives failed to download:")
    for fname, err in failed_downloads:
        print(f"  {fname}: {err}")

# =============================================================================
# STEP 3: EXTRACT TAR.GZ FILES INTO aorc_hourly/
# Each archive contains one CSV per basin. CSVs from all years are merged
# into per-basin files by appending year-by-year.
# Extracted CSVs land directly in aorc_dir — NeuralHydrology reads them there.
# =============================================================================

print("\nExtracting archives ...")

for fname, _ in tqdm(tar_files, desc="Extracting archives", unit='year'):
    tar_path = tar_dir / fname
    if not tar_path.exists():
        print(f"  Skipping {fname} — not downloaded.")
        continue

    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        for member in members:
            # Extract only CSV files, stripping any subdirectory structure
            if member.name.endswith('.csv'):
                member.name = Path(member.name).name  # flatten to filename only
                dest = aorc_dir / member.name

                if dest.exists():
                    # Append this year's data (skip header line)
                    f = tar.extractfile(member)
                    if f:
                        lines = f.read().decode('utf-8').splitlines()
                        with open(dest, 'a') as out:
                            out.write('\n'.join(lines[1:]) + '\n')  # skip header
                else:
                    # First year — extract with header
                    tar.extract(member, path=aorc_dir)

print("\nExtraction complete.")

# =============================================================================
# STEP 4: CLEANUP — remove tar files to save space
# Comment this out if you want to keep the raw archives.
# =============================================================================

print("Cleaning up tar files ...")
for fname, _ in tar_files:
    tar_path = tar_dir / fname
    if tar_path.exists():
        tar_path.unlink()
tar_dir.rmdir()
print("Done.")

# =============================================================================
# SUMMARY
# =============================================================================

csv_count = len(list(aorc_dir.glob('*.csv')))
print(f"""
{'='*60}
  ALL DONE
{'='*60}
  Basin CSV files  : {csv_count}
  Location         : {aorc_dir}"""
)