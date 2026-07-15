from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
PARQUET_PATH = DATA_DIR / "yellow_tripdata_2026-01.parquet"
ZONE_SHP = DATA_DIR / "taxi_zones.shp"
SEED = 42
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
