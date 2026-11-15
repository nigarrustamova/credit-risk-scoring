"""Loading the Home Credit Default Risk tables.

The Kaggle zip holds 7 CSV tables (~2.7 GB). They are read once, numeric columns are downcast to float32/int32
and each table is cached as parquet in data/parquet/, which is ~5x smaller and ~20x faster to load.

    python -m crs.data            # unpack data/raw/home-credit-default-risk.zip and build the parquet cache
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PARQUET = ROOT / "data" / "parquet"

TABLES = ["application_train", "application_test", "bureau", "bureau_balance", "previous_application",
          "POS_CASH_balance", "installments_payments", "credit_card_balance"]


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        s = df[c]
        if s.dtype == np.float64:
            df[c] = s.astype(np.float32)
        elif s.dtype == np.int64:
            df[c] = pd.to_numeric(s, downcast="integer")
        elif (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) < 0.5 * len(s):
            df[c] = s.astype("category")
    return df


def unpack(zip_path: Path | None = None) -> None:
    """Extract the Kaggle zip (or any zip with the CSVs) into data/raw/."""
    zip_path = zip_path or next(iter(sorted(RAW.glob("*.zip")) or sorted(ROOT.glob("*.zip"))), None)
    if zip_path is None:
        return
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".csv") and not (RAW / Path(name).name).exists():
                print("extracting", name, flush=True)
                z.extract(name, RAW)


def build_cache() -> None:
    PARQUET.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        out = PARQUET / f"{t}.parquet"
        if out.exists():
            continue
        csv = RAW / f"{t}.csv"
        if not csv.exists() and (RAW / f"{t}.csv.gz").exists():
            csv = RAW / f"{t}.csv.gz"                    # gzipped CSVs work too
        if not csv.exists():
            print("missing", csv, file=sys.stderr)
            continue
        print("caching", t, flush=True)
        _downcast(pd.read_csv(csv)).to_parquet(out, index=False)


def load(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = PARQUET / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path, columns=columns)
    return _downcast(pd.read_csv(RAW / f"{name}.csv", usecols=columns))


def load_all() -> dict[str, pd.DataFrame]:
    return {t: load(t) for t in TABLES}


if __name__ == "__main__":
    unpack()
    build_cache()
    for t in TABLES:
        p = PARQUET / f"{t}.parquet"
        if p.exists():
            import pyarrow.parquet as pq
            print(f"{t:24s} {pq.ParquetFile(p).metadata.num_rows:>11,} rows")
