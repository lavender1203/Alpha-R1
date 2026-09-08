"""qlib data preparation and initialization helpers."""

import shutil
from pathlib import Path

QLIB_FIELDS = ["open", "high", "low", "close", "volume", "vwap"]


def dump_csv_to_qlib(csv_dir: str | Path, qlib_dir: str | Path,
                     include_fields: list[str] | None = None) -> None:
    """Convert a directory of OHLCV CSV files into qlib binary format.

    Expects one CSV per instrument with columns
    ``date,open,high,low,close,volume[,vwap]``. A missing ``vwap`` column is
    approximated by the typical price ``(high + low + close) / 3``.
    """
    try:
        from qlib.scripts.dump_bin import DumpDataAll
    except ImportError as e:
        raise ImportError(
            "qlib.scripts.dump_bin is only available when qlib is installed "
            "from source (git clone https://github.com/microsoft/qlib); the "
            "pyqlib wheel does not ship it. Either install qlib from source "
            "or download an official pre-built data bundle instead."
        ) from e

    import pandas as pd

    csv_dir = Path(csv_dir).expanduser()
    qlib_dir = Path(qlib_dir).expanduser()
    fields = include_fields or QLIB_FIELDS

    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no CSV files found in {csv_dir}")

    staging = qlib_dir / "_staging_csv"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "vwap" not in df.columns:
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
        df = df[["date"] + [f for f in fields if f in df.columns]]
        df.to_csv(staging / csv_path.name, index=False)

    try:
        DumpDataAll(
            csv_path=str(staging),
            qlib_dir=str(qlib_dir),
            include_fields=",".join(fields),
            date_field_name="date",
        ).dump()
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def init_qlib(qlib_dir: str | Path, kernels: int = 1, **kwargs) -> None:
    """Initialize a qlib runtime against a prepared binary data directory.

    ``kernels`` defaults to 1 (single-process evaluation) because the custom
    cross-sectional operators fetch full panels through ``D.features``, which
    is not available in spawned worker processes.
    """
    try:
        import qlib
    except ImportError as e:
        raise ImportError(
            "pyqlib is required for backtesting: pip install pyqlib"
        ) from e

    provider_uri = str(Path(qlib_dir).expanduser())
    if not Path(provider_uri).is_dir():
        raise FileNotFoundError(
            f"qlib data not found at {provider_uri}; run scripts/prepare_qlib_data.py first"
        )
    qlib.init(provider_uri=provider_uri, kernels=kernels, **kwargs)
