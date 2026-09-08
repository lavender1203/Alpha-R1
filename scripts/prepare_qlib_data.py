#!/usr/bin/env python3
"""Convert OHLCV CSV files in data/stock_data/ into qlib binary format."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.backtest.data import dump_csv_to_qlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", default="data/stock_data",
                        help="directory of OHLCV CSV files (default: data/stock_data)")
    parser.add_argument("--qlib-dir", default="~/.qlib/qlib_data/alpha_r1",
                        help="output qlib data directory")
    args = parser.parse_args()
    dump_csv_to_qlib(args.csv_dir, args.qlib_dir)
    print(f"qlib data written to {args.qlib_dir}")


if __name__ == "__main__":
    main()
