#!/usr/bin/env python3
"""Run qlib single-factor backtests for Alpha101 factors (paper Section 3.1.3).

Saves one performance-vector JSON (P_i) per factor under
``configs/backtest.yaml: output_dir``.
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.backtest import backtest_factors
from alpha_r1.factors.alpha101 import parse_alpha_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", default="all",
                        help="factor subset: 'all', '001-101' or '001,005' (default: all)")
    parser.add_argument("--config", default="configs/backtest.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    names = parse_alpha_spec(args.alphas)
    print(f"Backtesting {len(names)} factors on {config['market']} "
          f"({config['start_date']}..{config['end_date']})")
    backtest_factors(names, config)


if __name__ == "__main__":
    main()
