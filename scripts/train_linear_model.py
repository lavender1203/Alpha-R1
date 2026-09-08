#!/usr/bin/env python3
"""Estimate the fixed linear return model (paper Appendix F).

OLS of forward H-day returns on cross-sectionally standardized Alpha101
factor values over the historical window, written as betas.csv (consumed by
scripts/run_strategy_backtest.py and training/reward.py).
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.backtest.data import init_qlib
from alpha_r1.backtest.alpha101_qlib import (
    QLIB_EXPRESSIONS, custom_ops_config, set_cs_universe, clear_panel_cache,
)
from alpha_r1.backtest.linear_model import estimate_betas, save_betas
from alpha_r1.factors.alpha101 import parse_alpha_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--alphas", default="all",
                        help="factor subset: 'all', '001-101' or '001,005' (default: all)")
    parser.add_argument("--start-date", default=None,
                        help="estimation window start (default: config train_start_date)")
    parser.add_argument("--end-date", default=None,
                        help="estimation window end (default: config train_end_date)")
    parser.add_argument("--output", default="result/linear_model/betas.csv")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    start = args.start_date or config["train_start_date"]
    end = args.end_date or config["train_end_date"]
    holding = config.get("holding_days", 5)

    names = parse_alpha_spec(args.alphas)
    set_cs_universe(config["market"])
    init_qlib(config["qlib_data_dir"], custom_ops=custom_ops_config())
    clear_panel_cache()

    from qlib.data import D

    instruments = D.instruments(market=config["market"])
    expressions = [QLIB_EXPRESSIONS[n] for n in names]
    fwd_expr = f"Ref($close, -{holding}) / $close - 1"
    raw = D.features(instruments, expressions + [fwd_expr],
                     start_time=start, end_time=end)

    factor_panels = {n: raw[QLIB_EXPRESSIONS[n]].unstack(level="instrument")
                     for n in names}
    fwd = raw[fwd_expr].unstack(level="instrument")

    betas, intercept = estimate_betas(factor_panels, fwd)
    save_betas(betas, intercept, args.output)
    print(f"[linear-model] {len(names)} betas (intercept={intercept:.6f}) -> {args.output}")
    top = betas.abs().nlargest(5).index
    for n in top:
        print(f"  {n}: {betas[n]:+.6f}")


if __name__ == "__main__":
    main()
