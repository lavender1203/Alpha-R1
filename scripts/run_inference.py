#!/usr/bin/env python3
"""Run Alpha-R1 factor-selection inference (paper Section 3.3).

Input: a directory of factor description txt files (alphaNNN.txt), concatenated
into the decision-context prompt. Output: result_YYYYMMDD.json per decision day.
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.factors import parse_alpha_spec
from alpha_r1.inference import generate_decision_days, make_backend, run_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--factor-des-dir", default="result/alpha_des",
                        help="directory of alphaNNN.txt descriptions "
                             "(default: result/alpha_des; try examples/factor_descriptions)")
    parser.add_argument("--alphas", default="all",
                        help="factor subset: 'all', '001-040', or '001,003,007'")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--dates", default=None,
                        help="comma-separated explicit dates; overrides --start-date/--end-date")
    parser.add_argument("--market-state-dir", default=None,
                        help="optional directory of daily market-state txt files")
    parser.add_argument("--output-dir", default=None,
                        help="override configs/inference.yaml output_dir")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.dates:
        dates = args.dates.split(",")
    elif args.start_date and args.end_date:
        dates = generate_decision_days(args.start_date, args.end_date)
    else:
        parser.error("either --dates or both --start-date and --end-date are required")
    factor_names = parse_alpha_spec(args.alphas)
    backend = make_backend(config)
    run_inference(
        backend,
        factor_des_dir=args.factor_des_dir,
        dates=dates,
        output_dir=args.output_dir or config.get("output_dir", "result/alpha_select"),
        config=config,
        market_state_dir=args.market_state_dir,
        factor_names=factor_names,
    )


if __name__ == "__main__":
    main()
