#!/usr/bin/env python3
"""Parse Alpha-R1 result_*.json files into selections.json + summary.csv."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.parsing import parse_result_dir, write_parsed_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True,
                        help="directory containing result_YYYYMMDD.json files")
    parser.add_argument("--output-dir", default=None,
                        help="defaults to <result-dir>/parsed")
    parser.add_argument("--max-factors", type=int, default=10)
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(args.result_dir) / "parsed")
    selections, invalid = parse_result_dir(args.result_dir, args.max_factors)
    write_parsed_outputs(selections, invalid, output_dir)
    if invalid:
        print(f"[parse] {len(invalid)} invalid response(s):")
        for r in invalid:
            print(f"  {r['date']}: {r['source_file']}")


if __name__ == "__main__":
    main()
