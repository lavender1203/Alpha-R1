#!/usr/bin/env python3
"""Build the global market memory M_global (paper Section 3.1.2)."""

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.generation import OpenRouterClient, build_market_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--model", default=None, help="override the OpenRouter model id")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    client = OpenRouterClient(
        model=args.model or config.get("model", ""),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 4096),
        max_retries=config.get("max_retries", 5),
        retry_delay=config.get("retry_delay", 2),
        base_url=config.get("base_url", "https://openrouter.ai/api/v1"),
    )
    asyncio.run(build_market_memory(
        client,
        price_dir=config["price_market_dir"],
        news_dir=config["news_dir"],
        output_dir=config["market_memory_dir"],
        start_date=args.start_date,
        end_date=args.end_date,
    ))


if __name__ == "__main__":
    main()
