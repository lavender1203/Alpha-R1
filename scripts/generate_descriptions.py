#!/usr/bin/env python3
"""Generate factor semantic descriptions alpha_des (paper Section 3.2.1)."""

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_r1.factors.alpha101 import parse_alpha_spec
from alpha_r1.generation import OpenRouterClient, generate_descriptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--model", default=None, help="override the OpenRouter model id")
    parser.add_argument("--alphas", default="all",
                        help="factor subset: 'all', '001-101' or '001,005' (default: all)")
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
    names = parse_alpha_spec(args.alphas)
    print(f"Generating descriptions for {len(names)} factors")
    asyncio.run(generate_descriptions(client, names, config))


if __name__ == "__main__":
    main()
