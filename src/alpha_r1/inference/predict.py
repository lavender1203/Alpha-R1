"""Alpha-R1 factor-selection inference loop (paper Section 3.3).

For each decision day the semantic context C_t = {alpha_des} (+ optional
market state S_t) is sent to the model; raw responses are stored as
``result_YYYYMMDD.json`` for downstream parsing.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from ..factors.descriptions import format_descriptions_block, load_descriptions
from ..parsing.output_parser import parse_alpha_list
from .prompts import build_selection_prompt


def make_backend(config: dict):
    backend = config.get("backend", "hf")
    if backend == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(
            model_id=config["model_id"],
            temperature=config.get("temperature", 0.0),
            top_p=config.get("top_p", 0.7),
            max_tokens=config.get("max_tokens", 12288),
            max_model_len=config.get("max_model_len", 40960),
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
        )
    if backend == "hf":
        from .model import HFBackend

        return HFBackend(
            model_id=config["model_id"],
            temperature=config.get("temperature", 0.0),
            top_p=config.get("top_p", 0.7),
            max_tokens=config.get("max_tokens", 12288),
            device_map=config.get("device_map", "auto"),
            torch_dtype=config.get("torch_dtype", "bfloat16"),
        )
    raise ValueError(f"unknown backend {backend!r}; expected 'hf' or 'vllm'")


def generate_decision_days(start_date: str, end_date: str) -> list[str]:
    """Weekday decision days in [start_date, end_date] (YYYY-MM-DD)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days


def _load_market_state(market_state_dir: Path | None, date: str) -> str | None:
    if market_state_dir is None:
        return None
    for candidate in (f"{date}.txt", f"{date}_23:59:59.txt"):
        path = market_state_dir / candidate
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return None


def run_inference(backend, factor_des_dir: str | Path, dates: list[str],
                  output_dir: str | Path, config: dict,
                  market_state_dir: str | Path | None = None,
                  factor_names: list[str] | None = None,
                  max_retries: int = 3) -> list[dict]:
    """Run factor selection for each decision day and save raw responses."""
    descriptions = load_descriptions(factor_des_dir, factor_names)
    factor_block = format_descriptions_block(descriptions)
    max_factors = config.get("max_factors", 10)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = Path(market_state_dir) if market_state_dir else None
    if state_dir is not None:
        available = {p.stem[:10] for p in state_dir.glob("*.txt")}
        dates = [d for d in dates if d in available]
        if not dates:
            raise ValueError(f"no market state files matching the requested dates in {state_dir}")

    results = []
    for i, date in enumerate(dates, start=1):
        prompt = build_selection_prompt(
            target_date=date,
            factor_block=factor_block,
            market_state=_load_market_state(state_dir, date),
            max_factors=max_factors,
        )
        response, format_valid = "", False
        for attempt in range(max_retries):
            try:
                response = backend.generate(prompt)
            except Exception as e:
                print(f"[inference] {date} attempt {attempt + 1} error: {e}")
                time.sleep(2 * (attempt + 1))
                continue
            if parse_alpha_list(response, max_factors) is not None:
                format_valid = True
                break
            print(f"[inference] {date} attempt {attempt + 1}: invalid format, retrying")

        record = {
            "date": date.replace("-", ""),
            "raw_response": response,
            "response_type": "xml",
            "model": config["model_id"],
            "backend": config.get("backend", "hf"),
            "format_valid": format_valid,
            "selected_factors": parse_alpha_list(response, max_factors) or [],
            "sampling": {
                "temperature": config.get("temperature", 0.0),
                "top_p": config.get("top_p", 0.7),
                "max_tokens": config.get("max_tokens", 12288),
            },
        }
        out_path = output_dir / f"result_{record['date']}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"[inference] {i}/{len(dates)} {date} -> {out_path} "
              f"(format_valid={format_valid})")
        results.append(record)
    return results
