"""Factor semantic description generation (paper Section 3.2.1).

For each factor i, an LLM combines the global market memory with the factor's
backtest performance vector::

    alpha_des,i = F_LLM(M_global, P_i)

producing a structured profile (mechanism / economic logic / regime
suitability / failure conditions) saved as ``alphaNNN.txt``. These files are
the input of the Alpha-R1 inference prompt.
"""

import asyncio
import json
from pathlib import Path

from ..factors.alpha101 import ALPHA101
from .openrouter_client import OpenRouterClient
from .prompts import DESCRIPTION_PROMPT, DESCRIPTION_SYSTEM, format_backtest_summary


def _load_backtest(backtest_dir: Path, name: str) -> dict | None:
    path = backtest_dir / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_memory(memory_dir_or_file: str | Path) -> str:
    path = Path(memory_dir_or_file)
    if path.is_dir():
        path = path / "M_global.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"market memory not found at {path}; run scripts/build_market_memory.py first"
        )
    return path.read_text(encoding="utf-8").strip()


async def _generate_one(client: OpenRouterClient, name: str, p_i: dict | None,
                        market_memory: str, semaphore: asyncio.Semaphore,
                        output_dir: Path) -> dict:
    async with semaphore:
        summary = format_backtest_summary(p_i) if p_i else "(no backtest result available)"
        prompt = DESCRIPTION_PROMPT.format(
            alpha_name=name,
            formula=ALPHA101[name],
            bt_start=(p_i or {}).get("window", {}).get("start", "?"),
            bt_end=(p_i or {}).get("window", {}).get("end", "?"),
            market=(p_i or {}).get("market", "?"),
            backtest_summary=summary,
            market_memory=market_memory,
        )
        try:
            report = await client.chat(DESCRIPTION_SYSTEM, prompt)
        except Exception as e:
            print(f"[generate] {name} failed: {e}")
            return {"factor": name, "status": "error", "error": str(e)}

        out_path = output_dir / f"{name}.txt"
        out_path.write_text(report, encoding="utf-8")
        print(f"[generate] {name} -> {out_path}")
        return {"factor": name, "status": "success", "output_file": str(out_path)}


async def generate_descriptions(client: OpenRouterClient, names: list[str],
                                config: dict) -> list[dict]:
    """Generate ``alphaNNN.txt`` descriptions for the given factors."""
    names = [n.lower() for n in names]
    unknown = [n for n in names if n not in ALPHA101]
    if unknown:
        raise ValueError(f"unknown factor names: {unknown}")

    market_memory = _load_memory(config["market_memory_dir"])
    backtest_dir = Path(config["backtest_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(config.get("max_workers", 3))
    tasks = [
        _generate_one(client, name, _load_backtest(backtest_dir, name),
                      market_memory, semaphore, output_dir)
        for name in names
    ]
    results = []
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
        print(f"[generate] progress: {len(results)}/{len(tasks)}")

    ok = sum(1 for r in results if r["status"] == "success")
    print(f"[generate] done: {ok}/{len(results)} succeeded")
    return results
