"""Iterative market memory construction (paper Section 3.1.2).

Daily atomic units (price-market descriptions and news descriptions) are
aggregated week by week through an LLM:

    M_w = F_LLM(I_w + M_{w-1})

yielding the global market memory ``M_global`` over the backtest window, used
for factor semantic profiling (Section 3.2.1).
"""

import re
from datetime import datetime
from pathlib import Path

from .openrouter_client import OpenRouterClient
from .prompts import MEMORY_SYSTEM, MEMORY_UPDATE_PROMPT

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _load_daily_units(price_dir: Path, news_dir: Path) -> dict[str, str]:
    """Merge price and news descriptions into one text per calendar day."""
    daily: dict[str, dict[str, str]] = {}
    for kind, directory in (("price", price_dir), ("news", news_dir)):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.txt")):
            m = _DATE_RE.search(path.name)
            if not m:
                continue
            daily.setdefault(m.group(1), {})[kind] = path.read_text(encoding="utf-8").strip()

    merged = {}
    for date, parts in sorted(daily.items()):
        blocks = []
        if parts.get("price"):
            blocks.append(f"[Price market description]\n{parts['price']}")
        if parts.get("news"):
            blocks.append(f"[News market description]\n{parts['news']}")
        if blocks:
            merged[date] = "\n\n".join(blocks)
    return merged


def _weeks(dates: list[str]) -> list[list[str]]:
    """Group sorted dates into ISO weeks (Mon-Sun)."""
    groups: dict[tuple[int, int], list[str]] = {}
    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        key = dt.isocalendar()[:2]
        groups.setdefault(key, []).append(d)
    return [groups[k] for k in sorted(groups)]


async def build_market_memory(client: OpenRouterClient, price_dir: str | Path,
                              news_dir: str | Path, output_dir: str | Path,
                              start_date: str | None = None,
                              end_date: str | None = None) -> str:
    """Build M_global by iteratively aggregating daily units week by week.

    Weekly intermediate summaries are saved next to the final
    ``M_global.txt`` in ``output_dir``.
    """
    daily = _load_daily_units(Path(price_dir), Path(news_dir))
    if not daily:
        raise FileNotFoundError(
            f"no daily descriptions found in {price_dir} or {news_dir}"
        )
    if start_date:
        daily = {d: t for d, t in daily.items() if d >= start_date}
    if end_date:
        daily = {d: t for d, t in daily.items() if d <= end_date}
    if not daily:
        raise ValueError("no daily descriptions left after date filtering")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    memory = "(empty: the backtest window starts here)"
    weekly = _weeks(sorted(daily))
    for i, week_dates in enumerate(weekly, start=1):
        items = "\n\n".join(f"[{d}]\n{daily[d]}" for d in week_dates)
        prompt = MEMORY_UPDATE_PROMPT.format(
            start_date=week_dates[0],
            end_date=week_dates[-1],
            weekly_items=items,
            previous_memory=memory,
        )
        memory = await client.chat(MEMORY_SYSTEM, prompt)
        week_path = output_dir / f"M_week_{i:03d}_{week_dates[0]}_{week_dates[-1]}.txt"
        week_path.write_text(memory, encoding="utf-8")
        print(f"[memory] week {i}/{len(weekly)} ({week_dates[0]}..{week_dates[-1]}) -> {week_path}")

    out = output_dir / "M_global.txt"
    out.write_text(memory, encoding="utf-8")
    print(f"[memory] M_global -> {out}")
    return memory
