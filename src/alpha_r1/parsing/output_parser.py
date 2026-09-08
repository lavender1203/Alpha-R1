"""Parsing of Alpha-R1 model outputs.

Extracts the selected factor list from ``<alpha_list>...</alpha_list>``,
validates factor names against the Alpha101 pool, de-duplicates and enforces
the max-factors constraint (paper Section 3.4.2 structural validity).
"""

import csv
import json
import re
from pathlib import Path

from ..factors.alpha101 import ALPHA_NAMES

_LIST_RE = re.compile(r"<alpha_list>(.*?)</alpha_list>", re.DOTALL | re.IGNORECASE)
_NAME_RE = re.compile(r"alpha(\d{3})", re.IGNORECASE)


def parse_alpha_list(response: str, max_factors: int = 10) -> list[str] | None:
    """Extract the validated factor list from a raw model response.

    Returns ``None`` when the response does not contain a well-formed
    ``<alpha_list>`` block (callers treat this as a format failure).
    """
    if not response:
        return None
    m = _LIST_RE.search(response)
    if not m:
        return None
    names = []
    for raw in _NAME_RE.findall(m.group(1)):
        name = f"alpha{raw}"
        if name in ALPHA_NAMES and name not in names:
            names.append(name)
    return names[:max_factors]


def parse_result_file(path: str | Path, max_factors: int = 10) -> dict:
    """Parse one ``result_YYYYMMDD.json`` file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    factors = parse_alpha_list(data.get("raw_response", ""), max_factors)
    return {
        "date": data.get("date", ""),
        "selected_factors": factors or [],
        "format_valid": factors is not None,
        "model": data.get("model", ""),
        "source_file": str(path),
    }


def parse_result_dir(result_dir: str | Path, max_factors: int = 10) -> tuple[dict, list[dict]]:
    """Parse every result file in a directory.

    Returns:
        selections: ``{date: [alpha, ...]}``
        invalid: records of files whose response failed format validation
    """
    result_dir = Path(result_dir)
    selections: dict[str, list[str]] = {}
    invalid: list[dict] = []
    for path in sorted(result_dir.glob("result_*.json")):
        record = parse_result_file(path, max_factors)
        selections[record["date"]] = record["selected_factors"]
        if not record["format_valid"]:
            invalid.append({"date": record["date"], "source_file": record["source_file"]})
    return selections, invalid


def write_parsed_outputs(selections: dict, invalid: list[dict], output_dir: str | Path) -> None:
    """Write ``selections.json`` and ``summary.csv`` for a parsed result directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "selections.json").write_text(
        json.dumps(selections, ensure_ascii=False, indent=2)
    )

    total = len(selections)
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "n_factors", "factors", "format_valid"])
        invalid_dates = {r["date"] for r in invalid}
        for date, factors in sorted(selections.items()):
            writer.writerow([date, len(factors), ";".join(factors), date not in invalid_dates])
    print(f"[parse] {total - len(invalid)}/{total} valid -> {output_dir}")
