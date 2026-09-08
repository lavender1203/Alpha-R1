"""Loading and concatenation of factor semantic descriptions.

Factor descriptions are plain-text reports (one ``alphaNNN.txt`` per factor)
produced by :mod:`alpha_r1.generation`. The inference prompt is built by
concatenating these files, so this module is the shared loading logic used by
both inference and tooling.
"""

import re
from pathlib import Path

from .alpha101 import ALPHA_NAMES

_NAME_RE = re.compile(r"alpha(\d{3})", re.IGNORECASE)


def name_from_path(path: Path) -> str | None:
    """Extract the canonical factor name (``alphaNNN``) from a file path."""
    m = _NAME_RE.search(Path(path).stem)
    return f"alpha{m.group(1)}" if m else None


def load_descriptions(des_dir: str | Path, names: list[str] | None = None) -> dict[str, str]:
    """Load factor descriptions from a directory of ``alphaNNN.txt`` files.

    Args:
        des_dir: directory containing one text file per factor.
        names: optional whitelist of factor names to load (defaults to all
            valid Alpha101 files found in ``des_dir``).

    Returns:
        Mapping of factor name -> description text, sorted by factor name.
    """
    des_dir = Path(des_dir)
    if not des_dir.is_dir():
        raise FileNotFoundError(f"factor description directory not found: {des_dir}")

    wanted = {n.lower() for n in names} if names else None
    descriptions: dict[str, str] = {}
    for path in sorted(des_dir.glob("*.txt")):
        name = name_from_path(path)
        if name is None or name not in ALPHA_NAMES:
            continue
        if wanted is not None and name not in wanted:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            descriptions[name] = text
    if not descriptions:
        raise ValueError(f"no valid factor descriptions found in {des_dir}")
    return descriptions


def format_descriptions_block(descriptions: dict[str, str]) -> str:
    """Concatenate descriptions into the prompt block used at inference time."""
    parts = []
    for name, text in descriptions.items():
        parts.append(f"[{name}]\n{text}")
    return "\n\n".join(parts)
