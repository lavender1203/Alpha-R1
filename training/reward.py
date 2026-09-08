"""Reference market-feedback reward for Alpha-R1 GRPO training (paper Section 3.4.2).

Simplified reproduction of R_final = R_adjusted - P_structural:

1. Parse <alpha_list> from the model response; unparseable output or an empty
   all-invalid selection incurs the structural penalty (lowest score). Lists
   longer than MAX_FACTORS are silently truncated.
2. R_base: score stocks with a fixed linear model (beta_i estimated offline on
   the 2020-2023 pre-training window, loaded from an external file), take the
   top-N equal-weight portfolio, and compute its excess return over the
   benchmark over the holding period H, scaled by 100.
3. R_adjusted: an optional LLM-as-judge consistency penalty adjusts R_base
   asymmetrically (Eq. 9). The judge is a stub here; plug in any LLM client.

This is a reference implementation: numeric results depend on your factor
data, price data and beta estimates. Compatible with verl's
custom_reward_function interface (`compute_score` / `compute_score_batch`).

When no explicit RewardContext is passed, one is built lazily from environment
variables: ALPHA_R1_BETAS_CSV, ALPHA_R1_FACTOR_VALUES_DIR, ALPHA_R1_PRICES_CSV,
ALPHA_R1_BENCHMARK_CSV.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

_LIST_RE = re.compile(r"<alpha_list>(.*?)</alpha_list>", re.DOTALL | re.IGNORECASE)
_NAME_RE = re.compile(r"alpha(\d{3})", re.IGNORECASE)

MAX_FACTORS = 10        # structural constraint (paper Section 3.3)
TOP_N = 10              # portfolio size (paper Section 4.1.2)
HOLDING_DAYS = 5        # holding period H
MIN_SCORE = -10.0       # score assigned on structural violations


def parse_alpha_list(response: str, valid_names: set[str]) -> list[str] | None:
    """Extract and validate the selected factor list; None if unparseable."""
    if not response:
        return None
    m = _LIST_RE.search(response)
    if not m:
        return None
    names = []
    for raw in _NAME_RE.findall(m.group(1)):
        name = f"alpha{raw}"
        if name in valid_names and name not in names:
            names.append(name)
    return names[:MAX_FACTORS]


class RewardContext:
    """Holds the data the reward needs, loaded once per training run.

    Expected files:
      - betas_csv: columns ``factor,beta`` (+ optional intercept row ``_intercept``)
      - factor_values_dir: per-day CSVs ``YYYY-MM-DD.csv`` with columns
        ``instrument`` + one column per factor (previous-day factor values)
      - prices_csv: columns ``date,instrument,close`` (long format)
      - benchmark_csv: columns ``date,close``
    """

    def __init__(self, betas_csv: str, factor_values_dir: str,
                 prices_csv: str, benchmark_csv: str):
        betas = pd.read_csv(betas_csv).set_index("factor")["beta"]
        if "_intercept" in betas.index:
            self.beta0 = float(betas["_intercept"])
            betas = betas.drop("_intercept")
        else:
            self.beta0 = 0.0
        self.betas = betas
        self.factor_values_dir = Path(factor_values_dir)
        prices = pd.read_csv(prices_csv, parse_dates=["date"])
        self.prices = prices.pivot(index="date", columns="instrument", values="close").sort_index()
        benchmark = pd.read_csv(benchmark_csv, parse_dates=["date"],
                                index_col="date")["close"].sort_index()
        # Align to the price calendar so positional indexing below is consistent.
        self.benchmark = benchmark.reindex(self.prices.index).ffill()

    def factor_values(self, date: pd.Timestamp) -> pd.DataFrame | None:
        path = self.factor_values_dir / f"{date:%Y-%m-%d}.csv"
        if not path.exists():
            return None
        return pd.read_csv(path).set_index("instrument")

    def holding_returns(self, date: pd.Timestamp) -> tuple[pd.Series, float] | None:
        """Per-stock and benchmark returns from `date` close to close+H."""
        cal = self.prices.index
        pos = cal.searchsorted(date)
        if pos >= len(cal) or cal[pos] != date or pos + HOLDING_DAYS >= len(cal):
            return None
        start, end = self.prices.iloc[pos], self.prices.iloc[pos + HOLDING_DAYS]
        bench_start, bench_end = self.benchmark.iloc[pos], self.benchmark.iloc[pos + HOLDING_DAYS]
        if not (np.isfinite(bench_start) and np.isfinite(bench_end)):
            return None
        return end / start - 1, float(bench_end / bench_start - 1)


_DEFAULT_CONTEXT: "RewardContext | None" = None


def get_default_context() -> RewardContext:
    """Lazy singleton context built from ALPHA_R1_* environment variables."""
    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is None:
        env = {k: os.environ.get(k) for k in (
            "ALPHA_R1_BETAS_CSV", "ALPHA_R1_FACTOR_VALUES_DIR",
            "ALPHA_R1_PRICES_CSV", "ALPHA_R1_BENCHMARK_CSV")}
        missing = [k for k, v in env.items() if not v]
        if missing:
            raise ValueError(
                "no RewardContext provided and environment variables are unset: "
                + ", ".join(missing))
        _DEFAULT_CONTEXT = RewardContext(
            betas_csv=env["ALPHA_R1_BETAS_CSV"],
            factor_values_dir=env["ALPHA_R1_FACTOR_VALUES_DIR"],
            prices_csv=env["ALPHA_R1_PRICES_CSV"],
            benchmark_csv=env["ALPHA_R1_BENCHMARK_CSV"])
    return _DEFAULT_CONTEXT


def llm_judge_penalty(context: str, response: str) -> float:
    """LLM-as-judge consistency penalty in [0, 10] (paper Eq. 8). Stub: plug in
    your LLM client here; returns 0 (no penalty) by default."""
    return 0.0


def compute_score(data_source: str, solution_str: str, ground_truth,
                  extra_info: dict | None = None, context: RewardContext | None = None,
                  enable_llm_judge: bool = False, **_kwargs) -> float:
    """verl-compatible reward entry. Returns a scalar reward.

    ``extra_info`` must carry ``date`` (YYYY-MM-DD) of the decision day.
    """
    extra_info = extra_info or {}
    if context is None:
        context = get_default_context()

    factors = parse_alpha_list(solution_str, set(context.betas.index))
    # An empty selection is allowed at inference time (the prompt permits
    # "select none"), but during training it is treated as a structural
    # violation: the policy must learn to commit to a non-empty subset.
    if not factors:  # unparseable, all-invalid, or empty selection
        return MIN_SCORE

    date = pd.Timestamp(extra_info["date"])
    values = context.factor_values(date)
    horizon = context.holding_returns(date)
    if values is None or horizon is None:
        return MIN_SCORE
    stock_returns, bench_return = horizon

    # Linear scoring: predicted return = beta0 + sum(beta_i * V_i)
    available = [f for f in factors if f in values.columns]
    if not available:
        return MIN_SCORE
    pred = context.beta0 + values[available].mul(context.betas[available], axis=1).sum(axis=1)
    top = pred.nlargest(TOP_N).index
    port_return = float(stock_returns.reindex(top).mean())
    if not np.isfinite(port_return):
        return MIN_SCORE

    r_base = (port_return - bench_return) * 100  # Eq. 7

    if enable_llm_judge:
        p_norm = llm_judge_penalty(str(extra_info), solution_str) / 10.0
        r_adj = r_base * (1 - p_norm) if r_base > 0 else r_base * (1 + p_norm)  # Eq. 9
    else:
        r_adj = r_base
    return float(r_adj)


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos,
                        context: RewardContext | None = None, **kwargs) -> list[float]:
    """Batch wrapper matching verl's batch reward-manager convention."""
    return [
        compute_score(ds, sol, gt, ei, context=context, **kwargs)
        for ds, sol, gt, ei in zip(data_sources, solution_strs, ground_truths, extra_infos)
    ]


if __name__ == "__main__":
    # Offline beta estimation sketch (paper Section 4.1.1, 2020-2023 window):
    # regress per-day cross-sectional forward returns on factor values, then
    # average the daily coefficients. Replace with your own estimation data.
    print("This module is meant to be used as a verl custom_reward_function.")
