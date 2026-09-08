"""Prompt templates for factor semantic description generation.

Follows the paper's construction (Section 3.2.1):
``alpha_des,i = F_LLM(M_global, P_i)`` — the global market memory combined
with the factor's backtest performance vector yields a structured profile
covering mechanism, regime suitability and failure conditions.
"""

MEMORY_SYSTEM = (
    "You are a senior macro and market analyst. You integrate daily price "
    "action and news narratives into a coherent, compressed long-term market "
    "memory, preserving regime shifts, style rotation and major events."
)

MEMORY_UPDATE_PROMPT = """Below are the daily market descriptions (price action and news) for the week from {start_date} to {end_date}, together with the accumulated market memory so far.

Update the memory into a continuous market narrative covering the evolution of trends, styles, sentiment and key events. Requirements: objective, compressed, no loss of critical turning points, at most 800 words.

[Daily market information this week]
{weekly_items}

[Previous market memory]
{previous_memory}

Output the updated market memory directly, without extra commentary."""

DESCRIPTION_SYSTEM = (
    "You are a senior quantitative investment expert specializing in alpha "
    "factor analysis and market regime adaptability. You combine a factor's "
    "mathematical rationale, historical backtest results and market "
    "environment into a comprehensive, insightful factor report."
)

DESCRIPTION_PROMPT = """Generate a comprehensive analysis report for the alpha factor below.

# Report format

{alpha_name} factor analysis report:
1. Factor formula: state the mathematical formula first.
2. Mathematical logic and economic rationale: explain the principle and the investment logic behind it.
3. Market-regime effectiveness analysis: reason from IC dynamics, strategy NAV, excess return over the benchmark and the historical market environment; conclude under which market conditions the factor works and under which it fails.
4. Emphasize how market regime features (bull, bear, range-bound, style rotation, etc.) affect factor effectiveness.

# Requirements

1. Objective and concise; do not restate exact numbers, only describe the effectiveness conclusions across market regimes; no investment advice; at most 500 words.
2. Output the complete report directly, without extra commentary.

# Reference information

[Factor]
Name: {alpha_name}
Formula: {formula}

[Single-factor backtest results ({bt_start} to {bt_end}, universe {market})]
{backtest_summary}

[Market memory over the backtest window]
{market_memory}
"""


def format_backtest_summary(p_i: dict) -> str:
    """Render a backtest result JSON (P_i) as a compact text summary."""
    lines = []
    ic = p_i.get("ic") or {}
    if ic:
        lines.append(
            "IC: mean_ic={mean_ic}, rank_ic={mean_rank_ic}, ic_ir={ic_ir}, "
            "rank_ic_ir={rank_ic_ir}, rank_ic_win_rate={rank_ic_win_rate}".format(
                **{k: ic.get(k) for k in
                   ("mean_ic", "mean_rank_ic", "ic_ir", "rank_ic_ir", "rank_ic_win_rate")}
            )
        )
    pf = p_i.get("portfolio") or {}
    if pf:
        lines.append(
            "Portfolio: total_return={total_return}, annual_return={annual_return}, "
            "annual_volatility={annual_volatility}, sharpe={sharpe_ratio}, "
            "max_drawdown={max_drawdown}, avg_daily_turnover={avg_daily_turnover}".format(
                **{k: pf.get(k) for k in
                   ("total_return", "annual_return", "annual_volatility",
                    "sharpe_ratio", "max_drawdown", "avg_daily_turnover")}
            )
        )
        for year, stats in sorted((pf.get("yearly_breakdown") or {}).items()):
            lines.append(f"  {year}: return={stats.get('return')}, sharpe={stats.get('sharpe')}")
    return "\n".join(lines) if lines else "(no backtest result available)"
