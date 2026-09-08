"""Prompt construction for Alpha-R1 factor-selection inference.

The decision context (paper Section 3.3) concatenates the semantic factor
descriptions with the optional current market state, and asks for a factor
list wrapped in <alpha_list>...</alpha_list>.
"""

SYSTEM_PROMPT = (
    "You are a senior quantitative investment expert specializing in selecting "
    "the most suitable alpha factor combination for current market conditions "
    "and asset characteristics."
)

SELECTION_PROMPT = """You are selecting alpha factors for a 5-day short-horizon stock strategy (positions entered on the decision day and held for 5 trading days) for the trading day {target_date}.

[Target date]
{target_date}
{market_section}
[Candidate factor descriptions]
{factor_block}
{asset_section}
Analysis steps:
1. Read every candidate factor description carefully and analyze each factor's nature and characteristics.
2. Given the current market environment, evaluate each factor's expected performance and pre-select candidate factors.
3. Further refine the selection considering the asset pool's industry distribution, market-cap profile and style preferences.
4. Make the final decision: select at most {max_factors} factors.

Output requirements:
1. First reason in detail about why you select these factors and why you exclude others.
2. Then output the final factor list wrapped in <alpha_list></alpha_list>, using names in the form <alpha00n>.
3. Select the combination you expect to be most profitable; the total number of factors must not exceed {max_factors}.
4. If you believe no factor can deliver positive returns, select none.
5. Consider every candidate factor; do not omit any from your analysis.

Output format:

Factor analysis reasoning: ...
The most suitable factor selection for the current market is: <alpha_list><alpha001>...</alpha00n></alpha_list>"""


def build_selection_prompt(target_date: str, factor_block: str,
                           market_state: str | None = None,
                           asset_info: str | None = None,
                           max_factors: int = 10) -> str:
    market_section = f"\n[Current market environment]\n{market_state}\n" if market_state else ""
    asset_section = f"\n[Asset pool information]\n{asset_info}\n" if asset_info else ""
    return SELECTION_PROMPT.format(
        target_date=target_date,
        factor_block=factor_block,
        market_section=market_section,
        asset_section=asset_section,
        max_factors=max_factors,
    )
