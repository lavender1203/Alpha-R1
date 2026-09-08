"""Alpha101 factor library (Kakushadze, 2016).

Single source of truth for the candidate factor pool. Contains the 82
computationally feasible factors retained in the Alpha-R1 paper (Section 4.1),
expressed in the original WorldQuant-style formula language.
"""

ALPHA101 = {
    'alpha001': '(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)',
    'alpha002': '(correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))',
    'alpha003': '(-1 * correlation(rank(open), rank(volume), 10))',
    'alpha004': '(-1 * Ts_Rank(rank(low), 9))',
    'alpha005': '(rank((open - (sum(vwap, 10) / 10))) * (rank((close - vwap))))',
    'alpha006': '(-1 * correlation(open, volume, 10))',
    'alpha007': '((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))',
    'alpha008': '(rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))',
    'alpha009': '(delta(close, 1))',
    'alpha010': 'rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (1* delta(close, 1))))',
    'alpha011': '((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))*sign(close - vwap)',
    'alpha012': '(sign(delta(volume, 1)) * delta(close, 1))',
    'alpha013': '(rank(covariance(rank(close), rank(volume), 5)) - 0.5)* sign(delta(close,5))',
    'alpha014': '((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))*(-1 * sign(delta(close,10)))',
    'alpha015': '(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))',
    'alpha016': '(-1 * rank(covariance(rank(high), rank(volume), 5)))',
    'alpha017': '(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5))) * (-1 * sign(delta(close,5)))',
    'alpha018': '(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))',
    'alpha019': '((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))',
    'alpha020': '(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))',
    'alpha021': '((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))',
    'alpha022': '(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))',
    'alpha023': '(((sum(high, 20) / 20) < high) ?  delta(high, 2) : 0)',
    'alpha024': '((close - delay(close, 5)) / delay(close, 5))',
    'alpha025': 'rank((-1 * returns * adv20 * vwap * (high - close)) / ((sum(adv20, 20) / 20) * (sum((high - close), 20) / 20)))',
    'alpha026': 'ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)',
    'alpha027': '((0.5 > rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)',
    'alpha028': 'scale((correlation(adv20, low, 5) + ((high + low) / 2) - close))',
    'alpha029': 'min(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 5) + ts_rank(delay((-1 * returns), 6), 5)',
    'alpha030': 'sign(close-delay(close,1))*((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20)',
    'alpha031': 'rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3))) + sign(scale(correlation(adv20, low, 12))) + sign(delta(close, 5))',
    'alpha032': 'scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230)))',
    'alpha033': 'rank(((1 - (open / close))^1))',
    'alpha034': 'rank((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1))))',
    'alpha035': '(Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16)) * (1 - Ts_Rank(returns, 32)))',
    'alpha036': '(rank(correlation((close - open), delay(volume, 1), 15)) * 2.21 + rank((open- close)) * 0.7 + rank(Ts_Rank(delay((-1 * returns), 6), 5)) * 0.73 + rank(abs(correlation(vwap,adv20, 6))) + rank((((sum(close, 200) / 200) - open) * (close - open))) * 0.6)',
    'alpha037': '(rank(correlation(delay((close - open), 1), close, 200)) + rank((close - open)))',
    'alpha038': '(rank(Ts_Rank(close, 10)) * rank((close / open)))',
    'alpha039': '((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250)))) * rank(-delta(close, 7))',
    'alpha040': '((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)) * rank(stddev(close, 10))',
    'alpha041': '(((high * low)^0.5) - vwap)',
    'alpha042': '(rank((vwap - close)) / rank((vwap + close)))',
    'alpha043': '(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))',
    'alpha044': '(-1 * correlation(high, rank(volume), 5))',
    'alpha045': '(rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))',
    'alpha046': 'rank(-((((delay(close, 10) - delay(close, 20)) / delay(close, 20)) - ((close - delay(close, 10)) / delay(close, 10)))))',
    'alpha047': '((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5)) - rank((vwap - delay(vwap, 5)))',
    'alpha049': '(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) > (0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))',
    'alpha050': 'ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)',
    'alpha051': '(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) > (0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))',
    'alpha052': '(((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5)',
    'alpha053': 'delta((((close - low) - (high - close)) / (close - low)), 9)',
    'alpha054': '((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5))) * rank(ts_rank(volume, 5))',
    'alpha055': '(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6)) * rank(stddev(close, 20))',
    'alpha057': '((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))',
    'alpha060': '((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))',
    'alpha061': 'sign(rank((vwap - ts_min(vwap, 16))) - rank(correlation(vwap, adv180, 18)))',
    'alpha062': '(rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) - rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * (-ts_sum(returns, 5))',
    'alpha064': '(rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 13),sum(adv120, 13), 17)) - rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 4))) * -1',
    'alpha065': 'rank(correlation(vwap, sum(adv60, 9), 6)) - rank(open - ts_min(open, 14))',
    'alpha066': '(rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611))',
    'alpha068': 'rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157)) - Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333)',
    'alpha071': 'max(rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501)), rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662)))',
    'alpha072': 'rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) - rank(decay_linear(-correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011))',
    'alpha073': 'max(rank(decay_linear(-delta(vwap, 5), 3)),Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3), 17))',
    'alpha074': '(rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11)) - rank(correlation(close, sum(adv30, 37), 15))) / (rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11)) + rank(correlation(close, sum(adv30, 37), 15)) + 1e-6)',
    'alpha075': '(rank(correlation(vwap, volume, 4.24304)) - rank(correlation(rank(low), rank(adv50),12.4413))) * (-Ts_Rank(close, 20) + 0.5)',
    'alpha077': 'rank(decay_linear((vwap - ((high + low) / 2)), 20.0451)) + rank(decay_linear(-correlation(((high + low) / 2), adv40, 3.1614), 5.64125))',
    'alpha078': 'rank(correlation(ts_mean((low * 0.352233 + vwap * 0.647767), 20), ts_mean(adv40, 20), 7)) + rank(correlation(vwap, volume, 6))',
    'alpha081': 'rank(correlation(vwap, sum(adv10, 50), 8)) - rank(correlation(rank(vwap), rank(volume), 5))',
    'alpha083': '(rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume)) * (vwap - close)) / ((high - low) / (sum(close, 5) / 5))',
    'alpha084': 'SignedPower(Ts_Rank((vwap - ts_max(vwap, 15)), 21), Abs(delta(close,5)) + 1)',
    'alpha085': '(rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408))) * sign(ts_sum(returns, 5))',
    'alpha086': '(rank(close - vwap) - Ts_Rank(correlation(close, sum(adv20, 15), 6), 20)) * -1',
    'alpha088': 'min(rank(decay_linear(-((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), Ts_Rank(decay_linear(-correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))',
    'alpha092': 'max(Ts_Rank(decay_linear((((high + low) / 2 + close) > (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(-low), rank(adv30), 7.58555), 6.94024), 6.80584))',
    'alpha094': 'rank((vwap - ts_min(vwap, 12)) * correlation(Ts_Rank(vwap, 20), Ts_Rank(adv60, 4), 18))',
    'alpha095': 'sign(Ts_Rank((rank(correlation(sum(((high + low)/ 2), 20), sum(adv40, 20), 13))^5), 12) - rank(open - ts_min(open, 12)))',
    'alpha096': 'max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 4),4), 8), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7),Ts_Rank(adv60, 4), 4), 13), 14), 13))',
    'alpha098': '(rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) + rank(decay_linear(-Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))',
    'alpha099': '(correlation(sum(((high + low) / 2), 20), sum(adv60, 20), 9) - correlation(low, volume, 6)) * -1',
    'alpha101': '((close - open) / ((high - low) + .001))',
}

#: Valid factor names (lowercase, ``alpha`` + zero-padded index).
ALPHA_NAMES = frozenset(ALPHA101)


def get_formula(name: str) -> str:
    """Return the formula for a factor, e.g. ``get_formula("alpha001")``."""
    key = name.lower()
    if key not in ALPHA101:
        raise KeyError(f"unknown factor {name!r}; {len(ALPHA101)} factors available")
    return ALPHA101[key]


def is_valid_name(name: str) -> bool:
    return name.lower() in ALPHA_NAMES


def parse_alpha_spec(spec: str) -> list[str]:
    """Parse a factor subset spec: 'all', '001-101', '001,005,010' or '001'."""
    spec = spec.strip().lower()
    if spec == "all":
        return sorted(ALPHA101)
    if "," in spec:
        names = [f"alpha{int(x):03d}" for x in spec.split(",")]
    elif "-" in spec:
        start, end = spec.split("-")
        names = [f"alpha{i:03d}" for i in range(int(start), int(end) + 1)]
    else:
        names = [f"alpha{int(spec):03d}"]
    selected = [n for n in names if n in ALPHA101]
    if not selected:
        raise ValueError(f"spec {spec!r} selects no known factors")
    return selected
