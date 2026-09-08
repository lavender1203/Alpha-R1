"""Alpha101 formulas translated to qlib expression syntax.

qlib's built-in operators cover most of the WorldQuant operator set, with two
important differences:

- qlib ``Rank(x, N)`` is a *time-series* rolling percentile rank, which matches
  WorldQuant ``ts_rank(x, N)``.
- WorldQuant ``rank(x)`` is *cross-sectional* (per trading day across all
  instruments); qlib has no built-in equivalent, so we ship custom ``CSRank``
  and ``CSScale`` operators (plus ``DecayLinear``, ``SignedPower``,
  ``TsArgMax`` / ``TsArgMin`` for WorldQuant semantics).

The custom operators live in :mod:`alpha_r1.backtest.qlib_custom_ops` and are
registered through qlib's ``custom_ops`` mechanism; call
:func:`register_custom_ops` (or pass ``custom_ops=custom_ops_config()`` to
``qlib.init``) before evaluating any expression from :data:`QLIB_EXPRESSIONS`.

Rolling windows given as floats in the original formulas are rounded to the
nearest int, following the WorldQuant reference implementation.
"""

from ..factors.alpha101 import ALPHA101


def set_cs_universe(market: str) -> None:
    """Set the universe over which cross-sectional operators rank/scale."""
    from . import qlib_custom_ops

    qlib_custom_ops.set_cs_universe(market)


def clear_panel_cache() -> None:
    from . import qlib_custom_ops

    qlib_custom_ops.clear_panel_cache()


def register_custom_ops() -> None:
    """Register the custom operators with qlib's operator registry.

    qlib resets its registry on every ``qlib.init`` call, so registration must
    happen through the ``custom_ops`` config or after init.
    """
    from qlib.data.ops import Operators

    from .qlib_custom_ops import CUSTOM_OPS

    Operators.register(CUSTOM_OPS)


def custom_ops_config() -> list[dict]:
    """Return a ``custom_ops`` config suitable for ``qlib.init(custom_ops=...)``."""
    from .qlib_custom_ops import CUSTOM_OPS

    return [{"class": cls.__name__, "module_path": "alpha_r1.backtest.qlib_custom_ops"}
            for cls in CUSTOM_OPS]


# ---------------------------------------------------------------------------
# Formula translations
# ---------------------------------------------------------------------------
# Field / operator mapping used:
#   open/high/low/close/volume/vwap -> $open/$high/$low/$close/$volume/$vwap
#   returns                         -> ($close/Ref($close,1)-1)
#   advN                            -> Mean($volume, N)
#   rank(x)                         -> CSRank(x)          (cross-sectional)
#   ts_rank / Ts_Rank(x, N)         -> Rank(x, N)         (time-series)
#   delay -> Ref, sum/ts_sum -> Sum, ts_mean -> Mean, stddev -> Std
#   correlation -> Corr, covariance -> Cov, delta -> Delta
#   ts_min -> Min, ts_max -> Max, ts_argmax/Ts_ArgMax -> TsArgMax, Ts_ArgMin -> TsArgMin
#   decay_linear -> DecayLinear, scale -> CSScale, sign -> Sign, abs -> Abs, log -> Log
#   ternary (c) ? a : b             -> If((c), a, b)

QLIB_EXPRESSIONS = {
    'alpha001': '(CSRank(TsArgMax(SignedPower(If(($close/Ref($close,1)-1) < 0, Std(($close/Ref($close,1)-1), 20), $close), 2), 5)) - 0.5)',
    'alpha002': 'Corr(CSRank(Delta(Log($volume), 2)), CSRank(($close - $open) / $open), 6)',
    'alpha003': '(-1 * Corr(CSRank($open), CSRank($volume), 10))',
    'alpha004': '(-1 * Rank(CSRank($low), 9))',
    'alpha005': '(CSRank($open - Sum($vwap, 10) / 10) * CSRank($close - $vwap))',
    'alpha006': '(-1 * Corr($open, $volume, 10))',
    'alpha007': 'If(Mean($volume,20) < $volume, (-1 * Rank(Abs(Delta($close, 7)), 60)) * Sign(Delta($close, 7)), -1)',
    'alpha008': 'CSRank(Sum($open,5) * Sum(($close/Ref($close,1)-1), 5) - Ref(Sum($open,5) * Sum(($close/Ref($close,1)-1), 5), 10))',
    'alpha009': 'Delta($close, 1)',
    'alpha010': 'CSRank(If(0 < Min(Delta($close,1), 4), Delta($close,1), If(Max(Delta($close,1), 4) < 0, Delta($close,1), Delta($close,1))))',
    'alpha011': '(CSRank(Max($vwap-$close, 3)) + CSRank(Min($vwap-$close, 3))) * CSRank(Delta($volume, 3)) * Sign($close - $vwap)',
    'alpha012': '(Sign(Delta($volume, 1)) * Delta($close, 1))',
    'alpha013': '(CSRank(Cov(CSRank($close), CSRank($volume), 5)) - 0.5) * Sign(Delta($close,5))',
    'alpha014': '((-1 * CSRank(Delta(($close/Ref($close,1)-1), 3))) * Corr($open, $volume, 10)) * (-1 * Sign(Delta($close,10)))',
    'alpha015': '(-1 * Sum(CSRank(Corr(CSRank($high), CSRank($volume), 3)), 3))',
    'alpha016': '(-1 * CSRank(Cov(CSRank($high), CSRank($volume), 5)))',
    'alpha017': '(((-1 * CSRank(Rank($close, 10))) * CSRank(Delta(Delta($close, 1), 1))) * CSRank(Rank($volume / Mean($volume,20), 5))) * (-1 * Sign(Delta($close,5)))',
    'alpha018': '(-1 * CSRank(Std(Abs($close - $open), 5) + ($close - $open) + Corr($close, $open, 10)))',
    'alpha019': '((-1 * Sign(($close - Ref($close, 7)) + Delta($close, 7))) * (1 + CSRank(1 + Sum(($close/Ref($close,1)-1), 250))))',
    'alpha020': '(((-1 * CSRank($open - Ref($high, 1))) * CSRank($open - Ref($close, 1))) * CSRank($open - Ref($low, 1)))',
    'alpha021': 'If(0 < Min(Delta($close,1), 5), Delta($close,1), If(Max(Delta($close,1), 5) < 0, Delta($close,1), -1 * Delta($close,1)))',
    'alpha022': '(-1 * (Delta(Corr($high, $volume, 5), 5) * CSRank(Std($close, 20))))',
    'alpha023': 'If(Sum($high, 20) / 20 < $high, Delta($high, 2), 0)',
    'alpha024': '(($close - Ref($close, 5)) / Ref($close, 5))',
    'alpha025': 'CSRank((-1 * ($close/Ref($close,1)-1) * Mean($volume,20) * $vwap * ($high - $close)) / ((Sum(Mean($volume,20), 20) / 20) * (Sum(($high - $close), 20) / 20)))',
    'alpha026': 'Max(Corr(Rank($volume, 5), Rank($high, 5), 5), 3)',
    'alpha027': 'If(0.5 > CSRank(Sum(Corr(CSRank($volume), CSRank($vwap), 6), 2) / 2.0), -1, 1)',
    'alpha028': 'CSScale((Corr(Mean($volume,20), $low, 5) + (($high + $low) / 2) - $close))',
    'alpha029': 'Min(CSRank(CSRank(CSScale(Log(Sum(Min(CSRank(CSRank((-1 * CSRank(Delta(($close - 1),5))))), 2), 1))))), 5) + Rank(Ref((-1 * ($close/Ref($close,1)-1)), 6), 5)',
    'alpha030': 'Sign($close-Ref($close,1))*((1.0 - CSRank(((Sign(($close - Ref($close, 1))) + Sign((Ref($close, 1) - Ref($close, 2)))) + Sign((Ref($close, 2) - Ref($close, 3)))))) * Sum($volume, 5)) / Sum($volume, 20)',
    'alpha031': 'CSRank(CSRank(CSRank(DecayLinear((-1 * CSRank(CSRank(Delta($close, 10)))), 10)))) + CSRank((-1 * Delta($close, 3))) + Sign(CSScale(Corr(Mean($volume,20), $low, 12))) + Sign(Delta($close, 5))',
    'alpha032': 'CSScale(((Sum($close, 7) / 7) - $close)) + (20 * CSScale(Corr($vwap, Ref($close, 5), 230)))',
    'alpha033': 'CSRank((1 - ($open / $close)))',
    'alpha034': 'CSRank((1 - CSRank((Std(($close/Ref($close,1)-1), 2) / Std(($close/Ref($close,1)-1), 5)))) + (1 - CSRank(Delta($close, 1))))',
    'alpha035': '(Rank($volume, 32) * (1 - Rank((($close + $high) - $low), 16)) * (1 - Rank(($close/Ref($close,1)-1), 32)))',
    'alpha036': '(CSRank(Corr(($close - $open), Ref($volume, 1), 15)) * 2.21 + CSRank(($open - $close)) * 0.7 + CSRank(Rank(Ref((-1 * ($close/Ref($close,1)-1)), 6), 5)) * 0.73 + CSRank(Abs(Corr($vwap, Mean($volume,20), 6))) + CSRank((((Sum($close, 200) / 200) - $open) * ($close - $open))) * 0.6)',
    'alpha037': '(CSRank(Corr(Ref(($close - $open), 1), $close, 200)) + CSRank(($close - $open)))',
    'alpha038': '(CSRank(Rank($close, 10)) * CSRank(($close / $open)))',
    'alpha039': '((-1 * CSRank((Delta($close, 7) * (1 - CSRank(DecayLinear(($volume / Mean($volume,20)), 9)))))) * (1 + CSRank(Sum(($close/Ref($close,1)-1), 250)))) * CSRank(-Delta($close, 7))',
    'alpha040': '((-1 * CSRank(Std($high, 10))) * Corr($high, $volume, 10)) * CSRank(Std($close, 10))',
    'alpha041': '(Power(($high * $low), 0.5) - $vwap)',
    'alpha042': '(CSRank(($vwap - $close)) / CSRank(($vwap + $close)))',
    'alpha043': '(Rank(($volume / Mean($volume,20)), 20) * Rank((-1 * Delta($close, 7)), 8))',
    'alpha044': '(-1 * Corr($high, CSRank($volume), 5))',
    'alpha045': '(CSRank((Sum(Ref($close, 5), 20) / 20)) * Corr($close, $volume, 2)) * CSRank(Corr(Sum($close, 5), Sum($close, 20), 2))',
    'alpha046': 'CSRank(-((((Ref($close, 10) - Ref($close, 20)) / Ref($close, 20)) - (($close - Ref($close, 10)) / Ref($close, 10)))))',
    'alpha047': '((CSRank((1 / $close)) * $volume) / Mean($volume,20)) * (($high * CSRank(($high - $close))) / (Sum($high, 5) / 5)) - CSRank(($vwap - Ref($vwap, 5)))',
    'alpha049': 'If(((((Ref($close, 20) - Ref($close, 10)) / 10) - ((Ref($close, 10) - $close) / 10)) > 0.1), 1, ((-1 * 1) * ($close - Ref($close, 1))))',
    'alpha050': 'Max(CSRank(Corr(CSRank($volume), CSRank($vwap), 5)), 5)',
    'alpha051': 'If(((((Ref($close, 20) - Ref($close, 10)) / 10) - ((Ref($close, 10) - $close) / 10)) > 0.05), 1, ((-1 * 1) * ($close - Ref($close, 1))))',
    'alpha052': '(((-1 * Min($low, 5)) + Ref(Min($low, 5), 5)) * CSRank(((Sum(($close/Ref($close,1)-1), 240) - Sum(($close/Ref($close,1)-1), 20)) / 220))) * Rank($volume, 5)',
    'alpha053': 'Delta(((($close - $low) - ($high - $close)) / ($close - $low)), 9)',
    'alpha054': '((-1 * (($low - $close) * Power($open, 5))) / (($low - $high) * Power($close, 5))) * CSRank(Rank($volume, 5))',
    'alpha055': '(-1 * Corr(CSRank((($close - Min($low, 12)) / (Max($high, 12) - Min($low, 12)))), CSRank($volume), 6)) * CSRank(Std($close, 20))',
    'alpha057': '(($close - $vwap) / DecayLinear(CSRank(TsArgMax($close, 30)), 2))',
    'alpha060': '((2 * CSScale(CSRank((((($close - $low) - ($high - $close)) / ($high - $low)) * $volume)))) - CSScale(CSRank(TsArgMax($close, 10))))',
    'alpha061': 'Sign(CSRank(($vwap - Min($vwap, 16))) - CSRank(Corr($vwap, Mean($volume,180), 18)))',
    'alpha062': '(CSRank(Corr($vwap, Sum(Mean($volume,20), 22), 10)) - CSRank(((CSRank($open) + CSRank($open)) < (CSRank((($high + $low) / 2)) + CSRank($high))))) * (-Sum(($close/Ref($close,1)-1), 5))',
    'alpha064': '(CSRank(Corr(Sum((($open * 0.178404) + ($low * (1 - 0.178404))), 13), Sum(Mean($volume,120), 13), 17)) - CSRank(Delta((((($high + $low) / 2) * 0.178404) + ($vwap * (1 - 0.178404))), 4))) * -1',
    'alpha065': 'CSRank(Corr($vwap, Sum(Mean($volume,60), 9), 6)) - CSRank($open - Min($open, 14))',
    'alpha066': '(CSRank(DecayLinear(Delta($vwap, 4), 7)) + Rank(DecayLinear((((($low * 0.96633) + ($low * (1 - 0.96633))) - $vwap) / ($open - (($high + $low) / 2))), 11), 7))',
    'alpha068': 'CSRank(Delta((($close * 0.518371) + ($low * (1 - 0.518371))), 1)) - Rank(Corr(CSRank($high), CSRank(Mean($volume,15)), 9), 14)',
    'alpha071': 'Max(CSRank(DecayLinear(Corr(Rank($close, 3), Rank(Mean($volume,180), 12), 18), 4)), CSRank(DecayLinear(Power(CSRank((($low + $open) - ($vwap + $vwap))), 2), 16)))',
    'alpha072': 'CSRank(DecayLinear(Corr((($high + $low) / 2), Mean($volume,40), 9), 10)) - CSRank(DecayLinear(-Corr(Rank($vwap, 4), Rank($volume, 19), 7), 3))',
    'alpha073': 'Max(CSRank(DecayLinear(-Delta($vwap, 5), 3)), Rank(DecayLinear((((Delta((($open * 0.147155) + ($low * (1 - 0.147155))), 2) / (($open * 0.147155) + ($low * (1 - 0.147155)))) * -1)), 3), 17))',
    'alpha074': '(CSRank(Corr(CSRank((($high * 0.0261661) + ($vwap * (1 - 0.0261661)))), CSRank($volume), 11)) - CSRank(Corr($close, Sum(Mean($volume,30), 37), 15))) / (CSRank(Corr(CSRank((($high * 0.0261661) + ($vwap * (1 - 0.0261661)))), CSRank($volume), 11)) + CSRank(Corr($close, Sum(Mean($volume,30), 37), 15)) + 1e-6)',
    'alpha075': '(CSRank(Corr($vwap, $volume, 4)) - CSRank(Corr(CSRank($low), CSRank(Mean($volume,50)), 12))) * (-Rank($close, 20) + 0.5)',
    'alpha077': 'CSRank(DecayLinear(($vwap - (($high + $low) / 2)), 20)) + CSRank(DecayLinear(-Corr((($high + $low) / 2), Mean($volume,40), 3), 6))',
    'alpha078': 'CSRank(Corr(Mean(($low * 0.352233 + $vwap * 0.647767), 20), Mean(Mean($volume,40), 20), 7)) + CSRank(Corr($vwap, $volume, 6))',
    'alpha081': 'CSRank(Corr($vwap, Sum(Mean($volume,10), 50), 8)) - CSRank(Corr(CSRank($vwap), CSRank($volume), 5))',
    'alpha083': '(CSRank(Ref((($high - $low) / (Sum($close, 5) / 5)), 2)) * CSRank(CSRank($volume)) * ($vwap - $close)) / (($high - $low) / (Sum($close, 5) / 5))',
    'alpha084': 'SignedPower(Rank(($vwap - Max($vwap, 15)), 21), Abs(Delta($close,5)) + 1)',
    'alpha085': 'Power(CSRank(Corr((($high * 0.876703) + ($close * (1 - 0.876703))), Mean($volume,30), 10)), CSRank(Corr(Rank((($high + $low) / 2), 4), Rank($volume, 10), 7))) * Sign(Sum(($close/Ref($close,1)-1), 5))',
    'alpha086': '(CSRank($close - $vwap) - Rank(Corr($close, Sum(Mean($volume,20), 15), 6), 20)) * -1',
    'alpha088': 'Min(CSRank(DecayLinear(-((CSRank($open) + CSRank($low)) - (CSRank($high) + CSRank($close))), 8)), Rank(DecayLinear(-Corr(Rank($close, 8), Rank(Mean($volume,60), 21), 8), 7), 3))',
    'alpha092': 'Max(Rank(DecayLinear((((($high + $low) / 2 + $close) > ($low + $open))), 15), 19), Rank(DecayLinear(Corr(CSRank(-$low), CSRank(Mean($volume,30)), 8), 7), 7))',
    'alpha094': 'CSRank(($vwap - Min($vwap, 12)) * Corr(Rank($vwap, 20), Rank(Mean($volume,60), 4), 18))',
    'alpha095': 'Sign(Rank(Power(CSRank(Corr(Sum((($high + $low) / 2), 20), Sum(Mean($volume,40), 20), 13)), 5), 12) - CSRank($open - Min($open, 12)))',
    'alpha096': 'Max(Rank(DecayLinear(Corr(CSRank($vwap), CSRank($volume), 4), 4), 8), Rank(DecayLinear(TsArgMax(Corr(Rank($close, 7), Rank(Mean($volume,60), 4), 4), 13), 14), 13))',
    'alpha098': '(CSRank(DecayLinear(Corr($vwap, Sum(Mean($volume,5), 26), 5), 7)) + CSRank(DecayLinear(-Rank(TsArgMin(Corr(CSRank($open), CSRank(Mean($volume,15)), 21), 9), 7), 8)))',
    'alpha099': '(Corr(Sum((($high + $low) / 2), 20), Sum(Mean($volume,60), 20), 9) - Corr($low, $volume, 6)) * -1',
    'alpha101': '(($close - $open) / (($high - $low) + 0.001))',
}

assert set(QLIB_EXPRESSIONS) == set(ALPHA101), "translation table must cover every Alpha101 factor"


def get_qlib_expression(name: str) -> str:
    """Return the qlib expression for a factor, e.g. ``get_qlib_expression("alpha001")``."""
    return QLIB_EXPRESSIONS[name.lower()]
