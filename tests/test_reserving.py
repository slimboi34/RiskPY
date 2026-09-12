"""Tests for the claims reserving layer.

The GenIns (Taylor–Ashe 1983) triangle is the anchor: Mack (1993) and England &
Verrall (2002) published enough numbers for it — the age-to-age factors, the
total reserve, the total standard error, the ODP scale parameter — that a typo
anywhere in the chain has nowhere to hide, so those are asserted tightly.

Everything else is an identity, because an identity catches a class of errors
where a single number catches one instance: ``cdf[k] = f[k]···f[-1]·tail``,
``latest · cdf[latest_index] = ultimate``, ``se² = process² + parameter²``,
Bornhuetter–Ferguson collapsing to the chain ladder when the prior *is* the
chain-ladder ultimate, Cape Cod reproducing a hand-computable pool, and the
Python chain ladder agreeing with the compiled ``riskpy.LossTriangle``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskpy import LossTriangle, mc, reserving
from riskpy.reserving import Triangle

NAN = float("nan")

TRI = reserving.genins()
CL = reserving.chain_ladder(TRI)
MACK = reserving.mack_chain_ladder(TRI)

# Mack (1993), Table 1: the volume-weighted age-to-age factors of GenIns.
PUBLISHED_FACTORS = [
    3.4906, 1.7473, 1.4574, 1.1739, 1.1038, 1.0863, 1.0539, 1.0766, 1.0177,
]


def small_triangle() -> Triangle:
    """A 3x3 run-off small enough to check every factor by hand."""
    return Triangle([[100.0, 150.0, 180.0], [200.0, 260.0], [300.0]])


# ---------------------------------------------------------------------------
# Triangle: construction
# ---------------------------------------------------------------------------


def test_ragged_and_nan_padded_construction_agree():
    ragged = Triangle.from_incremental(reserving._GENINS_INCREMENTAL)
    width = max(len(row) for row in reserving._GENINS_INCREMENTAL)
    padded = np.full((len(reserving._GENINS_INCREMENTAL), width), NAN)
    for i, row in enumerate(reserving._GENINS_INCREMENTAL):
        padded[i, : len(row)] = row
    from_array = Triangle(padded, cumulative=False)
    assert np.array_equal(ragged.cumulative, from_array.cumulative, equal_nan=True)
    assert np.array_equal(ragged.observed, from_array.observed)
    # ... and the same again through a cumulative array, which is what genins() is.
    assert np.array_equal(TRI.cumulative, ragged.cumulative, equal_nan=True)
    assert np.array_equal(Triangle(TRI.cumulative).cumulative, TRI.cumulative, equal_nan=True)


def test_incremental_and_cumulative_round_trip():
    inc = TRI.incremental
    assert np.array_equal(Triangle.from_incremental(inc).cumulative, TRI.cumulative, equal_nan=True)
    # The incremental view is the row-wise difference of the cumulative one.
    observed = TRI.observed
    cum = TRI.cumulative
    assert inc[observed][0] == pytest.approx(cum[observed][0])
    running = np.where(observed, np.nancumsum(np.where(observed, inc, 0.0), axis=1), NAN)
    assert np.allclose(running[observed], cum[observed], rtol=1e-12)
    # And the raw incremental data is what came back out.
    assert inc[0, 0] == pytest.approx(reserving._GENINS_INCREMENTAL[0][0])
    assert inc[3, 4] == pytest.approx(reserving._GENINS_INCREMENTAL[3][4])
    assert np.isnan(inc[9, 1])


def test_shape_labels_and_diagonal():
    assert (TRI.n_origin, TRI.n_dev) == (10, 10)
    assert TRI.origin == list(range(10))
    assert list(TRI.latest_index) == list(range(9, -1, -1))
    # The latest diagonal is the row sum of the observed incrementals.
    for i, row in enumerate(reserving._GENINS_INCREMENTAL):
        assert TRI.latest_diagonal[i] == pytest.approx(sum(row), rel=1e-12)
    assert TRI.observed.sum() == 55
    labelled = Triangle(TRI.cumulative, origin=[1988 + k for k in range(10)])
    assert labelled.origin[-1] == 1997
    assert "10 origins" in repr(labelled) and "1997" in repr(labelled)


def test_link_ratios_are_the_raw_material_of_the_factors():
    ratios = TRI.link_ratios()
    assert ratios.shape == (10, 9)
    cum, obs = TRI.cumulative, TRI.observed
    assert ratios[0, 0] == pytest.approx(cum[0, 1] / cum[0, 0], rel=1e-14)
    assert np.isnan(ratios[9, 0])  # origin 9 has no second column yet
    # The volume-weighted factor is the C[:, k]-weighted mean of the link ratios.
    for k in range(9):
        rows = obs[:, k] & obs[:, k + 1]
        weights = cum[rows, k]
        weighted = float((weights * ratios[rows, k]).sum() / weights.sum())
        assert CL.factors[k] == pytest.approx(weighted, rel=1e-12)


def test_construction_from_a_pandas_frame_and_back():
    pandas = pytest.importorskip("pandas")
    frame = TRI.to_frame()
    assert isinstance(frame, pandas.DataFrame)
    assert frame.shape == (10, 10)
    assert frame.index.name == "origin" and frame.columns.name == "dev"
    assert np.array_equal(Triangle(frame).cumulative, TRI.cumulative, equal_nan=True)


def test_cumulative_property_is_a_copy():
    values = TRI.cumulative
    values[0, 0] = -1.0
    assert TRI.cumulative[0, 0] > 0.0


def test_decreasing_rows_need_an_opt_in():
    with pytest.raises(ValueError, match="decrease at origin row 0"):
        Triangle([[100.0, 90.0]])
    incurred = Triangle([[100.0, 90.0]], allow_decreasing=True)
    assert incurred.cumulative[0, 1] == pytest.approx(90.0)


def test_triangle_rejects_bad_data():
    with pytest.raises(ValueError, match="must be 2-D"):
        Triangle(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="list of rows"):
        Triangle(5)
    with pytest.raises(ValueError, match="data is empty"):
        Triangle([])
    with pytest.raises(ValueError, match="at least one origin and one column"):
        Triangle(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="non-numeric"):
        Triangle([["a", "b"]])
    with pytest.raises(ValueError, match="infinite"):
        Triangle([[1.0, float("inf")]])
    with pytest.raises(ValueError, match="no observations at all"):
        Triangle([[1.0, 2.0], [NAN, NAN]])
    with pytest.raises(ValueError, match="no holes"):
        Triangle([[1.0, NAN, 3.0]])
    with pytest.raises(ValueError, match="no observations in any"):
        Triangle([[1.0, NAN], [2.0, NAN]])
    with pytest.raises(ValueError, match="is negative"):
        Triangle([[-1.0, 2.0]])
    with pytest.raises(ValueError, match="2 labels but the triangle has 1 rows"):
        Triangle([[1.0, 2.0]], origin=[1, 2])
    with pytest.raises(ValueError, match="must be unique"):
        Triangle([[1.0, 2.0], [3.0, NAN]], origin=[7, 7])


# ---------------------------------------------------------------------------
# Factors, cumulative factors, patterns and tails
# ---------------------------------------------------------------------------


def test_published_genins_age_to_age_factors():
    assert len(CL.factors) == 9
    for k, published in enumerate(PUBLISHED_FACTORS):
        assert round(float(CL.factors[k]), 4) == published


def test_cdf_is_the_running_product_of_the_factors():
    for tail in (1.0, 1.05):
        cdf = reserving.cdf_from_factors(CL.factors, tail)
        assert len(cdf) == len(CL.factors) + 1
        assert cdf[-1] == pytest.approx(tail, rel=1e-15)
        for k in range(len(CL.factors)):
            assert cdf[k] == pytest.approx(
                float(np.prod(CL.factors[k:])) * tail, rel=1e-12
            )
    assert np.allclose(reserving.cdf_from_factors(CL.factors), CL.cdf, rtol=1e-14)


def test_development_pattern_is_the_reciprocal_cdf():
    pattern = reserving.development_pattern(CL.cdf)
    assert np.allclose(pattern * CL.cdf, 1.0, rtol=1e-15)
    assert pattern[-1] == pytest.approx(1.0)          # fully developed at the end
    assert np.all(np.diff(pattern) > 0.0)             # and monotone on the way there
    # It is exactly the percentage-developed column of the chain-ladder summary.
    assert np.allclose(pattern[TRI.latest_index], CL.percent_developed, rtol=1e-12)


def test_fit_tail_recovers_a_known_exponential_decay():
    # f_k = 1 + a·e^{b k} for k = 1..9 is exactly what the exponential fit assumes,
    # so the regression must recover (ln a, b) and the tail must be the analytic
    # product of the next n_extrapolate factors.
    a, b, n = 0.8, -0.55, 50
    factors = [1.0 + a * math.exp(b * k) for k in range(1, 10)]
    expected = math.prod(1.0 + a * math.exp(b * k) for k in range(10, 10 + n))
    assert reserving.fit_tail(factors, n_extrapolate=n) == pytest.approx(expected, rel=1e-12)
    # 50 periods really is infinity for an exponential: the remaining factors are
    # within 1e-13 of 1, so doubling the horizon does not move the answer.
    assert reserving.fit_tail(factors, n_extrapolate=200) == pytest.approx(expected, rel=1e-12)

    stats = pytest.importorskip("scipy.stats")
    fit = stats.linregress(np.arange(1.0, 10.0), np.log(np.asarray(factors) - 1.0))
    assert fit.slope == pytest.approx(b, rel=1e-10)
    assert fit.intercept == pytest.approx(math.log(a), rel=1e-10)


def test_fit_tail_inverse_power_is_heavier_and_horizon_dependent():
    factors = list(CL.factors)
    exponential = reserving.fit_tail(factors, method="exponential")
    inverse = reserving.fit_tail(factors, method="inverse_power")
    assert 1.0 < exponential < inverse
    # The exponential product has converged; the inverse-power one has not.
    assert reserving.fit_tail(factors, n_extrapolate=200) == pytest.approx(exponential, rel=1e-12)
    assert reserving.fit_tail(factors, method="inverse_power", n_extrapolate=200) > inverse


def test_factor_and_tail_helpers_reject_bad_input():
    with pytest.raises(ValueError, match="1-D sequence"):
        reserving.cdf_from_factors([[1.0, 2.0]])
    with pytest.raises(ValueError, match="finite and positive"):
        reserving.cdf_from_factors([1.0, -2.0])
    with pytest.raises(ValueError, match="finite positive factor"):
        reserving.cdf_from_factors([1.5], tail=0.0)
    with pytest.raises(ValueError, match="finite positive factor"):
        reserving.cdf_from_factors([1.5], tail=NAN)
    with pytest.raises(ValueError, match="finite positive factors"):
        reserving.development_pattern([1.0, 0.0])
    with pytest.raises(ValueError, match="exponential"):
        reserving.fit_tail([1.5, 1.2], method="linear")
    with pytest.raises(ValueError, match="n_extrapolate"):
        reserving.fit_tail([1.5, 1.2], n_extrapolate=0)
    with pytest.raises(ValueError, match="finite numbers"):
        reserving.fit_tail([1.5, NAN])
    with pytest.raises(ValueError, match="at least two factors above 1"):
        reserving.fit_tail([1.5, 1.0, 0.9])
    with pytest.raises(ValueError, match="not decaying"):
        reserving.fit_tail([1.1, 1.2, 1.3])


# ---------------------------------------------------------------------------
# Chain ladder
# ---------------------------------------------------------------------------


def test_published_genins_chain_ladder_reserve():
    assert CL.total_reserve == pytest.approx(18_680_856.0, rel=1e-6)
    assert CL.reserve[-1] == pytest.approx(4_625_811.0, rel=1e-6)
    assert CL.ultimate[-1] == pytest.approx(4_969_825.0, rel=1e-6)
    assert CL.reserve[0] == pytest.approx(0.0, abs=1e-9)   # oldest origin is run off
    assert CL.total_ultimate == pytest.approx(CL.total_latest + CL.total_reserve, rel=1e-12)


def test_ultimate_is_the_latest_diagonal_grossed_up_by_the_cdf():
    assert np.allclose(CL.latest * CL.cdf[TRI.latest_index], CL.ultimate, rtol=1e-12)
    assert np.allclose(CL.reserve, CL.ultimate - CL.latest, rtol=1e-12)
    assert np.allclose(CL.percent_developed, CL.latest / CL.ultimate, rtol=1e-12)
    # full_triangle carries the tail nowhere: it stops at the last column.
    assert np.allclose(CL.full_triangle[:, -1] * CL.tail, CL.ultimate, rtol=1e-14)
    observed = TRI.observed
    assert np.allclose(CL.full_triangle[observed], TRI.cumulative[observed], rtol=1e-14)


def test_a_tail_scales_every_ultimate_by_exactly_the_tail():
    tailed = reserving.chain_ladder(TRI, tail=1.05)
    assert np.allclose(tailed.ultimate, CL.ultimate * 1.05, rtol=1e-12)
    assert np.allclose(tailed.reserve, CL.reserve + 0.05 * CL.ultimate, rtol=1e-12)
    assert np.allclose(tailed.factors, CL.factors, rtol=1e-14)   # the tail is not a factor
    assert tailed.cdf[-1] == pytest.approx(1.05)


def test_a_fully_developed_square_has_no_reserve():
    square = Triangle([[100.0, 150.0, 180.0], [120.0, 170.0, 200.0], [140.0, 190.0, 210.0]])
    result = reserving.chain_ladder(square)
    assert np.allclose(result.reserve, 0.0, atol=1e-12)
    assert result.total_reserve == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(result.ultimate, result.latest, rtol=1e-14)
    assert np.allclose(result.percent_developed, 1.0, rtol=1e-14)
    # Nothing is projected, so Mack has nothing to be uncertain about either.
    mack = reserving.mack_chain_ladder(square)
    assert mack.total_se == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(mack.se, 0.0, atol=1e-9)


def test_averaging_rules_are_hand_computable():
    tri = Triangle([[100.0, 150.0, 180.0], [200.0, 260.0]])
    volume = reserving.chain_ladder(tri, average="volume").factors
    simple = reserving.chain_ladder(tri, average="simple").factors
    regression = reserving.chain_ladder(tri, average="regression").factors
    assert volume[0] == pytest.approx(410.0 / 300.0, rel=1e-14)
    assert simple[0] == pytest.approx((1.5 + 1.3) / 2.0, rel=1e-14)
    assert regression[0] == pytest.approx(67_000.0 / 50_000.0, rel=1e-14)
    # Only one origin sees the second step, so every rule agrees there.
    for rule in (volume, simple, regression):
        assert rule[1] == pytest.approx(180.0 / 150.0, rel=1e-14)
    assert "simple" in reserving.chain_ladder(tri, average="simple").method


def test_n_periods_uses_only_the_most_recent_link_ratios():
    tri = small_triangle()
    everything = reserving.chain_ladder(tri).factors
    recent = reserving.chain_ladder(tri, n_periods=1).factors
    assert everything[0] == pytest.approx(410.0 / 300.0, rel=1e-14)
    assert recent[0] == pytest.approx(260.0 / 200.0, rel=1e-14)
    assert recent[1] == pytest.approx(180.0 / 150.0, rel=1e-14)
    # Asking for more periods than exist is the same as asking for all of them.
    assert np.allclose(reserving.chain_ladder(tri, n_periods=99).factors, everything, rtol=1e-14)


def test_chain_ladder_matches_the_compiled_loss_triangle():
    compiled = LossTriangle()
    cum, obs = TRI.cumulative, TRI.observed
    for i in range(TRI.n_origin):
        compiled.add_origin_year(i, [float(v) for v in cum[i, obs[i]]])
    assert np.allclose(np.asarray(compiled.get_ultimate_losses()), CL.ultimate, rtol=1e-6)
    assert np.allclose(np.asarray(compiled.get_development_factors()), CL.factors, rtol=1e-6)
    assert np.allclose(np.asarray(compiled.get_ibnr_reserves()), CL.reserve, rtol=1e-6, atol=1e-6)


def test_chain_ladder_accepts_raw_data_and_prints():
    result = reserving.chain_ladder([[100.0, 150.0, 180.0], [200.0, 260.0], [300.0]])
    assert result.total_reserve > 0.0
    text = result.summary()
    assert "origin" in text and "total" in text and "factors" in text
    assert "total reserve" in repr(result)
    pandas = pytest.importorskip("pandas")
    frame = result.to_frame()
    assert isinstance(frame, pandas.DataFrame) and list(frame.index) == [0, 1, 2]
    assert "ultimate" in frame.columns


def test_chain_ladder_rejects_bad_arguments():
    with pytest.raises(ValueError, match="volume"):
        reserving.chain_ladder(TRI, average="median")
    with pytest.raises(ValueError, match="n_periods"):
        reserving.chain_ladder(TRI, n_periods=0)
    with pytest.raises(ValueError, match="n_periods"):
        reserving.chain_ladder(TRI, n_periods=2.5)
    with pytest.raises(ValueError, match="finite positive factor"):
        reserving.chain_ladder(TRI, tail=-1.0)
    zeros = Triangle([[0.0, 0.0, 5.0], [0.0, 0.0], [0.0]])
    with pytest.raises(ValueError, match="sums to zero"):
        reserving.chain_ladder(zeros, average="volume")
    with pytest.raises(ValueError, match="contains a zero"):
        reserving.chain_ladder(zeros, average="simple")
    with pytest.raises(ValueError, match="contains a zero"):
        reserving.chain_ladder(zeros, average="regression")


# ---------------------------------------------------------------------------
# Mack
# ---------------------------------------------------------------------------


def test_published_genins_mack_standard_errors():
    assert MACK.total_se == pytest.approx(2_447_095.0, rel=1e-4)
    assert MACK.se[-1] == pytest.approx(1_363_155.0, rel=1e-4)
    assert MACK.total_reserve == pytest.approx(CL.total_reserve, rel=1e-12)
    assert np.allclose(MACK.factors, CL.factors, rtol=1e-14)
    assert MACK.total_cv == pytest.approx(MACK.total_se / MACK.total_reserve, rel=1e-14)


def test_mack_variance_decomposition_is_pythagorean():
    assert np.allclose(MACK.se ** 2, MACK.process_se ** 2 + MACK.parameter_se ** 2, rtol=1e-12)
    assert MACK.total_se ** 2 == pytest.approx(
        MACK.total_process_se ** 2 + MACK.total_parameter_se ** 2, rel=1e-12
    )
    # Origins share the estimated factors, so the total is worse than independence.
    independent = math.sqrt(float((MACK.se ** 2).sum()))
    assert independent < MACK.total_se
    assert MACK.total_process_se == pytest.approx(independent := math.sqrt(
        float((MACK.process_se ** 2).sum())), rel=1e-12)
    # A run-off origin has no reserve and therefore no error at all.
    assert MACK.se[0] == 0.0 and MACK.reserve[0] == 0.0
    assert math.isnan(float(MACK.cv[0]))
    assert MACK.cv[-1] == pytest.approx(MACK.se[-1] / MACK.reserve[-1], rel=1e-14)
    # sigma is a variance; the printed sigma is its root.
    assert len(MACK.sigma) == 9
    assert math.sqrt(float(MACK.sigma[0])) == pytest.approx(400.36, rel=1e-3)


def test_reserve_percentile_is_a_moment_matched_lognormal():
    stats = pytest.importorskip("scipy.stats")
    mean, se = MACK.total_reserve, MACK.total_se
    sigma_sq = math.log(1.0 + (se / mean) ** 2)
    reference = stats.lognorm(s=math.sqrt(sigma_sq), scale=math.exp(math.log(mean) - 0.5 * sigma_sq))
    for level in (0.5, 0.75, 0.95, 0.995):
        assert MACK.reserve_percentile(level) == pytest.approx(reference.ppf(level), rel=1e-10)
    assert reference.mean() == pytest.approx(mean, rel=1e-12)
    assert reference.std() == pytest.approx(se, rel=1e-12)
    # The median of a lognormal sits below its mean, and the levels are ordered.
    assert MACK.reserve_percentile(0.5) < mean < MACK.reserve_percentile(0.75)
    levels = [MACK.reserve_percentile(p) for p in (0.5, 0.75, 0.9, 0.995)]
    assert levels == sorted(levels)
    # Per origin as well, and a run-off origin has no spread to speak of.
    assert MACK.reserve_percentile(0.995, origin=9) > MACK.reserve[-1]
    assert MACK.reserve_percentile(0.995, origin=0) == 0.0


def test_mack_with_a_tail_adds_one_more_uncertain_step():
    tailed = reserving.mack_chain_ladder(TRI, tail=1.05)
    assert tailed.total_reserve > MACK.total_reserve
    assert tailed.total_se > MACK.total_se
    assert tailed.tail_sigma > 0.0 and tailed.tail_se > 0.0
    assert np.allclose(tailed.se ** 2, tailed.process_se ** 2 + tailed.parameter_se ** 2, rtol=1e-12)
    # The oldest origin is fully run off in the triangle but still faces the tail.
    assert tailed.se[0] > 0.0
    # A tail known without error adds reserve but less uncertainty than an estimated one.
    certain = reserving.mack_chain_ladder(TRI, tail=1.05, tail_sigma=0.0, tail_se=0.0)
    assert certain.total_reserve == pytest.approx(tailed.total_reserve, rel=1e-12)
    assert certain.total_se < tailed.total_se
    assert certain.se[0] == 0.0


def test_mack_rejects_bad_arguments():
    with pytest.raises(ValueError, match="two development periods"):
        reserving.mack_chain_ladder(Triangle([[1.0], [2.0]]))
    with pytest.raises(ValueError, match="tail_sigma"):
        reserving.mack_chain_ladder(TRI, tail=1.05, tail_sigma=-1.0)
    with pytest.raises(ValueError, match="tail_se"):
        reserving.mack_chain_ladder(TRI, tail=1.05, tail_se=NAN)
    with pytest.raises(ValueError, match="cannot be estimated"):
        reserving.mack_chain_ladder(Triangle([[1.0, 2.0], [3.0]]))
    # A triangle with no dispersion at all has no sigma trend to extrapolate.
    rigid = Triangle([[100.0, 200.0, 300.0], [100.0, 200.0], [100.0]])
    with pytest.raises(ValueError, match="cannot extrapolate σ"):
        reserving.mack_chain_ladder(rigid, tail=1.05)
    with pytest.raises(ValueError, match="standard error"):
        reserving.mack_chain_ladder(rigid, tail=1.05, tail_sigma=1.0)
    with pytest.raises(ValueError, match="probability in"):
        MACK.reserve_percentile(1.5)
    with pytest.raises(ValueError, match="not one of the triangle"):
        MACK.reserve_percentile(0.95, origin=99)
    # A shrinking incurred triangle gives a negative reserve, which no lognormal fits.
    shrinking = Triangle(
        [[100.0, 90.0, 88.0], [100.0, 95.0], [100.0]], allow_decreasing=True
    )
    with pytest.raises(ValueError, match="needs a positive reserve"):
        reserving.mack_chain_ladder(shrinking).reserve_percentile(0.95)


def test_mack_summary_prints_its_extra_columns():
    text = MACK.summary()
    assert "se" in text and "cv" in text and "sigma" in text
    assert "process" in text and "parameter" in text


# ---------------------------------------------------------------------------
# The expected-loss-ratio family
# ---------------------------------------------------------------------------


PREMIUM = np.full(10, 10_000_000.0)


def test_bornhuetter_ferguson_collapses_to_the_chain_ladder():
    # If the a-priori ultimate *is* the chain-ladder ultimate, BF reproduces it —
    # every origin, not just the total. That is the identity the method is famous
    # for failing to be independent of.
    bf = reserving.bornhuetter_ferguson(TRI, PREMIUM, CL.ultimate / PREMIUM)
    assert np.allclose(bf.ultimate, CL.ultimate, rtol=1e-10)
    assert np.allclose(bf.reserve, CL.reserve, rtol=1e-10, atol=1e-6)
    assert bf.total_reserve == pytest.approx(CL.total_reserve, rel=1e-10)
    # expected_claims is the same method with the prior handed over directly.
    ec = reserving.expected_claims(TRI, CL.ultimate)
    assert np.allclose(ec.ultimate, CL.ultimate, rtol=1e-10)
    assert np.allclose(ec.reserve, bf.reserve, rtol=1e-10, atol=1e-6)


def test_bornhuetter_ferguson_reserve_is_linear_in_the_prior():
    half = reserving.bornhuetter_ferguson(TRI, PREMIUM, 0.30)
    full = reserving.bornhuetter_ferguson(TRI, PREMIUM, 0.60)
    assert np.allclose(2.0 * half.reserve, full.reserve, rtol=1e-12)
    # The reserve never touches the claims paid so far; the ultimate does.
    assert np.allclose(full.ultimate - full.latest, full.reserve, rtol=1e-12)
    # A run-off origin has nothing left to come, whatever the prior says.
    assert full.reserve[0] == pytest.approx(0.0, abs=1e-9)
    # Reserve = premium × ELR × (1 - 1/cdf), by hand for the youngest origin.
    unreported = 1.0 - 1.0 / CL.cdf[0]
    assert full.reserve[-1] == pytest.approx(10_000_000.0 * 0.60 * unreported, rel=1e-12)


def test_bornhuetter_ferguson_takes_a_supplied_pattern_and_ignores_tail():
    given = reserving.bornhuetter_ferguson(
        TRI, PREMIUM, 0.6, factors=reserving.cdf_from_factors(CL.factors, 1.2)
    )
    same = reserving.bornhuetter_ferguson(TRI, PREMIUM, 0.6, factors=given.__class__ and
                                          reserving.cdf_from_factors(CL.factors, 1.2), tail=9.0)
    assert np.allclose(given.reserve, same.reserve, rtol=1e-14)
    # A heavier pattern leaves more to come than the default no-tail one.
    default = reserving.bornhuetter_ferguson(TRI, PREMIUM, 0.6)
    assert given.total_reserve > default.total_reserve
    # A scalar ELR and a per-origin one containing the same number agree.
    per_origin = reserving.bornhuetter_ferguson(TRI, PREMIUM, [0.6] * 10)
    assert np.allclose(per_origin.reserve, default.reserve, rtol=1e-14)


def test_cape_cod_single_origin_is_hand_computable():
    # One origin, 100 -> 150, and a pattern saying a further 25% is to come.
    # Used-up premium = 1000 / 1.25 = 800, so ELR = 150 / 800 = 0.1875, and the
    # reserve is 1000 x 0.1875 x (1 - 1/1.25) = 37.5. Cape Cod on a single origin
    # always reproduces the chain ladder: ultimate = latest x cdf.
    tri = Triangle([[100.0, 150.0]])
    cc = reserving.cape_cod(tri, [1000.0], factors=[2.0, 1.25])
    assert cc.extra["expected_loss_ratio"][0] == pytest.approx(0.1875, rel=1e-14)
    assert cc.reserve[0] == pytest.approx(37.5, rel=1e-14)
    assert cc.ultimate[0] == pytest.approx(150.0 * 1.25, rel=1e-14)


def test_cape_cod_pools_two_origins_by_used_up_premium():
    # cdf = [1.5, 1.0]; used-up premium = [1000, 1000/1.5]; latest = [150, 120].
    # ELR = 270 / (1000 + 2000/3) = 0.162, and only the young origin has a reserve.
    tri = Triangle([[100.0, 150.0], [120.0]])
    cc = reserving.cape_cod(tri, [1000.0, 1000.0])
    assert np.allclose(cc.extra["expected_loss_ratio"], 0.162, rtol=1e-14)
    assert cc.reserve[0] == pytest.approx(0.0, abs=1e-12)
    assert cc.reserve[1] == pytest.approx(1000.0 * 0.162 * (1.0 - 1.0 / 1.5), rel=1e-14)
    assert cc.total_ultimate == pytest.approx(150.0 + 174.0, rel=1e-14)


def test_cape_cod_pooled_loss_ratio_identity_and_decay():
    cc = reserving.cape_cod(TRI, PREMIUM)
    elr = cc.extra["expected_loss_ratio"]
    assert np.allclose(elr, elr[0], rtol=1e-14)       # decay = 1 pools into one ratio
    used_up = float((PREMIUM / CL.cdf[TRI.latest_index]).sum())
    assert float(elr[0]) * used_up == pytest.approx(TRI.latest_diagonal.sum(), rel=1e-12)
    # A decayed Cape Cod gives every origin its own ratio, between the pooled one
    # and its own chain-ladder loss ratio.
    gluck = reserving.cape_cod(TRI, PREMIUM, decay=0.75)
    assert not np.allclose(gluck.extra["expected_loss_ratio"], elr[0], rtol=1e-6)
    assert gluck.extra["decay"] == 0.75
    assert np.all(gluck.reserve >= 0.0)
    text = cc.summary()
    assert "expected_loss_ratio" in text and "decay" in text


def test_elr_methods_reject_bad_arguments():
    with pytest.raises(ValueError, match="looks like age-to-age factors"):
        reserving.bornhuetter_ferguson(TRI, PREMIUM, 0.6, factors=CL.factors)
    with pytest.raises(ValueError, match="cumulative-to-ultimate"):
        reserving.bornhuetter_ferguson(TRI, PREMIUM, 0.6, factors=[1.0] * 3)
    with pytest.raises(ValueError, match="finite and positive"):
        reserving.bornhuetter_ferguson(TRI, PREMIUM, 0.6, factors=[-1.0] * 10)
    with pytest.raises(ValueError, match="one value per origin"):
        reserving.bornhuetter_ferguson(TRI, [1e7] * 3, 0.6)
    with pytest.raises(ValueError, match="premiums must be finite"):
        reserving.bornhuetter_ferguson(TRI, [NAN] * 10, 0.6)
    with pytest.raises(ValueError, match="must be non-negative"):
        reserving.bornhuetter_ferguson(TRI, PREMIUM, -0.6)
    with pytest.raises(ValueError, match="finite positive factor"):
        reserving.bornhuetter_ferguson(TRI, PREMIUM, 0.6, tail=0.0)
    with pytest.raises(ValueError, match="decay must be in"):
        reserving.cape_cod(TRI, PREMIUM, decay=0.0)
    with pytest.raises(ValueError, match="decay must be in"):
        reserving.cape_cod(TRI, PREMIUM, decay=1.5)
    with pytest.raises(ValueError, match="positive used-up premium"):
        reserving.cape_cod(TRI, [0.0] * 10)
    with pytest.raises(ValueError, match="ultimates_prior"):
        reserving.expected_claims(TRI, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


BOOT_N = 200
BOOT = reserving.bootstrap_chain_ladder(TRI, n=BOOT_N, seed=20240607)


def test_bootstrap_mean_sits_just_above_the_chain_ladder_reserve():
    # England & Verrall: the point estimate stays the chain ladder's; the bootstrap
    # mean is above it by the resampling bias. 200 samples is a coarse mean, so the
    # tolerance is 5% — enough to catch a scale error, not enough to be flaky.
    assert BOOT.total_mean == pytest.approx(CL.total_reserve, rel=0.05)
    assert np.allclose(BOOT.mean, CL.reserve, rtol=0.35, atol=1.0)
    assert np.allclose(BOOT.cl_reserve, CL.reserve, rtol=1e-14)
    assert BOOT.reserves.shape == (BOOT_N, 10)
    assert BOOT.total.shape == (BOOT_N,)
    assert np.allclose(BOOT.total, BOOT.reserves.sum(axis=1), rtol=1e-12)
    assert BOOT.mean[0] == 0.0 and BOOT.se[0] == 0.0   # the run-off origin never moves
    assert BOOT.total_se > 0.0


def test_bootstrap_scale_parameter_matches_england_and_verrall():
    # phi = 52,601 for GenIns (England & Verrall 2002, Table 3).
    assert BOOT.phi == pytest.approx(52_601.0, rel=1e-4)
    assert BOOT.n == BOOT_N and BOOT.seed == 20240607 and BOOT.process == "odp"


def test_bootstrap_is_reproducible_and_seed_sensitive():
    again = reserving.bootstrap_chain_ladder(TRI, n=BOOT_N, seed=20240607)
    assert np.array_equal(again.total, BOOT.total)
    other = reserving.bootstrap_chain_ladder(TRI, n=BOOT_N, seed=1)
    assert not np.array_equal(other.total, BOOT.total)
    assert other.total_mean == pytest.approx(CL.total_reserve, rel=0.05)


def test_bootstrap_process_error_adds_spread_on_top_of_parameter_error():
    parameter_only = reserving.bootstrap_chain_ladder(TRI, n=BOOT_N, seed=20240607, process="none")
    gamma = reserving.bootstrap_chain_ladder(TRI, n=BOOT_N, seed=20240607, process="gamma")
    assert parameter_only.total_se < BOOT.total_se
    assert gamma.total_se > parameter_only.total_se
    # Process error is noise about the same projection, so the means barely move.
    assert parameter_only.total_mean == pytest.approx(BOOT.total_mean, rel=0.05)


def test_bootstrap_result_is_a_monte_carlo_result():
    result = BOOT.result()
    assert isinstance(result, mc.Result)
    assert result.trials == BOOT_N and result.seed == 20240607
    assert result.inputs == {}
    assert result.label == "total reserve"
    assert np.array_equal(result.values, BOOT.total)
    for level in (0.75, 0.95, 0.995):
        assert result.var(level) == pytest.approx(BOOT.percentile(level), rel=1e-12)
        assert result.tvar(level) >= result.var(level)
    assert result.mean == pytest.approx(BOOT.total_mean, rel=1e-12)
    assert result.var(0.5) < result.var(0.95) < result.var(0.995)
    # Mutating the Result must not reach back into the bootstrap.
    result.values[0] = -1.0
    assert BOOT.total[0] >= 0.0


def test_bootstrap_percentiles_and_printing():
    assert BOOT.percentile(0.5) == pytest.approx(float(np.percentile(BOOT.total, 50.0)), rel=1e-12)
    assert BOOT.percentile(0.95, origin=9) == pytest.approx(
        float(np.percentile(BOOT.reserves[:, 9], 95.0)), rel=1e-12
    )
    assert BOOT.total_cv == pytest.approx(BOOT.total_se / BOOT.total_mean, rel=1e-14)
    text = BOOT.summary()
    assert "Bootstrap chain ladder" in text and "phi" in text and "99.5%" in text
    assert "s.e." in repr(BOOT)
    pandas = pytest.importorskip("pandas")
    frame = BOOT.to_frame()
    assert isinstance(frame, pandas.DataFrame) and "cl_reserve" in frame.columns


def test_bootstrap_rejects_bad_arguments():
    with pytest.raises(ValueError, match="n must be an integer"):
        reserving.bootstrap_chain_ladder(TRI, n=1)
    with pytest.raises(ValueError, match="process must be"):
        reserving.bootstrap_chain_ladder(TRI, n=10, process="poisson")
    with pytest.raises(ValueError, match="more cells than parameters"):
        reserving.bootstrap_chain_ladder(Triangle([[1.0, 2.0], [3.0]]), n=10)
    shrinking = Triangle(
        [[100.0, 90.0, 95.0], [100.0, 95.0], [100.0]], allow_decreasing=True
    )
    with pytest.raises(ValueError, match="not an ODP triangle"):
        reserving.bootstrap_chain_ladder(shrinking, n=10)
    with pytest.raises(ValueError, match="probability in"):
        BOOT.percentile(0.0)
    with pytest.raises(ValueError, match="not one of the triangle"):
        BOOT.percentile(0.5, origin="1998")


# ---------------------------------------------------------------------------
# Verification hook and exports
# ---------------------------------------------------------------------------


def test_verification_checks_pass():
    checks = reserving._verification_checks()
    assert len(checks) >= 4
    for name, value, reference, tolerance in checks:
        assert isinstance(name, str) and name
        assert math.isfinite(value) and math.isfinite(reference)
        assert abs(value - reference) <= tolerance, name


def test_all_exports_exist():
    for name in reserving.__all__:
        assert hasattr(reserving, name), name
    assert not any(name.startswith("_") for name in reserving.__all__)
