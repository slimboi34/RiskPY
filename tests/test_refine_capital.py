"""0.8.0 refinements to riskpy.capital and riskpy._special: the Student t at extreme degrees of
freedom, the Frank, Clayton and Gumbel copulas at extreme theta, the t copula's memory, the
Euler VaR residual and input validation.

Each check has a reference the library has no hand in: mpmath at 60 or more digits, a
closed form worked in the test, or a property every valid sample has (uniform marginals,
Kendall's tau within four standard errors of its closed form, a capital figure below the
comonotone bound).
"""
import math
import time
import warnings

import pytest

from riskpy import _special as sp
from riskpy import capital
from riskpy.capital import (
    ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula, StudentTCopula, aggregate_capital,
)
from riskpy.errors import MissingDependencyError, RiskPyError


def _np():
    return pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# An mpmath oracle for the Student t


def _dps(df):
    return max(60, int(math.log10(df)) + 60) if df > 1 else 60


def _t_cdf_mp(mp, t, df):
    t, df = mp.mpf(t), mp.mpf(df)
    if t == 0:
        return mp.mpf(0.5)
    t2 = t * t
    inside = mp.betainc(mp.mpf(0.5), df / 2, 0, t2 / (df + t2), regularized=True)        # P(|T| < |t|)
    if inside > 0.5:
        tail = mp.betainc(df / 2, mp.mpf(0.5), 0, df / (df + t2), regularized=True) / 2
    else:
        tail = (1 - inside) / 2
    return 1 - tail if t > 0 else tail


def _t_ppf_mp(mp, p, df, start):
    """Newton's method in high precision from ``start`` (the double answer under test)."""
    with mp.workdps(_dps(df)):
        t, target = mp.mpf(start), mp.mpf(p)
        df_ = mp.mpf(df)
        log_norm = mp.loggamma((df_ + 1) / 2) - mp.loggamma(df_ / 2) - mp.log(df_ * mp.pi) / 2
        for _ in range(60):
            pdf = mp.exp(log_norm - (df_ + 1) / 2 * mp.log1p(t * t / df_))
            step = (_t_cdf_mp(mp, t, df) - target) / pdf
            t -= step
            if abs(step) <= abs(t) * mp.mpf(10) ** -40:
                break
        return +t


@pytest.mark.parametrize("df", [1e4, 1e8, 1e12, 1e13, 1e15, 1e16, 1e20, 1e50, 1e300])
def test_t_ppf_at_large_df_matches_mpmath(df):
    # 0.7.0 solved the centre on an incomplete beta whose argument df/(df + t²) had rounded
    # to 1: wrong from df ~ 1e12, and from df ~ 1e23 the bracket search never returned.
    mp = pytest.importorskip("mpmath")
    for p in (0.4, 0.6, 0.5 + 1e-12, 0.26, 0.74, 0.1, 0.995, 1e-10, 1e-100):
        got = sp.t_ppf(p, df)
        ref = _t_ppf_mp(mp, p, df, got)
        assert abs(float((got - ref) / ref)) < 1e-14, (df, p, got, ref)


@pytest.mark.parametrize("df", [0.5, 3.0, 50.0, 9999.0])
def test_t_ppf_central_band_matches_mpmath(df):
    mp = pytest.importorskip("mpmath")
    for p in (0.4, 0.6, 0.5 + 1e-12, 0.5 - 3e-7, 0.26, 0.74):
        got = sp.t_ppf(p, df)
        ref = _t_ppf_mp(mp, p, df, got)
        assert abs(float((got - ref) / ref)) < 1e-13, (df, p, got, ref)


def test_t_quantile_and_t_copula_answer_at_once_for_any_df():
    start = time.perf_counter()
    for df in (1e17, 1e23, 1e100, 1e300, 1.7e308):
        for p in (0.6, 0.4, 0.995, 1e-300):
            # by df = 1e17 the t quantile is the normal one to within rounding
            assert sp.t_ppf(p, df) == pytest.approx(sp.norm_ppf(p), rel=1e-12, abs=1e-300)
    from riskpy import tools

    gauss = tools.call("copula_dependence", {"family": "gaussian", "rho": 0.5, "level": 0.6})["joint_exceedance"]
    for df in (1e8, 1e16, 1e23, 1e300):
        out = tools.call("copula_dependence", {"family": "t", "rho": 0.5, "df": df, "level": 0.6})
        assert out["joint_exceedance"] == pytest.approx(gauss, rel=1e-6)
    assert time.perf_counter() - start < 10.0


def test_a_t_quantile_beyond_double_precision_is_infinite_not_a_math_error():
    # 0.7.0 raised a bare "math domain error" here from log(0) in the central branch
    assert sp.t_ppf(0.6, 1e-20) == math.inf and sp.t_ppf(0.4, 1e-20) == -math.inf
    assert sp.t_ppf(0.6, 1e-300) == math.inf
    for level in (0.4, 0.6, 0.995):
        with pytest.raises(RiskPyError) as exc:
            StudentTCopula(0.5, df=1e-20).joint_exceedance(level)
        assert (exc.value.code, exc.value.param) == ("RP304", "df")


def test_t_cdf_scalar_and_array_match_mpmath():
    mp = pytest.importorskip("mpmath")
    np = _np()
    ts = [0.0, 1e-200, 1e-12, 1e-3, 0.3, -0.7, 1.5, -4.0, 12.0, -60.0, math.inf, -math.inf]
    for df in (1e-3, 0.5, 3.0, 50.0, 999.0, 9999.0, 1e4, 1e6, 1e12, 1e17, 1e300):
        vec = sp.t_cdf_vec(np.array(ts), df)
        for t, v in zip(ts, vec):
            scalar = sp.t_cdf(t, df)
            if math.isinf(t):
                ref = 1.0 if t > 0 else 0.0
            else:
                with mp.workdps(_dps(df)):
                    ref = float(_t_cdf_mp(mp, t, df))
            scale = max(min(ref, 1.0 - ref), 1e-300)          # what a quantile needs: relative in the tail
            assert abs(scalar - ref) / scale < 1e-12, (df, t, scalar, ref)
            assert abs(v - ref) / scale < 1e-12, (df, t, v, ref)
    assert np.isnan(sp.t_cdf_vec(np.array([np.nan]), 4.0)[0])


def test_the_incomplete_beta_continued_fraction_has_a_fixed_cap_and_a_coded_error():
    np = _np()
    for a in (1e12, 1e14):
        x = (a + 1.0) / (2.0 * a + 2.0)          # the switch point, where the fraction is slowest
        start = time.perf_counter()
        with pytest.raises(RiskPyError) as exc:
            sp.betainc(a, a, x)
        assert exc.value.code == "RP304"
        with pytest.raises(RiskPyError) as exc:
            sp.betainc_vec(a, a, np.array([0.3, x]))
        assert exc.value.code == "RP304"
        assert time.perf_counter() - start < 5.0
    # ordinary arguments are untouched, and a NaN comes back as NaN rather than uninitialised memory
    # (the front factor x^a·y^b carries |a·ln x|·ε of rounding: 3e-9 at a = 1e8, as in 0.7.0)
    assert sp.betainc(1e6, 1e6, 0.5) == pytest.approx(0.5, abs=1e-10)
    assert sp.betainc(1e8, 1e8, 0.5) == pytest.approx(0.5, abs=1e-8)
    out = sp.betainc_vec(2.0, 3.0, np.array([np.nan, 0.5, 0.0, 1.0]))
    assert np.isnan(out[0]) and out[1:].tolist() == pytest.approx([0.6875, 0.0, 1.0], abs=1e-15)


# ---------------------------------------------------------------------------
# The t copula


def _tau_and_se(u, np):
    """Kendall's tau of columns 0 and −1 with its U-statistic standard error 2·sd(h₁)/√n."""
    a, b = u[:, 0], u[:, -1]
    n = a.size
    h = np.empty(n)
    for s in range(0, n, 1000):
        h[s:s + 1000] = (np.sign(a[s:s + 1000, None] - a[None, :])
                         * np.sign(b[s:s + 1000, None] - b[None, :])).sum(axis=1) / (n - 1)
    return float(h.mean()), 2.0 * float(h.std(ddof=1)) / math.sqrt(n)


def _check_sample(u, tau, np, n_se=4.0):
    stats = pytest.importorskip("scipy.stats")
    n = u.shape[0]
    assert np.all((u > 1e-16) & (u < 1.0 - 1e-16)), "mass at the clip value"
    for j in range(u.shape[1]):
        assert stats.kstest(u[:, j], "uniform").statistic < 2.5 / math.sqrt(n)
    est, se = _tau_and_se(u, np)
    assert abs(est - tau) < n_se * se, (est, tau, se)


def test_t_copula_with_huge_df_is_the_gaussian_copula():
    # 0.7.0 rounded df/(df + t²) to 1 from df ~ 1e17 on, so every uniform came out 0.5
    np = _np()
    g = GaussianCopula(0.5).sample(20_000, seed=9)
    for df in (1e17, 1e20, 1e300):
        u = StudentTCopula(0.5, df=df).sample(20_000, seed=9)        # the same latent normals come first
        assert np.max(np.abs(u - g)) < 1e-7
    from riskpy import tools

    base = {"means": [100.0, 100.0], "sds": [30.0, 30.0], "correlation": [[1, 0.5], [0.5, 1]], "trials": 20000,
            "seed": 1}
    gauss = tools.call("capital_allocation", {**base, "copula": "gaussian"})["total"]
    assert gauss == pytest.approx(385.67, abs=0.01)
    for df in (1e12, 1e20, 1e300):
        total = tools.call("capital_allocation", {**base, "copula": "t", "df": df})["total"]
        assert total == pytest.approx(gauss, rel=1e-6)


def test_t_copula_uniforms_are_the_t_cdf_to_rounding():
    mp = pytest.importorskip("mpmath")
    np = _np()
    for df in (4.0, 50.0, 9999.0, 1e5):
        cop = StudentTCopula(0.3, df=df)
        u = cop.sample(2000, seed=5)
        rng = np.random.default_rng(5)
        z = cop._latent(2000, rng)
        t = z / np.sqrt(rng.chisquare(df, size=2000) / df)[:, None]
        # the cells nearest 0.5 are where 0.7.0 lost the most (2.6e-7 at df = 9999)
        for i in np.argsort(np.abs(t.ravel()))[:5]:
            with mp.workdps(_dps(df)):
                ref = float(_t_cdf_mp(mp, float(t.ravel()[i]), df))
            assert u.ravel()[i] == pytest.approx(ref, abs=2e-16), (df, t.ravel()[i])


@pytest.mark.parametrize("df", [1e-300, 1e-3, 0.01, 0.05, 0.3])
def test_t_copula_with_almost_no_degrees_of_freedom_keeps_uniform_marginals(df):
    # χ²(df)/df underflowed to zero in 70% of rows at df = 0.001 (3% at 0.01) and 0.7.0 put
    # each such row at a corner of the unit square; any elliptical copula has τ = 2/π·asin ρ.
    np = _np()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for rho in (0.3, -0.6):
            u = StudentTCopula(rho, df=df).sample(5000, seed=3)
            _check_sample(u, 2.0 / math.pi * math.asin(rho), np)


def test_t_copula_df_floor():
    with pytest.raises(RiskPyError) as exc:
        StudentTCopula(0.5, df=5e-324)
    assert (exc.value.code, exc.value.param) == ("RP102", "df")
    assert StudentTCopula(0.5, df=1e-300).df == 1e-300


def test_t_copula_samples_in_blocks_with_the_same_draws():
    np = _np()
    import tracemalloc

    R = np.full((5, 5), 0.3)
    np.fill_diagonal(R, 1.0)
    for df in (4.0, 0.01):
        cop = StudentTCopula(R, df=df)
        n = 2 * (capital._BLOCK_CELLS // 5) + 17          # several blocks and a ragged last one
        cop.sample(10, seed=1)
        tracemalloc.start()
        u = cop.sample(n, seed=2)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        # 0.7.0 peaked near 170 bytes a cell (4.5 GB at the capital_allocation limit of 2e7 cells)
        assert peak / u.size < 40.0
        if df >= 0.2:
            rng = np.random.default_rng(2)
            z = cop._latent(n, rng)
            t = z / np.sqrt(rng.chisquare(df, size=n) / df)[:, None]
            assert np.array_equal(u, np.clip(sp.t_cdf_vec(t, df), 1e-16, 1.0 - 1e-16))


# ---------------------------------------------------------------------------
# Archimedean copulas at extreme theta


@pytest.mark.parametrize("theta,dim", [(40.0, 2), (100.0, 2), (500.0, 2), (-40.0, 2), (-100.0, 2), (-500.0, 2),
                                       (40.0, 3), (100.0, 3), (500.0, 3)])
def test_frank_sampler_at_large_theta(theta, dim):
    # 0.7.0: two risks cancelled 1 + r to nothing in the upper tail (v infinite, clipped to 1);
    # three or more drew the frailty with p = 1 − e^(−θ) rounded, so tau stuck at 0.897.
    np = _np()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        u = FrankCopula(theta, dim=dim).sample(4000, seed=11)
    _check_sample(u, FrankCopula(theta).kendall_tau(), np)


def test_frank_conditional_inverse_is_exact_in_the_upper_tail():
    mp = pytest.importorskip("mpmath")
    np = _np()

    def conditional(v, u, t):          # C(v | u) = ∂C/∂u for the Frank copula
        eu, ev, e1 = mp.exp(-t * u), mp.exp(-t * v), mp.exp(-t)
        return eu * (ev - 1) / ((e1 - 1) + (eu - 1) * (ev - 1))

    for theta in (30.0, 40.0, 200.0):
        sample = FrankCopula(theta).sample(3000, seed=4)
        rng = np.random.default_rng(4)
        u, w = rng.random(3000), rng.random(3000)
        for i in np.argsort(sample[:, 1])[-4:]:         # the deepest upper-tail draws
            with mp.workdps(300):                       # e^(−200) must survive 1 + e^(−θ)
                c = conditional(mp.mpf(sample[i, 1]), mp.mpf(u[i]), mp.mpf(theta))
            assert float(c) == pytest.approx(w[i], abs=1e-12)


def test_frank_capital_allocation_stays_below_the_comonotone_bound():
    _np()
    from riskpy import tools

    # lognormal lines, mean 100, sd 30: TVaR_α = mean·Φ(σ − z_α)/(1 − α); comonotone TVaR is the sum
    sigma = math.sqrt(math.log1p(0.09))
    z = sp.norm_ppf(0.99)
    bound = 2.0 * 100.0 * sp.norm_cdf(sigma - z) / 0.01
    for theta in (40.0, 100.0, 500.0):
        out = tools.call("capital_allocation", {"means": [100.0, 100.0], "sds": [30.0, 30.0], "copula": "frank",
                                                "copula_theta": theta, "trials": 200_000, "seed": 1})
        assert out["total"] < bound * 1.01, (theta, out["total"], bound)       # 0.7.0: 1,275 against 421


@pytest.mark.parametrize("theta", [15.0, 80.0, 200.0, 1000.0])
def test_clayton_sampler_at_large_theta(theta):
    # a Gamma(1/θ) frailty underflows to zero for large θ and 0.7.0 then clipped the row to 1e-16
    np = _np()
    for dim, survival in ((2, False), (3, True)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            u = ClaytonCopula(theta, dim=dim, survival=survival).sample(4000, seed=6)
        _check_sample(u, theta / (theta + 2.0), np)


def test_clayton_and_gumbel_at_the_largest_theta_are_comonotone():
    np = _np()
    for cop in (ClaytonCopula(1.7e308), GumbelCopula(1.7e308)):
        assert cop.cdf(0.3, 0.6) == pytest.approx(0.3, abs=1e-12)
        assert cop.joint_exceedance(0.995) == pytest.approx(0.005, rel=1e-9)
    u = ClaytonCopula(1.7e308, dim=3).sample(1000, seed=1)
    assert np.all(np.isfinite(u)) and np.max(np.abs(u[:, 0] - u[:, 2])) < 1e-12


def test_gumbel_cdf_and_joint_exceedance_match_mpmath():
    mp = pytest.importorskip("mpmath")

    def cdf(u, v, t):
        u, v, t = mp.mpf(u), mp.mpf(v), mp.mpf(t)
        return mp.exp(-((-mp.log(u)) ** t + (-mp.log(v)) ** t) ** (1 / t))

    with mp.workdps(60):
        for theta in (1.0, 2.0, 50.0, 100.0, 1000.0, 1e6):
            cop = GumbelCopula(theta)
            for u, v in ((0.3, 0.6), (0.995, 0.995), (1 - 3e-7, 1 - 3e-7), (1 - 1e-12, 1 - 2e-12), (0.01, 0.9)):
                assert cop.cdf(u, v) == pytest.approx(float(cdf(u, v, theta)), rel=1e-13), (theta, u, v)
            for q in (1e-9, 0.25, 0.995, 1 - 3e-7, 1 - 1e-9):
                # 0.7.0 underflowed (−ln q)^θ: 1.4% off at θ = 50 and q = 1 − 3e-7, and the comonotone
                # answer from θ = 100; it also cancelled 1 − 2q + C(q, q) to a few digits near q = 1.
                ref = 1 - 2 * mp.mpf(q) + cdf(q, q, theta)
                assert cop.joint_exceedance(q) == pytest.approx(float(ref), rel=1e-13), (theta, q)


def test_frank_tau_series_and_tiny_tau():
    mp = pytest.importorskip("mpmath")

    def tau(theta):                     # 1 − 4/θ·(1 − D₁(θ)), D₁ the Debye function, in 50 digits
        with mp.workdps(50):
            th = mp.mpf(theta)
            d1 = mp.quad(lambda t: t / mp.expm1(t), [0, abs(th)]) / abs(th)
            if th < 0:
                d1 += -th / 2
            return float(1 - 4 / th * (1 - d1))

    for theta in (1e-9, 1e-4, 0.05, 0.0999, -0.0999, 0.1, 1.0, -3.0):
        assert FrankCopula(theta).kendall_tau() == pytest.approx(tau(theta), rel=1e-11)
    # 0.7.0 searched [1e-6, 500] and failed with a bare "no sign change" for |tau| below 1e-7
    for t in (5e-324, 1e-300, 1e-12, 1e-9, -1e-9, 1e-7, 9.9e-5, 1e-4, 1e-3, 0.3, -0.7):
        theta = FrankCopula.from_kendall_tau(t).theta
        assert FrankCopula(theta).kendall_tau() == pytest.approx(t, rel=1e-11, abs=1e-300), t


# ---------------------------------------------------------------------------
# Inputs


def test_two_by_two_matrix_copulas_need_no_numpy(monkeypatch):
    def missing():
        raise MissingDependencyError("NumPy is not installed", code="RP201")

    monkeypatch.setattr(capital, "_numpy", missing)
    cop = GaussianCopula([[1.0, 0.5], [0.5, 1.0]])
    assert repr(cop) == "GaussianCopula(0.5)" and cop.kendall_tau() == pytest.approx(1.0 / 3.0)
    assert cop.cdf(0.5, 0.5) == pytest.approx(0.25 + math.asin(0.5) / (2.0 * math.pi), abs=1e-12)
    t = StudentTCopula(((1.0, -0.3), (-0.3, 1.0)), df=4)
    assert t.tail_dependence().upper == pytest.approx(StudentTCopula(-0.3, df=4).tail_dependence().upper)
    with pytest.raises(RiskPyError) as exc:
        GaussianCopula([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    assert exc.value.code == "RP201" and "3 risks" in str(exc.value)
    for bad, code in (([[1.0, 0.5], [0.4, 1.0]], "RP100"), ([[1.0, 0.5], [0.5, 1.1]], "RP100"),
                      ([[1.0, 1.5], [1.5, 1.0]], "RP100"), ([[1.0, 0.5], [0.5]], "RP100"),
                      ([[1, "x"], ["x", 1]], "RP404"), ("0.5", "RP404"), ([[1.0, math.nan], [math.nan, 1.0]], "RP101")):
        with pytest.raises(RiskPyError) as exc:
            GaussianCopula(bad)
        assert (exc.value.code, exc.value.param) == (code, "correlation"), bad


def test_numpy_scalars_are_numbers():
    np = _np()
    losses = np.random.default_rng(0).lognormal(size=(1000, 2))
    assert aggregate_capital({"a": np.int64(3), "b": np.int64(4)}, np.eye(2)).total == pytest.approx(5.0)
    assert aggregate_capital([np.float32(3), np.float32(4)], np.eye(2)).total == pytest.approx(5.0)
    assert ClaytonCopula(np.int64(2)).theta == 2.0 and ClaytonCopula(2.0, dim=np.int8(3)).dim == 3
    assert StudentTCopula(0.5, df=np.int64(4)).df == 4.0
    assert GaussianCopula(np.float32(0.5)).kendall_tau() == pytest.approx(1.0 / 3.0, abs=1e-7)
    assert GumbelCopula(2.0).joint_exceedance(np.float32(0.995)) == pytest.approx(0.002936, abs=1e-5)
    assert capital.allocate(losses, level=np.float32(0.99)).level == pytest.approx(0.99)
    assert GaussianCopula(np.eye(3)).kendall_tau(np.int64(0), np.int64(1)) == 0.0
    # booleans are still not numbers, NumPy's included
    for bad in (True, np.bool_(True)):
        with pytest.raises(RiskPyError):
            ClaytonCopula(bad)
        with pytest.raises(RiskPyError):
            GaussianCopula(np.eye(3)).kendall_tau(bad, 2)


def test_contradictory_pairs_names_with_a_dataframe_and_text_are_refused():
    np = _np()
    with pytest.raises(RiskPyError) as exc:
        aggregate_capital({"a": 3, "b": 4}, {("a", "b"): 0.9, ("b", "a"): -0.9})
    assert (exc.value.code, exc.value.param) == ("RP113", "correlation")
    # the same pair in both orders with one value is fine
    assert aggregate_capital({"a": 3, "b": 4}, {("a", "b"): 0.5, ("b", "a"): 0.5}).total == pytest.approx(
        math.sqrt(9 + 16 + 12))
    with pytest.raises(RiskPyError) as exc:
        capital.allocate({"x": ["a", "b"], "y": [1.0, 2.0]})
    assert (exc.value.code, exc.value.param) == ("RP404", "losses")
    with pytest.raises(RiskPyError) as exc:
        capital.allocate({"x": np.full(5, 1e308), "y": np.full(5, 1e308)})
    assert (exc.value.code, exc.value.param) == ("RP304", "losses")
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(np.random.default_rng(0).lognormal(size=(100, 2)), columns=["x", "y"])
    with pytest.raises(RiskPyError) as exc:
        capital.allocate(df, names=["motor", "property"])
    assert (exc.value.code, exc.value.param) == ("RP113", "names")
    with pytest.raises(RiskPyError) as exc:
        capital.allocate(pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]}))
    assert (exc.value.code, exc.value.param) == ("RP404", "losses")


def test_euler_var_spreads_the_window_residual_additively():
    np = _np()
    n = 400_000
    half = max(int(round(0.5 * math.sqrt(n))), 5)
    se = 20.0 / math.sqrt(2 * half + 1)
    z99 = sp.norm_ppf(0.99)
    # A hedged book: B = −A + e with e independent of A, so S = e and E[A | S = s] = 100 exactly.
    # With the VaR near zero, 0.7.0's rescaling by VaR / window total moved A by 20% or more.
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        a = rng.normal(100.0, 20.0, n)
        e = rng.normal(-5.0 * z99, 5.0, n)
        alloc = capital.allocate({"A": a, "B": -a + e}, level=0.99, measure="var")
        assert abs(alloc.contributions["A"] - 100.0) < 5.0 * se
        assert sum(alloc.contributions.values()) == pytest.approx(alloc.total, abs=1e-9)
    # A perfect hedge: S = 0 in every trial, an atom at the VaR, so E[A | S = 0] = E[A].
    a = np.random.default_rng(1).normal(100.0, 20.0, 10_000)
    alloc = capital.allocate({"A": a, "B": -a}, level=0.99, measure="var")
    assert alloc.contributions["A"] == pytest.approx(a.mean(), rel=1e-12)
    # A positive book: the same as 0.7.0's multiplicative version, well inside the sampling error
    rng = np.random.default_rng(3)
    x = rng.multivariate_normal([100.0, 60.0], [[400.0, 300.0], [300.0, 900.0]], n)
    alloc = capital.allocate({"a": x[:, 0], "b": x[:, 1]}, level=0.99, measure="var")
    s = x.sum(axis=1)
    order = np.argsort(s, kind="mergesort")
    centre = int(math.ceil(n * 0.99 - 1e-9)) - 1
    local = x[order[centre - half:centre + half + 1]].mean(axis=0)
    old = local * (alloc.total / local.sum())
    assert np.max(np.abs(np.array(list(alloc.contributions.values())) - old)) < 0.1 * se
    var_s = 400.0 + 900.0 + 600.0
    exact_a = 100.0 + (400.0 + 300.0) / var_s * (alloc.total - 160.0)        # E[a | S = VaR], jointly normal
    assert alloc.contributions["a"] == pytest.approx(exact_a, abs=5.0 * se)


def test_docstring_examples_have_the_right_digits():
    import inspect

    assert f"{GaussianCopula(0.5).spearman_rho():.5f}" == "0.48258"
    assert "0.48258" in inspect.getsource(GaussianCopula.spearman_rho)
    assert f"{GumbelCopula(2.0).joint_exceedance(0.995):.5f}" == "0.00294"
    assert "0.00294" in inspect.getsource(capital.Copula.joint_exceedance)


def test_copula_methods_at_extreme_parameters_raise_only_coded_errors():
    # A fuzz over every family at tiny, huge and boundary parameters: each call either answers
    # with a probability (or a sample strictly inside the unit cube) or raises a RiskPyError.
    np = _np()
    cops = []
    for rho in (-1.0, -0.5, 0.0, 0.9999999, 1.0):
        cops.append(GaussianCopula(rho))
        cops += [StudentTCopula(rho, df=df) for df in (1e-300, 1e-3, 1.0, 1e4, 1e17, 1.7e308)]
    for theta in (1e-300, 1e-6, 1.0, 15.0, 1e3, 1e300, 1.7e308):
        cops += [ClaytonCopula(theta, dim, s) for dim in (2, 3) for s in (False, True)]
    for theta in (1.0, 1.0 + 1e-15, 50.0, 1e6, 1.7e308):
        cops += [GumbelCopula(theta, dim, s) for dim in (2, 3) for s in (False, True)]
    for theta in (-500.0, -37.0, -1e-9, 5e-324, 1e-6, 0.1, 25.0, 37.0, 500.0):
        cops += [FrankCopula(theta, dim) for dim in ((2, 3) if theta > 0 else (2,))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for cop in cops:
            calls = [cop.kendall_tau, cop.tail_dependence]
            calls += [lambda u=u, v=v: cop.cdf(u, v) for u, v in ((0.0, 0.5), (1e-300, 0.5), (0.3, 0.6),
                                                                  (1 - 1e-12, 1 - 1e-12), (5e-324, 5e-324))]
            calls += [lambda q=q: cop.joint_exceedance(q) for q in (1e-300, 1e-9, 0.5, 0.995, 1 - 1e-12)]
            for call in calls:
                try:
                    out = call()
                except RiskPyError:
                    continue
                values = list(out) if isinstance(out, tuple) else [out]
                assert all(isinstance(x, float) and -1.0 <= x <= 1.0 for x in values), (cop, values)
            u = cop.sample(500, seed=1)
            assert np.all((u > 0.0) & (u < 1.0)), cop
