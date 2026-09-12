"""Tests for the dependency-free special functions.

Two layers of evidence: closed-form identities that must hold to machine
precision regardless of implementation, and — when SciPy happens to be
installed — agreement with it as an independent oracle. SciPy is never a
dependency of the package; it is only ever a witness in the test suite.
"""

from __future__ import annotations

import math

import pytest

from riskpy import _special as sp


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------


def test_norm_cdf_at_zero_and_symmetry():
    assert sp.norm_cdf(0.0) == 0.5
    for x in (0.3, 1.0, 2.5, 6.0):
        assert sp.norm_cdf(x) + sp.norm_cdf(-x) == pytest.approx(1.0, abs=1e-16)


def test_norm_cdf_keeps_the_far_tail():
    # erf-based forms round to exactly zero out here; erfc does not.
    assert sp.norm_cdf(-30.0) > 0.0
    assert sp.norm_cdf(-37.0) > 0.0
    assert sp.norm_cdf(-8.0) == pytest.approx(6.220960574271784e-16, rel=1e-12)


def test_norm_ppf_inverts_cdf_across_the_range():
    # The lower tail round-trips to machine precision because erfc keeps it.
    for x in (-8.0, -5.5, -3.7, -1.0, -0.001, 0.0, 0.4, 2.2):
        assert sp.norm_ppf(sp.norm_cdf(x)) == pytest.approx(x, abs=1e-13)
    # The upper tail cannot: p = 1 - 1.9e-8 has only eight digits of tail left
    # in a double, so the round trip is limited by the *representation* of p,
    # not by the solver. Symmetry is the honest check there.
    assert sp.norm_ppf(sp.norm_cdf(5.5)) == pytest.approx(5.5, abs=1e-9)
    assert sp.norm_ppf(1.0 - sp.norm_cdf(-5.5)) == pytest.approx(5.5, abs=1e-9)
    assert -sp.norm_ppf(sp.norm_cdf(-5.5)) == pytest.approx(5.5, abs=1e-13)


def test_norm_ppf_known_values():
    assert sp.norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-14)
    assert sp.norm_ppf(0.5) == 0.0
    assert sp.norm_ppf(0.99) == pytest.approx(2.3263478740408408, abs=1e-14)
    assert sp.norm_ppf(0.0) == -math.inf and sp.norm_ppf(1.0) == math.inf


def test_norm_ppf_rejects_out_of_range():
    with pytest.raises(ValueError):
        sp.norm_ppf(1.5)
    with pytest.raises(ValueError):
        sp.norm_ppf(-0.1)


# ---------------------------------------------------------------------------
# Gamma family
# ---------------------------------------------------------------------------


def test_gammainc_closed_forms():
    assert sp.gammainc(1.0, 2.3) == pytest.approx(1.0 - math.exp(-2.3), abs=1e-14)
    assert sp.gammainc(3.7, 0.0) == 0.0
    assert sp.gammaincc(3.7, 0.0) == 1.0
    # P(n, x) is the Poisson survival function P(N >= n) for integer n.
    n, x = 4, 2.5
    poisson_tail = 1.0 - sum(math.exp(-x) * x ** k / math.factorial(k) for k in range(n))
    assert sp.gammainc(float(n), x) == pytest.approx(poisson_tail, abs=1e-14)


def test_gammainc_and_complement_sum_to_one():
    for a, x in ((0.2, 0.1), (0.5, 3.0), (2.5, 1.0), (10.0, 25.0), (100.0, 80.0)):
        assert sp.gammainc(a, x) + sp.gammaincc(a, x) == pytest.approx(1.0, abs=1e-14)


def test_gammainc_rejects_bad_arguments():
    with pytest.raises(ValueError):
        sp.gammainc(0.0, 1.0)
    with pytest.raises(ValueError):
        sp.gammainc(1.0, -1.0)
    with pytest.raises(ValueError):
        sp.gammaln(-1.0)


def test_digamma_and_trigamma_known_values():
    euler = 0.5772156649015329
    assert sp.digamma(1.0) == pytest.approx(-euler, abs=1e-13)
    assert sp.digamma(2.0) == pytest.approx(1.0 - euler, abs=1e-13)
    assert sp.trigamma(1.0) == pytest.approx(math.pi ** 2 / 6.0, abs=1e-13)
    # Recurrence psi(x + 1) = psi(x) + 1/x.
    assert sp.digamma(3.3) == pytest.approx(sp.digamma(2.3) + 1.0 / 2.3, abs=1e-13)


# ---------------------------------------------------------------------------
# Beta family and the distributions built on it
# ---------------------------------------------------------------------------


def test_betainc_closed_forms_and_symmetry():
    assert sp.betainc(1.0, 1.0, 0.37) == pytest.approx(0.37, abs=1e-14)
    assert sp.betainc(3.0, 1.0, 0.6) == pytest.approx(0.6 ** 3, abs=1e-14)
    assert sp.betainc(2.5, 4.0, 0.3) + sp.betainc(4.0, 2.5, 0.7) == pytest.approx(1.0, abs=1e-14)
    assert sp.betainc(2.0, 3.0, 0.0) == 0.0 and sp.betainc(2.0, 3.0, 1.0) == 1.0


def test_beta_function_matches_gamma_ratio():
    assert sp.beta_function(2.0, 3.0) == pytest.approx(1.0 / 12.0, rel=1e-13)


def test_student_t_reduces_to_cauchy_and_normal():
    assert sp.t_cdf(1.3, 1.0) == pytest.approx(0.5 + math.atan(1.3) / math.pi, abs=1e-13)
    assert sp.t_cdf(0.0, 7.0) == 0.5
    # Very many degrees of freedom is the normal.
    assert sp.t_cdf(1.5, 1e7) == pytest.approx(sp.norm_cdf(1.5), abs=1e-6)
    assert sp.t_ppf(0.9, 2e6) == pytest.approx(sp.norm_ppf(0.9), abs=1e-6)


def test_t_ppf_inverts_t_cdf():
    for df in (1.0, 2.5, 4.0, 30.0):
        for x in (-4.0, -0.7, 0.0, 1.9):
            assert sp.t_ppf(sp.t_cdf(x, df), df) == pytest.approx(x, abs=1e-8)


def test_chi_square_closed_form():
    # Two degrees of freedom is exponential with mean 2.
    assert sp.chi2_cdf(3.0, 2.0) == pytest.approx(1.0 - math.exp(-1.5), abs=1e-14)
    assert sp.chi2_cdf(0.0, 5.0) == 0.0
    assert sp.chi2_cdf(sp.chi2_ppf(0.95, 3.0), 3.0) == pytest.approx(0.95, abs=1e-12)


def test_distribution_functions_reject_bad_parameters():
    with pytest.raises(ValueError):
        sp.betainc(0.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        sp.betainc(1.0, 1.0, 1.5)
    with pytest.raises(ValueError):
        sp.t_cdf(1.0, 0.0)
    with pytest.raises(ValueError):
        sp.t_ppf(1.2, 3.0)
    with pytest.raises(ValueError):
        sp.chi2_ppf(0.0, 3.0)


def test_kolmogorov_distribution():
    assert sp.kolmogorov_sf(0.0) == 1.0
    assert sp.kolmogorov_sf(1.0) == pytest.approx(0.26999967167735456, abs=1e-12)
    assert sp.kolmogorov_sf(10.0) == 0.0
    assert 0.0 < sp.ks_pvalue(0.05, 400) < 1.0
    with pytest.raises(ValueError):
        sp.ks_pvalue(0.1, 0)


# ---------------------------------------------------------------------------
# Numerical routines
# ---------------------------------------------------------------------------


def test_root_finders():
    root = 2.0 ** (1.0 / 3.0)
    assert sp.brent(lambda x: x ** 3 - 2.0, 0.0, 2.0) == pytest.approx(root, abs=1e-12)
    assert sp.bisect(lambda x: x ** 3 - 2.0, 0.0, 2.0) == pytest.approx(root, abs=1e-10)
    assert sp.brent(lambda x: x - 1.0, 1.0, 2.0) == 1.0  # exact hit at the bracket
    with pytest.raises(ValueError, match="sign change"):
        sp.brent(lambda x: x * x + 1.0, -1.0, 1.0)
    with pytest.raises(ValueError, match="sign change"):
        sp.bisect(lambda x: x * x + 1.0, -1.0, 1.0)


def test_invert_monotone_expands_its_bracket():
    assert sp.invert_monotone(lambda x: x * x, 400.0, 0.0, 1.0) == pytest.approx(20.0, abs=1e-9)
    with pytest.raises(ValueError):
        sp.invert_monotone(lambda x: x * x, 400.0, 0.0, 1.0, expand=False)


def test_minimisers():
    assert sp.minimise_scalar(lambda x: (x - 1.3) ** 2 + 1.0, -5.0, 5.0) == pytest.approx(1.3, abs=1e-7)
    point, value = sp.nelder_mead(lambda v: (v[0] - 1.0) ** 2 + 10.0 * (v[1] + 2.0) ** 2 + 3.0, [0.0, 0.0])
    assert point[0] == pytest.approx(1.0, abs=1e-5)
    assert point[1] == pytest.approx(-2.0, abs=1e-5)
    assert value == pytest.approx(3.0, abs=1e-8)
    with pytest.raises(ValueError):
        sp.nelder_mead(lambda v: 0.0, [])
    with pytest.raises(ValueError):
        sp.nelder_mead(lambda v: 0.0, [0.0, 0.0], step=[0.1])


def test_simpson():
    assert sp.simpson(math.sin, 0.0, math.pi, 200) == pytest.approx(2.0, abs=1e-9)
    assert sp.simpson(lambda x: x ** 3, 0.0, 1.0, 2) == pytest.approx(0.25, abs=1e-15)  # exact for cubics
    with pytest.raises(ValueError):
        sp.simpson(math.sin, 0.0, 1.0, 1)


# ---------------------------------------------------------------------------
# Vectorised variants
# ---------------------------------------------------------------------------


def test_vectorised_variants_match_scalars():
    np = pytest.importorskip("numpy")
    x = np.linspace(0.0, 30.0, 301)
    scalar = np.array([sp.gammainc(2.5, float(v)) for v in x])
    assert np.max(np.abs(sp.gammainc_vec(2.5, x) - scalar)) < 1e-14
    u = np.linspace(0.0, 1.0, 201)
    scalar = np.array([sp.betainc(1.8, 4.2, float(v)) for v in u])
    assert np.max(np.abs(sp.betainc_vec(1.8, 4.2, u) - scalar)) < 1e-14
    z = np.linspace(-9.0, 9.0, 181)
    assert np.max(np.abs(sp.norm_cdf_vec(z) - np.array([sp.norm_cdf(float(v)) for v in z]))) == 0.0
    p = np.linspace(0.001, 0.999, 99)
    assert np.max(np.abs(sp.norm_ppf_vec(p) - np.array([sp.norm_ppf(float(v)) for v in p]))) == 0.0
    # Scalars in, scalars (0-d) out — the distributions rely on this.
    assert float(sp.norm_cdf_vec(1.96)) == pytest.approx(0.9750021048517795)
    assert sp.gammainc_vec(2.0, np.array([[1.0, 2.0], [3.0, 4.0]])).shape == (2, 2)


def test_vectorised_variants_reject_bad_input():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        sp.gammainc_vec(2.0, np.array([-1.0]))
    with pytest.raises(ValueError):
        sp.betainc_vec(2.0, 3.0, np.array([1.5]))


# ---------------------------------------------------------------------------
# SciPy as an oracle
# ---------------------------------------------------------------------------


def test_against_scipy_oracle():
    special = pytest.importorskip("scipy.special")
    stats = pytest.importorskip("scipy.stats")
    for p in (1e-300, 1e-12, 1e-6, 0.001, 0.3, 0.9, 0.999, 1.0 - 1e-12):
        assert sp.norm_ppf(p) == pytest.approx(float(stats.norm.ppf(p)), rel=4e-16, abs=1e-300)
    for x in (-40.0, -8.0, -3.0, 0.0, 2.5, 9.0):
        assert sp.norm_cdf(x) == pytest.approx(float(stats.norm.cdf(x)), rel=1e-13, abs=1e-300)
    for a in (0.1, 0.5, 1.0, 2.5, 10.0, 100.0):
        for x in (0.0, 0.01, 0.5, 1.0, 3.0, 10.0, 50.0, 200.0):
            assert sp.gammainc(a, x) == pytest.approx(float(special.gammainc(a, x)), abs=1e-13)
            assert sp.gammaincc(a, x) == pytest.approx(float(special.gammaincc(a, x)), abs=1e-13)
    for a in (0.3, 1.0, 2.5, 10.0, 100.0):
        for b in (0.5, 1.0, 3.0, 40.0):
            for x in (0.0, 0.01, 0.2, 0.5, 0.9, 0.999, 1.0):
                assert sp.betainc(a, b, x) == pytest.approx(float(special.betainc(a, b, x)), abs=1e-13)
    for df in (1.0, 2.5, 4.0, 30.0, 200.0):
        for x in (-5.0, -1.0, 0.3, 4.0):
            assert sp.t_cdf(x, df) == pytest.approx(float(stats.t.cdf(x, df)), abs=1e-12)
        for p in (0.001, 0.05, 0.5, 0.975, 0.9999):
            ref = float(stats.t.ppf(p, df))
            assert sp.t_ppf(p, df) == pytest.approx(ref, rel=1e-8, abs=1e-8)
    for x in (0.1, 0.5, 1.0, 2.7, 6.0, 50.0, 1000.0):
        assert sp.digamma(x) == pytest.approx(float(special.digamma(x)), abs=1e-13)
        assert sp.trigamma(x) == pytest.approx(float(special.polygamma(1, x)), abs=1e-13)
    assert sp.kolmogorov_sf(1.0) == pytest.approx(float(stats.kstwobign.sf(1.0)), abs=1e-12)
