import math

import pytest
from riskpy import FourierTransform


def test_fft_ifft_roundtrip():
    real = [1.0, 2.0, 3.0, 4.0]
    imag = [0.0, 0.0, 0.0, 0.0]
    fre, fim = FourierTransform.fft(real, imag)
    back_re, back_im = FourierTransform.ifft(fre, fim)
    for i, x in enumerate(real):
        assert back_re[i] == pytest.approx(x, abs=1e-10)
        assert back_im[i] == pytest.approx(0.0, abs=1e-10)


def test_fft_impulse():
    # DFT of unit impulse at 0 is all ones (on padded grid)
    real = [1.0]
    imag = [0.0]
    fre, fim = FourierTransform.fft(real, imag)
    for r, i in zip(fre, fim):
        assert r == pytest.approx(1.0, abs=1e-10)
        assert i == pytest.approx(0.0, abs=1e-10)


def test_fft_zero_pads_to_power_of_two():
    real = [1.0, 2.0, 3.0]  # length 3 → pad to 4
    imag = [0.0, 0.0, 0.0]
    fre, fim = FourierTransform.fft(real, imag)
    assert len(fre) == 4
    assert len(fim) == 4


def test_convolve_simple():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0]
    # Linear: [4, 13, 22, 15]
    out = FourierTransform.convolve(a, b)
    assert len(out) == 4
    assert out[0] == pytest.approx(4.0, abs=1e-9)
    assert out[1] == pytest.approx(13.0, abs=1e-9)
    assert out[2] == pytest.approx(22.0, abs=1e-9)
    assert out[3] == pytest.approx(15.0, abs=1e-9)


def test_convolve_empty():
    assert FourierTransform.convolve([], [1.0]) == []
    assert FourierTransform.convolve([1.0], []) == []


def test_compound_poisson_deterministic_closed_form():
    # Severity = 1 with probability 1 → S ~ Poisson(λ) on the non-negative integers
    lam = 2.0
    severity = [0.0, 1.0]  # P(X=0)=0, P(X=1)=1
    pmf = FourierTransform.compound_poisson_pmf(severity, lam, grid_size=64)
    # P(S=k) = e^{-λ} λ^k / k!
    for k in range(0, 15):
        expected = math.exp(-lam) * (lam ** k) / math.factorial(k)
        assert pmf[k] == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_compound_poisson_mean_lambda_e_sev():
    # severity: P(0)=0.5, P(1)=0.5 → E[X]=0.5; λ=4 → E[S]=2
    severity = [0.5, 0.5]
    lam = 4.0
    pmf = FourierTransform.compound_poisson_pmf(severity, lam, grid_size=128)
    mean = sum(k * p for k, p in enumerate(pmf))
    assert mean == pytest.approx(lam * 0.5, rel=1e-4, abs=1e-4)


def test_compound_poisson_validation():
    with pytest.raises(Exception):
        FourierTransform.compound_poisson_pmf([], 1.0)
    with pytest.raises(Exception):
        FourierTransform.compound_poisson_pmf([1.0], -1.0)
    with pytest.raises(Exception):
        FourierTransform.compound_poisson_pmf([-0.1, 1.1], 1.0)
    with pytest.raises(Exception):
        FourierTransform.compound_poisson_pmf([1.0], 1.0, grid_size=3)  # not power of 2


def test_fft_mismatched_lengths():
    with pytest.raises(Exception):
        FourierTransform.fft([1.0, 2.0], [0.0])


def test_fft_empty_returns_empty():
    re, im = FourierTransform.fft([], [])
    assert re == []
    assert im == []


def test_fft_rejects_non_finite():
    with pytest.raises(Exception):
        FourierTransform.fft([float("nan")], [0.0])
    with pytest.raises(Exception):
        FourierTransform.convolve([1.0], [float("inf")])


def test_compound_poisson_auto_normalizes_severity():
    # Unnormalized histogram should behave like the normalized PMF
    raw = [2.0, 2.0]  # same shape as [0.5, 0.5]
    lam = 3.0
    pmf_raw = FourierTransform.compound_poisson_pmf(raw, lam, grid_size=64)
    pmf_norm = FourierTransform.compound_poisson_pmf([0.5, 0.5], lam, grid_size=64)
    for a, b in zip(pmf_raw, pmf_norm):
        assert a == pytest.approx(b, abs=1e-9)
    assert sum(pmf_raw) == pytest.approx(1.0, abs=1e-9)
    mean = sum(k * p for k, p in enumerate(pmf_raw))
    assert mean == pytest.approx(lam * 0.5, rel=1e-3, abs=1e-3)


def test_compound_poisson_zero_mass_rejected():
    with pytest.raises(Exception):
        FourierTransform.compound_poisson_pmf([0.0, 0.0], 1.0)


def test_compound_poisson_lambda_zero():
    pmf = FourierTransform.compound_poisson_pmf([0.25, 0.75], 0.0, grid_size=32)
    assert pmf[0] == pytest.approx(1.0, abs=1e-10)
    assert sum(pmf) == pytest.approx(1.0, abs=1e-9)
