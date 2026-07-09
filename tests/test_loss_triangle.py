import pytest
from riskpy import LossTriangle


def _sample_triangle():
    t = LossTriangle()
    t.add_origin_year(2020, [100.0, 150.0, 180.0, 200.0])
    t.add_origin_year(2021, [120.0, 170.0, 200.0])
    t.add_origin_year(2022, [140.0, 190.0])
    t.add_origin_year(2023, [160.0])
    return t


def test_origin_year_and_period_counts():
    t = _sample_triangle()
    assert t.get_origin_year_count() == 4
    assert t.get_development_period_count() == 4


def test_development_factors_positive():
    t = _sample_triangle()
    factors = t.get_development_factors()
    assert len(factors) == 3
    for f in factors:
        assert f > 0.0


def test_ultimate_losses_length_and_order():
    t = _sample_triangle()
    ultimates = t.get_ultimate_losses()
    assert len(ultimates) == 4
    # Fully developed year should equal its latest diagonal
    assert ultimates[0] == pytest.approx(200.0, rel=1e-9)


def test_ibnr_total():
    # Hand-checked chain-ladder triangle (LDFs 1.5, 1.2, 10/9 → CDF 2.0, 4/3, 10/9, 1)
    # Ultimates 2000, 2400, 2800, 3200 → IBNR 0 + 240 + 700 + 1600 = 2540
    t = LossTriangle()
    t.add_origin_year(2019, [1000.0, 1500.0, 1800.0, 2000.0])
    t.add_origin_year(2020, [1200.0, 1800.0, 2160.0])
    t.add_origin_year(2021, [1400.0, 2100.0])
    t.add_origin_year(2022, [1600.0])
    ibnr = t.get_ibnr_reserves()
    assert ibnr[0] == pytest.approx(0.0, abs=1e-6)
    assert sum(ibnr) == pytest.approx(2540.0, rel=1e-6)


def test_single_year_no_development():
    t = LossTriangle()
    t.add_origin_year(2024, [500.0])
    assert t.get_development_factors() == []
    assert t.get_ultimate_losses() == pytest.approx([500.0])
    assert t.get_ibnr_reserves() == pytest.approx([0.0])


def test_empty_triangle():
    t = LossTriangle()
    assert t.get_origin_year_count() == 0
    assert t.get_development_period_count() == 0
    assert t.get_development_factors() == []
    assert t.get_ultimate_losses() == []
    assert t.get_ibnr_reserves() == []


def test_flat_development():
    t = LossTriangle()
    t.add_origin_year(2020, [100.0, 100.0, 100.0])
    t.add_origin_year(2021, [200.0, 200.0])
    factors = t.get_development_factors()
    assert all(f == pytest.approx(1.0) for f in factors)
    ultimates = t.get_ultimate_losses()
    assert ultimates[0] == pytest.approx(100.0)
    assert ultimates[1] == pytest.approx(200.0)


def test_ibnr_non_negative():
    t = _sample_triangle()
    for r in t.get_ibnr_reserves():
        assert r >= 0.0


def test_validation_rejects_bad_input():
    t = LossTriangle()
    with pytest.raises(Exception):
        t.add_origin_year(-1, [1.0])
    with pytest.raises(Exception):
        t.add_origin_year(2020, [])
    with pytest.raises(Exception):
        t.add_origin_year(2020, [-5.0])
    with pytest.raises(Exception):
        t.add_origin_year(2020, [float("nan")])


def test_uneven_development_ultimates():
    t = LossTriangle()
    t.add_origin_year(1, [10.0, 20.0, 30.0])
    t.add_origin_year(2, [15.0])
    # LDF0 = 20/10 = 2.0, LDF1 = 30/20 = 1.5 → year2 ultimate = 15 * 2 * 1.5 = 45
    assert t.get_ultimate_losses() == pytest.approx([30.0, 45.0])
    assert t.get_ibnr_reserves() == pytest.approx([0.0, 30.0])
