import pytest
from riskpy import RateAnalyzer


def test_on_level_premiums_basic():
    ra = RateAnalyzer()
    # Work from latest year backwards: last year factor 1.0, earlier years accumulate
    prem = [100.0, 110.0, 120.0]
    changes = [0.05, 0.10, 0.0]  # rate change effective after each year
    on_level = ra.on_level_premiums(prem, changes)
    assert len(on_level) == 3
    # Most recent year unchanged by subsequent changes
    assert on_level[2] == pytest.approx(120.0)
    # Prior year scaled by last change
    assert on_level[1] == pytest.approx(110.0 * 1.0)
    assert on_level[0] == pytest.approx(100.0 * 1.0 * 1.10)


def test_on_level_factor():
    ra = RateAnalyzer()
    f = ra.on_level_factor([0.10, 0.05])
    assert f == pytest.approx(1.10 * 1.05)


def test_trend_factor():
    ra = RateAnalyzer()
    assert ra.trend_factor(0.05, 2.0) == pytest.approx(1.05 ** 2)
    assert ra.trend_factor(0.0, 5.0) == pytest.approx(1.0)


def test_rate_change_impact():
    ra = RateAnalyzer()
    assert ra.rate_change_impact(100.0, 110.0) == pytest.approx(0.10)
    assert ra.rate_change_impact(200.0, 180.0) == pytest.approx(-0.10)


def test_combined_ratio():
    ra = RateAnalyzer(target_combined_ratio=1.0, expense_ratio=0.30)
    assert ra.combined_ratio(0.65) == pytest.approx(0.95)


def test_required_rate_change():
    ra = RateAnalyzer(target_combined_ratio=1.0, expense_ratio=0.30)
    # PLR = 0.70; current LR 0.84 → need 20% rate increase
    assert ra.required_rate_change(0.84) == pytest.approx(0.20)


def test_permissible_loss_ratio():
    ra = RateAnalyzer(target_combined_ratio=1.0, expense_ratio=0.25)
    assert ra.permissible_loss_ratio() == pytest.approx(0.75)


def test_loss_cost_rate():
    ra = RateAnalyzer()
    assert ra.loss_cost_rate(50000.0, 1000.0) == pytest.approx(50.0)


def test_on_level_premiums_validation():
    ra = RateAnalyzer()
    with pytest.raises(Exception):
        ra.on_level_premiums([100.0], [0.1, 0.2])  # size mismatch
    with pytest.raises(Exception):
        ra.on_level_premiums([100.0], [-1.0])  # -100% rate change
    with pytest.raises(Exception):
        ra.on_level_premiums([-10.0], [0.0])


def test_constructor_validation():
    with pytest.raises(Exception):
        RateAnalyzer(target_combined_ratio=0.0)
    with pytest.raises(Exception):
        RateAnalyzer(expense_ratio=1.0)
    with pytest.raises(Exception):
        RateAnalyzer(expense_ratio=-0.1)


def test_required_rate_change_and_trend_validation():
    ra = RateAnalyzer(target_combined_ratio=1.0, expense_ratio=0.30)
    with pytest.raises(Exception):
        ra.required_rate_change(-0.1)
    with pytest.raises(Exception):
        ra.trend_factor(-1.0, 1.0)
    with pytest.raises(Exception):
        ra.loss_cost_rate(10.0, 0.0)


def test_on_level_empty_and_single():
    ra = RateAnalyzer()
    assert ra.on_level_premiums([], []) == []
    assert ra.on_level_premiums([250.0], [0.10]) == pytest.approx([250.0])


def test_constructor_rejects_non_finite():
    with pytest.raises(Exception):
        RateAnalyzer(target_combined_ratio=float("nan"))
    with pytest.raises(Exception):
        RateAnalyzer(expense_ratio=float("inf"))
