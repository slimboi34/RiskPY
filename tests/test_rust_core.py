"""0.9.0: the compiled core is Rust. What changed on purpose, and what must not have.

* The random stream: PCG64-DXSM seeded through SplitMix64, written out again here in
  plain Python, so a seeded result is pinned by an independent implementation (and
  checked against NumPy's PCG64DXSM when NumPy is installed). The same seed must give
  the same numbers on every platform, which is what these pins check on each CI runner.
* The samplers, through the public simulate_* methods, against closed forms.
* The two deferred 0.8.0 fixes in ActuarialMath.
* ExcelExporter: what the C++ core actually did (the template is never read).
* The pybind11 conversion rules the bindings keep.
"""
from __future__ import annotations

import math
import re
import zipfile
from decimal import Decimal
from xml.etree import ElementTree

import pytest

import riskpy
from riskpy import ActuarialMath, ExcelExporter, FactorModel, MonteCarloSimulator, RateAnalyzer, RiskEngine

core = pytest.importorskip("riskpy.cpp_underwriter")

MASK64 = (1 << 64) - 1
MASK128 = (1 << 128) - 1
CHEAP_MULTIPLIER = 0xDA942042E4DD58B5
SEED_DOMAIN = 0x5249534B50590001


def _splitmix64(state):
    while True:
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        yield z ^ (z >> 31)


class _Pcg64Dxsm:
    """PCG64-DXSM as documented in rust/core/src/rng.rs, in plain Python."""

    def __init__(self, seed):
        sm = _splitmix64(seed ^ SEED_DOMAIN)
        initstate = (next(sm) << 64) | next(sm)
        initseq = (next(sm) << 64) | next(sm)
        self.inc = ((initseq << 1) | 1) & MASK128
        self.state = 0
        self._step()
        self.state = (self.state + initstate) & MASK128
        self._step()

    def _step(self):
        self.state = (self.state * CHEAP_MULTIPLIER + self.inc) & MASK128

    def next_u64(self):
        hi, lo = self.state >> 64, (self.state & MASK64) | 1
        hi ^= hi >> 32
        hi = (hi * CHEAP_MULTIPLIER) & MASK64
        hi ^= hi >> 48
        hi = (hi * lo) & MASK64
        self._step()
        return hi

    def uniform(self):
        return (self.next_u64() >> 11) * 2.0 ** -53


def test_the_python_reference_generator_is_numpys_pcg64dxsm():
    np = pytest.importorskip("numpy")
    ref = _Pcg64Dxsm(12345)
    bg = np.random.PCG64DXSM()
    bg.state = {"bit_generator": "PCG64DXSM", "state": {"state": ref.state, "inc": ref.inc},
                "has_uint32": 0, "uinteger": 0}
    assert [ref.next_u64() for _ in range(64)] == [int(x) for x in bg.random_raw(64)]


@pytest.mark.parametrize("seed", [1, 7, 12345, 2**32 - 1])
def test_seeded_draws_are_the_documented_stream_bit_for_bit(seed):
    # Catastrophe with p = 1 and alpha = 1: each trial takes one uniform for the event test
    # (always true) and one for the size, xm / (1 - u) — plain IEEE division, so the
    # expected values are exact on every platform.
    ref = _Pcg64Dxsm(seed)
    expected = []
    for _ in range(500):
        ref.uniform()
        expected.append(250.0 / (1.0 - ref.uniform()))
    got = MonteCarloSimulator(500, seed).simulate_catastrophe_loss(1.0, 250.0, 1.0)
    assert got == expected


def test_a_seed_repeats_and_seed_zero_does_not():
    for method, args in (("simulate_aggregate_loss", (4.0, 1.0, 0.7)), ("simulate_health_claims", (3.0, 0.5, 2.0, 10.0)),
                         ("simulate_life_portfolio", (500, 0.02, 0.2, 1.0)), ("simulate_economic_path", (1.0, 0.0, 0.3, 4))):
        first = getattr(MonteCarloSimulator(2000, 99), method)(*args)
        assert getattr(MonteCarloSimulator(2000, 99), method)(*args) == first
        assert getattr(MonteCarloSimulator(2000, 100), method)(*args) != first
        assert getattr(MonteCarloSimulator(2000, 0), method)(*args) != getattr(MonteCarloSimulator(2000, 0), method)(*args)


# ---------------------------------------------------------------- the samplers, by closed form


def _chi_square_p(observed, expected):
    stats = pytest.importorskip("scipy.stats")
    keep = [i for i, e in enumerate(expected) if e >= 5]
    stat = sum((observed[i] - expected[i]) ** 2 / expected[i] for i in keep)
    return stats.chi2.sf(stat, len(keep) - 1)


@pytest.mark.parametrize("lam", [0.7, 6.0, 10.0, 45.0])
def test_poisson_counts_follow_the_poisson_pmf(lam):
    stats = pytest.importorskip("scipy.stats")
    # severity exp(0 + 1e-12 Z) is 1 to twelve digits, so each trial is its claim count
    x = MonteCarloSimulator(100_000, 31).simulate_aggregate_loss(lam, 0.0, 1e-12)
    counts = [round(v) for v in x]
    top = int(lam + 12 * math.sqrt(lam) + 12)
    observed = [0] * (top + 1)
    for c in counts:
        observed[min(c, top)] += 1
    expected = [stats.poisson(lam).pmf(k) * len(x) for k in range(top)] + [stats.poisson(lam).sf(top - 1) * len(x)]
    assert _chi_square_p(observed, expected) > 1e-4


@pytest.mark.parametrize("n,q", [(40, 0.1), (20_000, 0.3), (10, 0.95)])
def test_deaths_follow_the_binomial_pmf_without_a_shock(n, q):
    stats = pytest.importorskip("scipy.stats")
    x = MonteCarloSimulator(100_000, 17).simulate_life_portfolio(n, q, 0.0, 1.0)
    dist = stats.binom(n, q)
    lo, hi = int(dist.ppf(1e-6)), int(dist.ppf(1 - 1e-6))
    observed = [0] * (hi - lo + 1)
    for v in x:
        observed[min(max(int(v), lo), hi) - lo] += 1
    expected = [dist.pmf(k) * len(x) for k in range(lo, hi + 1)]
    expected[0] += dist.cdf(lo - 1) * len(x)
    expected[-1] += dist.sf(hi) * len(x)
    assert _chi_square_p(observed, expected) > 1e-4


def test_economic_paths_are_lognormal():
    stats = pytest.importorskip("scipy.stats")
    np = pytest.importorskip("numpy")
    x = np.asarray(MonteCarloSimulator(100_000, 5).simulate_economic_path(100.0, 0.04, 0.3, 6))
    ks = stats.kstest(np.log(x / 100.0), stats.norm(6 * (0.04 - 0.045), 0.3 * math.sqrt(6)).cdf)
    assert ks.pvalue > 1e-4


def test_catastrophe_sizes_are_pareto_and_events_binomial():
    stats = pytest.importorskip("scipy.stats")
    np = pytest.importorskip("numpy")
    x = np.asarray(MonteCarloSimulator(200_000, 8).simulate_catastrophe_loss(0.1, 5e5, 2.2))
    hits = x[x > 0]
    assert stats.binomtest(len(hits), len(x), 0.1).pvalue > 1e-4
    assert stats.kstest(hits, stats.pareto(2.2, scale=5e5).cdf).pvalue > 1e-4


def test_health_claim_counts_are_negative_binomial():
    stats = pytest.importorskip("scipy.stats")
    # claims of size 1 (gamma shape 1e12, scale 1e-12: 1 to six digits), so each trial
    # is its count: NB with mean 6 and variance 6 + 0.8·36, i.e. r = 1/0.8, p = r/(r+6)
    x = MonteCarloSimulator(100_000, 21).simulate_health_claims(6.0, 0.8, 1e12, 1e-12)
    r = 1 / 0.8
    dist = stats.nbinom(r, r / (r + 6.0))
    counts = [round(v) for v in x]
    top = 80
    observed = [0] * (top + 1)
    for c in counts:
        observed[min(c, top)] += 1
    expected = [dist.pmf(k) * len(x) for k in range(top)] + [dist.sf(top - 1) * len(x)]
    assert _chi_square_p(observed, expected) > 1e-4


# ---------------------------------------------------------------- the deferred 0.8.0 fixes


def test_annuities_keep_their_digits_for_tiny_rates():
    # PV = n − n(n+1)/2·r + O(r²); FV = n + n(n−1)/2·r + O(r²). The 0.8.0 core lost about
    # half the digits here ((1 − (1+r)^−n)/r with r = 1e-10 leaves ~7 correct).
    for r in (1e-10, -1e-10, 1e-14, 5e-324):
        n = 30
        pv = ActuarialMath.present_value(r, n, 1.0)
        fv = ActuarialMath.future_value(r, n, 1.0)
        assert pv == pytest.approx(n - n * (n + 1) / 2 * r, rel=1e-14, abs=0)
        assert fv == pytest.approx(n + n * (n - 1) / 2 * r, rel=1e-14, abs=0)
    assert ActuarialMath.present_value(0.05, 10, 100.0) == pytest.approx(772.1734929184818, rel=1e-15)
    assert ActuarialMath.future_value(0.05, 10, 100.0) == pytest.approx(1257.789253554883, rel=1e-15)
    assert ActuarialMath.present_value(0.05, 0, 100.0) == 0.0


@pytest.mark.parametrize("call", [
    lambda: ActuarialMath.present_value(0.05, -1, 100.0),
    lambda: ActuarialMath.future_value(0.05, -10, 100.0),
    lambda: ActuarialMath.present_value(0.05, 5, math.nan),
    lambda: ActuarialMath.future_value(0.05, 5, math.inf),
    lambda: ActuarialMath.present_value(0.05, 5, -math.inf),
])
def test_annuities_refuse_negative_periods_and_non_finite_payments(call):
    with pytest.raises(ValueError):
        call()


# ---------------------------------------------------------------- ExcelExporter


def _cells(path):
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        strings = [si.findtext("m:t", default="", namespaces=ns)
                   for si in ElementTree.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        sheet = ElementTree.fromstring(z.read("xl/worksheets/sheet1.xml"))
        book = ElementTree.fromstring(z.read("xl/workbook.xml"))
    out = {}
    for c in sheet.iter("{%s}c" % ns["m"]):
        v = c.findtext("m:v", namespaces=ns)
        kind = c.get("t")
        out[c.get("r")] = strings[int(v)] if kind == "s" else ("error", v) if kind == "e" else float(v)
    return [s.get("name") for s in book.iter("{%s}sheet" % ns["m"])], out


def test_export_writes_labels_values_and_errors_and_never_reads_the_template(tmp_path):
    engine, exporter = RiskEngine(), ExcelExporter(str(tmp_path / "no-such-template.xlsx"))
    exporter.map_column("A", "age", "Age")
    exporter.map_column("C", "state", "State <&>")
    exporter.map_column("B", "premium_output", "Premium")
    exporter.map_column("D", "absent", "Absent")
    engine.attach_exporter(exporter)
    engine.set_math_logic(lambda d: d["age"] * 10.0)
    engine.execute({"age": 30.0, "state": "NY"})
    out = tmp_path / "quote.xlsx"
    engine.export_to_excel(str(out))
    sheets, cells = _cells(out)
    assert sheets == ["Sheet1"]
    assert cells == {"A1": "Age", "B1": "Premium", "C1": "State <&>", "D1": "Absent",
                     "A2": 30.0, "B2": 300.0, "C2": "NY"}
    engine.set_math_logic(lambda d: math.nan)
    engine.execute({"age": 1.0})
    engine.export_to_excel(str(out))
    assert _cells(out)[1]["B2"] == ("error", "#NUM!")


def test_batch_export_fills_missing_premiums_with_zero(tmp_path):
    engine, exporter = RiskEngine(), ExcelExporter("t")
    exporter.map_column("A", "age", "Age")
    exporter.map_column("B", "premium_output", "P")
    engine.attach_exporter(exporter)
    out = tmp_path / "batch.xlsx"
    engine.export_batch_to_excel(str(out), [{"age": 1.0}, {"age": 2.5}, {}], [10.0])
    assert _cells(out)[1] == {"A1": "Age", "B1": "P", "A2": 1.0, "B2": 10.0, "A3": 2.5, "B3": 0.0, "B4": 0.0}


def test_a_bad_column_raises_and_leaves_an_existing_file_alone(tmp_path):
    out = tmp_path / "keep.xlsx"
    out.write_bytes(b"previous contents")
    engine, exporter = RiskEngine(), ExcelExporter("t")
    exporter.map_column("a", "age", "Age")              # OpenXLSX takes upper-case letters only
    engine.attach_exporter(exporter)
    with pytest.raises(RuntimeError, match=re.escape('address "a1" is invalid')):
        engine.export_to_excel(str(out))
    assert out.read_bytes() == b"previous contents"
    assert [p.name for p in tmp_path.iterdir()] == ["keep.xlsx"]      # no temporary file left


def test_control_characters_still_give_a_readable_workbook(tmp_path):
    # OpenXLSX wrote them raw, which no XML parser (Excel, openpyxl) accepts.
    engine, exporter = RiskEngine(), ExcelExporter("t")
    exporter.map_column("A", "note", "Note\x01")
    engine.attach_exporter(exporter)
    engine.execute({"note": "a\x00b\r\n"})
    out = tmp_path / "ctl.xlsx"
    engine.export_to_excel(str(out))
    cells = _cells(out)[1]
    assert cells["A1"] == "Note_x0001_" and cells["A2"] == "a_x0000_b\r\n"


# ---------------------------------------------------------------- the binding keeps pybind11's rules


def test_the_extension_reports_the_package_version():
    assert core.__version__ == riskpy.__version__


def test_argument_conversion_matches_the_cpp_core():
    np = pytest.importorskip("numpy")
    ra = RateAnalyzer()
    assert ra.on_level_factor(np.array([0.1, 0.2])) == pytest.approx(1.32)      # arrays are sequences
    assert ra.on_level_factor((x for x in [0.1])) == pytest.approx(1.1)          # so are generators
    with pytest.raises(TypeError):
        ra.on_level_factor(iter([0.1]))                                           # plain iterators are not
    with pytest.raises(TypeError):
        ActuarialMath.lookup_mortality_rate(30.0)                                 # no float for an int
    assert ActuarialMath.lookup_mortality_rate(Decimal("35.9")) == 0.0025         # int() truncates
    with pytest.raises(TypeError):
        ActuarialMath.lookup_mortality_rate(2**31)                                # outside a C++ int
    with pytest.raises(TypeError):
        MonteCarloSimulator(10, -1)                                               # seed is unsigned
    MonteCarloSimulator(10, 2**32 - 1)
    m = FactorModel(100.0)
    m.add_multiplier(b"state", "NY", 2.0)                                         # bytes are strings
    assert m.calculate({"state": b"NY", "age": True}) == 200.0
    with pytest.raises(TypeError):
        m.calculate({"state": None})

