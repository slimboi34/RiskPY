"""Tests that kill mutants of the 0.7.0 -> 0.8.0 patch which the release's own regression tests
(and the rest of the suite) let survive. Added in 0.9.0, from the machine-driven testing of 0.8.0.

Mutation testing covered the lines changed in v0.8.0: 2,207 mutants. Each test names the mutants it kills
as ``mNNNNN`` ids from that campaign. Reference values were computed with mpmath at 40-50 digits and are
hard-coded, so the tests are fast and need no mpmath.

Seeded golden values: every one here comes from riskpy.mc / riskpy.capital, which draw from NumPy's
Generator in pure Python, so the 0.9.0 Rust core (a new random stream for MonteCarloSimulator and the
aggregate_loss tool) does not change them; none depends on the compiled core's stream. Two values pinned
0.8.0 behaviour that 0.9.0 changes on purpose: LIMITS["capital_cells"] (finding 15) and where verify_chain
places a line without a digest (finding 18).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from riskpy import _special as S
from riskpy.errors import RiskPyError, from_exception


def _code(exc_info) -> str:
    return getattr(exc_info.value, "code", None)


# ---------------------------------------------------------------------------
# riskpy._special


# mpmath (dps 50) Student-t quantiles: (p, df, quantile)
_T_PPF_REFERENCE = [
    (1e-3, 1.0, -318.3088389855504393),          # Cauchy: the bracket has to widen from the normal guess
    (1e-6, 1.0, -318309.88618274348839),
    (0.999, 1.0, 318.3088389855501632),
    (1e-3, 2.0, -22.327124770119875202),
    (0.01, 3.0, -4.5407028585681335202),
    (0.05, 0.5, -41.136000092878270888),
    (0.3, 1e4, -0.52441722834546679058),         # Cornish-Fisher central band
    (0.26, 1e4, -0.64336814663927772272),
    (0.7, 2e4, 0.52440887042053909078),
    (1e-10, 1e12, -6.3613409024700020861),
]


@pytest.mark.parametrize("p, df, expected", _T_PPF_REFERENCE)
def test_t_ppf_matches_mpmath_in_the_heavy_tails_and_the_central_band(p, df, expected):
    # kills m00104 (negated `hi > 709` guard: returned -inf as soon as the bracket widened)
    assert S.t_ppf(p, df) == pytest.approx(expected, rel=5e-15)


# mpmath (dps 50) quantiles in the central band (0.25 < p < 0.75) below df = 1e4, where t_ppf solves on
# P(0 < T < t) rather than on the tail: (p, df, quantile, relative tolerance)
_T_PPF_CENTRAL = [
    (0.3, 3.0, -0.584389727439818706583, 5e-15),
    (0.6, 0.5, 0.3979754267847905873762, 5e-15),
    (0.45, 50.0, -0.1263012399976589657464, 1e-14),
    (0.26, 1.0, -0.9390625058174923000472, 5e-15),
    (0.7, 0.05, 3119.219146032919829847, 1e-14),       # the bracket widens from e^2
    (0.3, 0.01, -768454187044735784213.8, 1e-12),
]


@pytest.mark.parametrize("p, df, expected, rel", _T_PPF_CENTRAL)
def test_t_ppf_central_band_below_df_1e4_matches_mpmath(p, df, expected, rel):
    # kills m00104 (negated `hi > 709` in the central search), m00154, m00215, m00286, m00291, m00300, m00305
    assert S.t_ppf(p, df) == pytest.approx(expected, rel=rel)


def test_t_ppf_central_quantile_beyond_1e154_is_finite():
    # kills m00217 (709 -> 354.5 in the central search)
    value = S.t_ppf(0.3, 0.001)
    assert math.isfinite(value) and value < -1e200
    assert S.t_cdf_vec(np.array([value]), 0.001)[0] == pytest.approx(0.3, rel=1e-12)


@pytest.mark.parametrize("t, expected", [(-0.72, 0.3395030875485291680189), (-0.75, 0.3348325671412551278485),
                                         (0.72, 0.6604969124514708319811), (-0.70, 0.3427003312603832669991)])
def test_t_cdf_vec_just_beyond_root_df_matches_mpmath(t, expected):
    # kills m00174: |t| slightly above sqrt(df) at df = 0.5 takes the reflected beta, which needs y = 1 - x
    assert S.t_cdf_vec(np.array([t]), 0.5)[0] == pytest.approx(expected, rel=2e-15)


def test_t_ppf_tiny_df_quantile_between_1e154_and_the_largest_double_is_finite():
    # kills m00217 (709 -> 354.5): a quantile of 3.96e168 is finite, not -inf
    value = S.t_ppf(0.01, 0.01)
    assert math.isfinite(value)
    assert value == pytest.approx(-3.9604401371524413568e+168, rel=1e-12)


def test_t_ppf_at_astronomical_df_is_the_normal_quantile_in_the_tails_too():
    # kills m00052 (or -> and: the tail at df = 1e300 went to the root finder)
    for p in (1e-3, 1e-12, 0.999):
        assert S.t_ppf(p, 1e300) == pytest.approx(S.norm_ppf(p), rel=1e-15)


@pytest.mark.parametrize("df", [0.0, -1.0, float("nan")])
def test_t_ppf_and_t_cdf_vec_refuse_a_bad_df_by_name(df):
    # kills m00020, m00100 (t_ppf guard) and m00032, m00065, m00117 (t_cdf_vec guard)
    for call in (lambda: S.t_ppf(0.3, df), lambda: S.t_cdf_vec(np.array([0.3, -2.0]), df)):
        with pytest.raises(ValueError, match="df must be > 0") as info:
            call()
        assert from_exception(info.value)["param"] == "df"


def test_t_cdf_vec_with_infinite_df_is_the_normal_cdf():
    # kills m00037 (the math.isinf(df) guard in _hill_z_vec) and m00066
    x = np.array([-3.0, -1.0, 0.0, 0.5, 2.0])
    got = S.t_cdf_vec(x, math.inf)
    # a few ulps: the vector path's NumPy ufuncs and the scalar math library round differently on Linux/Windows
    assert np.allclose(got, [S.norm_cdf(v) for v in x], rtol=1e-14, atol=0)


def test_vectorised_normal_functions_on_more_than_one_block():
    # kills m00070, m00071, m00131 (norm_cdf_vec block path) and the norm_ppf_vec block path:
    # capital_allocation at its default 200,000 trials sends arrays this large through them
    n = (1 << 16) + 1001
    x = np.linspace(-8.0, 8.0, n)
    got = S.norm_cdf_vec(x)
    want = np.array([S.norm_cdf(v) for v in x])
    assert got.shape == x.shape
    assert np.array_equal(got, want)
    p = np.linspace(1e-6, 1.0 - 1e-6, n)
    assert np.array_equal(S.norm_ppf_vec(p), np.array([S.norm_ppf(v) for v in p]))


def test_betainc_vec_endpoints_and_more_than_one_block():
    # kills m00113, m00115, m00223, m00225 (betainc_vec x = 0 / x = 1 and the block loop)
    x = np.array([0.0, 1.0, 0.25, 0.5, 0.0, 1.0])
    got = S.betainc_vec(2.0, 3.0, x)
    assert got.tolist() == pytest.approx([0.0, 1.0, 0.26171875, 0.6875, 0.0, 1.0], abs=1e-15)
    n = (1 << 16) + 7
    big = np.linspace(0.0, 1.0, n)
    vals = S.betainc_vec(2.0, 3.0, big)
    assert vals[0] == 0.0 and vals[-1] == 1.0
    # I_x(2, 3) = 6x^2 - 8x^3 + 3x^4 exactly
    assert np.allclose(vals, 6 * big**2 - 8 * big**3 + 3 * big**4, rtol=1e-13, atol=1e-15)


# ---------------------------------------------------------------------------
# riskpy._validate (through quant, which uses it)


def test_a_one_element_array_is_not_a_number_but_a_0d_array_is():
    # kills m00339, m00381: a 1-D NumPy array must be refused as a scalar argument
    from riskpy import quant

    with pytest.raises(RiskPyError) as info:
        quant.black_scholes(S=np.array([100.0]), K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert _code(info) == "RP101"
    ok = quant.black_scholes(S=np.array(100.0), K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert ok == pytest.approx(quant.black_scholes(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2))


def test_seed_zero_one_trial_and_a_seed_list_with_zero_are_valid():
    # kills m00366 (minimum 1 -> 2), m00389 / m00394 (seed 0 refused), m00397 / m00401 (seed list with 0)
    from riskpy import quant

    a = quant.gbm_paths(100.0, 0.05, 0.2, 1.0, steps=1, trials=1, seed=0)
    b = quant.gbm_paths(100.0, 0.05, 0.2, 1.0, steps=1, trials=1, seed=[0, 1])
    assert np.asarray(a).shape[0] in (1, 2) and np.all(np.isfinite(np.asarray(a)))
    assert np.all(np.isfinite(np.asarray(b)))


def test_whole_count_messages_name_the_minimum():
    # kills m00396, m00399, m00400 (the message chosen by `minimum == 1`)
    from riskpy._validate import _check_trials

    with pytest.raises(RiskPyError, match="steps must be a positive integer"):
        _check_trials("steps", 0)
    with pytest.raises(RiskPyError, match=r"nodes must be an integer >= 2"):
        _check_trials("nodes", 1, minimum=2)


# ---------------------------------------------------------------------------
# riskpy.audit

from riskpy import audit, tools  # noqa: E402


def _nest(n, kind="list"):
    value = 1.0
    for _ in range(n):
        value = [value] if kind == "list" else {"k": value}
    return value


def _chain(entries):
    """Entries with seq, prev and digest filled in: a valid chain of crafted entries."""
    prev, out = audit.GENESIS, []
    for k, entry in enumerate(entries, 1):
        entry = {"seq": k, **entry, "prev": prev}
        entry["digest"] = audit.digest(entry)
        prev = entry["digest"]
        out.append(entry)
    return out


def _write(path, entries, extra_lines=()):
    with open(path, "w", encoding="ascii") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
        for line in extra_lines:
            fh.write(line + "\n")


_BS = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}


@pytest.mark.parametrize("name", ["rel_tol", "abs_tol"])
@pytest.mark.parametrize("bad", [-1.0, -1e-12, True, "1e-9", None, float("nan")])
def test_replay_refuses_bad_tolerances(name, bad):
    # kills m00428, m00673, m00796, m00797
    with pytest.raises(RiskPyError) as info:
        audit.replay([], **{name: bad})
    assert _code(info) == "RP102" and info.value.param == name


def test_replay_accepts_zero_tolerances():
    # kills m00961 (value > 0 would refuse an exact replay)
    entries = _chain([{"tool": "expected_loss", "arguments": {"pd": 0.02, "lgd": 0.45, "ead": 1000.0},
                       "result": tools.call("expected_loss", {"pd": 0.02, "lgd": 0.45, "ead": 1000.0})}])
    outcome = audit.replay(entries, rel_tol=0, abs_tol=0.0)
    assert outcome.ok and outcome.matched == 1


@pytest.mark.parametrize("kind", ["list", "dict"])
def test_arguments_are_stored_exactly_up_to_255_levels_and_as_a_placeholder_beyond(kind):
    # kills m00442, m00443, m00569, m00570, m00982, m00983, m01017 (the 256-level storage limit)
    ledger = audit.Ledger(None)
    kept = ledger.append("t", {"a": _nest(255, kind)}, result=1)
    cut = ledger.append("t", {"a": _nest(256, kind)}, result=1)
    assert kept["arguments"] == {"a": _nest(255, kind)}
    assert cut["arguments"]["$riskpy"] == "unserialisable"


def test_a_crafted_argument_deeper_than_258_levels_is_skipped_not_rebuilt():
    # kills m00467, m00591, m00592, m00843, m00844, m00984, m00985 (the decode depth limit)
    def entry(n):
        return {"tool": "black_scholes", "arguments": {**_BS, "S": _nest(n)},
                "error": {"code": "RP404", "message": "S must be a number", "param": "S"}}

    shallow = audit.replay(_chain([entry(257)]))
    assert shallow.matched == 1 and not shallow.skipped
    deep = audit.replay(_chain([entry(258)]))
    assert deep.matched == 0 and "nested too deeply to rebuild" in deep.skipped[0]["reason"]


def test_crafted_nested_dicts_at_the_decode_limit():
    # kills m00986, m00987 (dict levels count towards the decode limit)
    def entry(n):
        return {"tool": "black_scholes", "arguments": {**_BS, "S": _nest(n, "dict")},
                "error": {"code": "RP404", "message": "S must be a number", "param": "S"}}

    assert audit.replay(_chain([entry(257)])).matched == 1
    assert audit.replay(_chain([entry(258)])).skipped


def test_an_argument_that_could_not_be_stored_exactly_is_skipped_on_replay():
    # kills m00477, m00601 (the "unrepresentable" tag) and m00738 (its reason is kept)
    with audit.record() as ledger:
        tools.call("black_scholes", {**_BS, "S": object()}, errors="return")
    assert ledger.entries[0]["arguments"]["S"]["$riskpy"] == "unrepresentable"
    outcome = audit.replay(ledger)
    assert outcome.matched == 0 and "could not be stored exactly" in outcome.skipped[0]["reason"]
    nested = _chain([{"tool": "black_scholes", "arguments": {"$riskpy": "object", "value": {
        **_BS, "S": {"$riskpy": "unrepresentable", "type": "X", "repr": "x"}}},
        "error": {"code": "RP404", "message": "m", "param": "S"}}])
    assert "could not be stored exactly" in audit.replay(nested).skipped[0]["reason"]


def test_a_long_repr_is_clipped_to_200_characters():
    # kills m00841, m00842
    class Long:
        def __repr__(self):
            return "L" * 500

    stored = audit.Ledger(None).append("t", {"a": Long()}, result=1)["arguments"]["a"]
    assert stored["repr"] == "L" * 200 + "…"


def test_a_broken_repr_is_stored_as_a_type_name():
    # kills m00709
    class Broken:
        def __repr__(self):
            raise RuntimeError("no repr")

    stored = audit.Ledger(None).append("t", {"a": Broken()}, result=1)["arguments"]["a"]
    assert stored["repr"] == "<Broken object>"


def test_integers_wider_than_2000_bits_are_stored_in_hex():
    # kills m00444, m00445, m00836
    ledger = audit.Ledger(None)
    assert ledger.append("t", {"a": 2 ** 1999}, result=1)["arguments"]["a"] == 2 ** 1999
    wide = ledger.append("t", {"a": 2 ** 2000}, result=1)["arguments"]["a"]
    assert wide == {"$riskpy": "int", "hex": hex(2 ** 2000)}


def test_sets_and_frozensets_round_trip_as_themselves():
    # kills m00933
    assert type(audit._decode(audit._encode(frozenset({1, 2})))) is frozenset
    assert type(audit._decode(audit._encode({1, 2}))) is set


def test_a_numpy_tag_without_numpy_is_a_cannot_rebuild(monkeypatch):
    # kills m00934
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(audit._CannotRebuild, match="NumPy is not installed"):
        audit._decode({"$riskpy": "numpy", "dtype": "float64", "value": 1.0})


@pytest.mark.parametrize("dtype", ["<M8[s]", "[('a', '<f8')]"])
def test_a_crafted_array_of_an_unsupported_dtype_is_malformed(dtype):
    # kills m00733, m00852, m00853
    value = {"$riskpy": "ndarray", "dtype": dtype, "shape": [1], "data": [1]}
    if dtype.startswith("["):
        value["dtype"] = [("a", "<f8")]
    with pytest.raises(audit._CannotRebuild, match="malformed"):
        audit._decode(value)


def test_digest_of_an_entry_nested_too_deeply_is_a_coded_error():
    # kills m00739
    deep = {}
    node = deep
    for _ in range(100_000):
        node["k"] = {}
        node = node["k"]
    with pytest.raises(RiskPyError) as info:
        audit.digest(deep)
    assert _code(info) == "RP100"


def test_error_fields_keep_short_json_scalars_and_clip_text():
    # kills m00496, m00498, m00500, m00635, m00636, m00638, m00640, m00771, m00879, m00880,
    # m00996, m00997, m01040, m00708
    ledger = audit.Ledger(None)
    e = ledger.append("t", {}, error={"code": "RP101", "message": "m", "param": None})["error"]
    assert e["param"] is None
    e = ledger.append("t", {}, error={"code": 7, "message": 1.5, "param": True})["error"]
    assert e == {"code": 7, "message": 1.5, "param": True}
    e = ledger.append("t", {}, error={"code": 2 ** 64 - 1, "message": 2 ** 64, "param": False})["error"]
    assert e["code"] == 2 ** 64 - 1 and e["message"] == str(2 ** 64) and e["param"] is False
    e = ledger.append("t", {}, error={"code": "X" * 45, "message": "y" * 1000, "param": ["p"] * 100})["error"]
    assert e["code"] == "X" * 40 + "…"
    assert e["message"] == "y" * 1000                      # exactly at the limit: not clipped
    assert e["param"].endswith("…") and len(e["param"]) == 81


def test_a_line_that_is_json_but_not_an_object_is_named(tmp_path):
    # kills m00483, m00618
    path = tmp_path / "l.jsonl"
    _write(path, _chain([{"tool": "t", "arguments": {}, "result": 1}]), ["[1, 2]"])
    check = audit.verify_chain(str(path))
    assert not check.ok and check.broken_at == 2 and "is not a JSON object" in check.reason


def test_a_duplicated_key_is_named_in_the_damage_report(tmp_path):
    # kills m00740, m00858
    path = tmp_path / "l.jsonl"
    good = _chain([{"tool": "t", "arguments": {}, "result": 1}])[0]
    line = json.dumps(good)
    line = line[:-1] + ', "result": 2}'
    path.write_text(line + "\n", encoding="ascii")
    check = audit.verify_chain(str(path))
    assert not check.ok and "'result'" in check.reason


def test_blank_lines_between_entries_are_ignored(tmp_path):
    # kills m00953
    path = tmp_path / "l.jsonl"
    entries = _chain([{"tool": "t", "arguments": {}, "result": k} for k in range(3)])
    path.write_text("\n\n".join(json.dumps(e) for e in entries) + "\n\n", encoding="ascii")
    assert audit.verify_chain(str(path)).ok


def test_each_kind_of_break_is_named_and_placed():
    # kills m00649, m00651, m00653, m00655, m00777, m00779, m00781, m00783, m00882, m00954, m00884
    base = _chain([{"tool": "t", "arguments": {}, "result": k} for k in range(3)])
    no_digest = [dict(e) for e in base]
    del no_digest[1]["digest"]
    r = audit.verify_chain(no_digest)
    assert (r.ok, r.broken_at, r.reason) == (False, 2, "the entry has no digest")
    renumbered = [dict(e) for e in base]
    del renumbered[1]["digest"]
    renumbered[1]["seq"] = 5
    # its position, not the seq it claims (0.9.0: finding 18 of the 0.8.0 machine testing; 0.8.0 reported 5)
    assert audit.verify_chain(renumbered).broken_at == 2
    text_seq = [dict(e) for e in base]
    del text_seq[1]["digest"]
    text_seq[1]["seq"] = "two"
    assert audit.verify_chain(text_seq).broken_at == 2            # ... and its position when that is not a number
    gap = [base[0], base[2]]
    r = audit.verify_chain(gap)
    assert not r.ok and r.broken_at == 2 and "sequence number" in r.reason
    relinked = [dict(e) for e in base]
    relinked[1]["prev"] = "f" * 64
    r = audit.verify_chain(relinked)
    assert not r.ok and r.broken_at == 2 and "does not link" in r.reason
    edited = [dict(e) for e in base]
    edited[1]["result"] = 99
    r = audit.verify_chain(edited)
    assert not r.ok and r.broken_at == 2 and "content was changed" in r.reason
    nan = [dict(e) for e in base]
    nan[1]["result"] = float("nan")
    r = audit.verify_chain(nan)
    assert not r.ok and r.broken_at == 2 and "not finite JSON" in r.reason


def test_an_in_memory_ledger_numbers_and_links_its_entries():
    # kills m00762
    ledger = audit.Ledger(None)
    ledger.append("t", {}, result=1)
    ledger.append("t", {}, result=2)
    assert [e["seq"] for e in ledger.entries] == [1, 2]
    assert audit.verify_chain(ledger).ok


def test_a_ledger_file_that_does_not_exist_yet_has_the_genesis_head(tmp_path):
    # kills m00870
    ledger = audit.Ledger(str(tmp_path / "not-yet.jsonl"))
    assert ledger.head == audit.GENESIS and len(ledger) == 0


def test_an_expected_head_does_not_hide_a_break_inside_the_chain():
    # kills m00506
    entries = _chain([{"tool": "t", "arguments": {}, "result": k} for k in range(3)])
    head = entries[-1]["digest"]
    entries[1] = dict(entries[1], result=42)
    r = audit.verify_chain(entries, expected_head=head)
    assert not r.ok and r.broken_at == 2 and "content was changed" in r.reason
    r = audit.verify_chain(entries, expected_entries=3)
    assert not r.ok and r.broken_at == 2 and "content was changed" in r.reason


def test_expected_entries_zero_and_entries_appended_after_the_end():
    # kills m00888, m00957, m01005, m01006
    assert audit.verify_chain([], expected_entries=0).ok
    entries = _chain([{"tool": "t", "arguments": {}, "result": k} for k in range(4)])
    r = audit.verify_chain(entries, expected_entries=2)
    assert not r.ok and r.broken_at == 3 and "2 entries were added" in r.reason


def test_replay_skips_an_entry_with_neither_result_nor_error():
    # kills m00521, m00677
    outcome = audit.replay(_chain([{"tool": "expected_loss", "arguments": {"pd": 0.1, "lgd": 0.5, "ead": 1.0}}]))
    assert outcome.matched == 0 and "neither a result nor an error" in outcome.skipped[0]["reason"]


def test_replay_skips_a_tool_this_installation_cannot_run(monkeypatch):
    # kills m00529, m00685
    entries = _chain([{"tool": "nearest_correlation", "arguments": {"matrix": [[1.0, 0.5], [0.5, 1.0]]},
                       "result": {"changed": False}}])
    monkeypatch.setattr(tools, "_extra_installed", lambda extra: False)
    outcome = audit.replay(entries)
    assert outcome.matched == 0 and outcome.skipped[0]["reason"].startswith("the tool needs")


def test_replay_skips_a_simulation_run_with_a_fresh_seed():
    # kills m00531, m00687, m00810
    args = {"expected_frequency": 2.0, "severity_mean": 10.0, "severity_sd": 5.0, "trials": 2000, "seed": 0}
    with audit.record() as ledger:
        tools.call("aggregate_loss", args)
    outcome = audit.replay(ledger)
    assert outcome.matched == 0 and not outcome.drifted
    assert "seed=0" in outcome.skipped[0]["reason"]


def test_a_refusal_that_now_succeeds_drifts_to_a_result():
    # kills m01008
    entries = _chain([{"tool": "expected_loss", "arguments": {"pd": 0.1, "lgd": 0.5, "ead": 1.0},
                       "error": {"code": "RP101", "message": "m", "param": "pd"}}])
    d = audit.replay(entries).drifted[0]["differences"][0]
    assert d["path"] == "error.code" and d["replayed"] == "a result"


def test_the_replay_summary_of_a_full_replay():
    # kills m00889, m00960
    entries = _chain([{"tool": "expected_loss", "arguments": {"pd": 0.1, "lgd": 0.5, "ead": 1.0},
                       "result": tools.call("expected_loss", {"pd": 0.1, "lgd": 0.5, "ead": 1.0})}])
    text = audit.replay(entries).summary()
    assert "1 matched" in text and "most recent" not in text


def _el_entries(scale):
    args = {"pd": 0.02, "lgd": 0.45, "ead": 1000.0}
    value = tools.call("expected_loss", args)
    return _chain([{"tool": "expected_loss", "arguments": args, "result": value * scale}])


@pytest.mark.parametrize("scale, drifted", [(1 + 5e-9, 1), (1 + 5e-10, 0)])
def test_report_and_status_checks_use_a_relative_tolerance_of_1e9(tmp_path, monkeypatch, scale, drifted):
    # kills m00908, m00909, m00919, m00920
    path = tmp_path / "l.jsonl"
    _write(path, _el_entries(scale))
    assert f"{1 - drifted} matched, {drifted} drifted" in audit.report(str(path), check=True)
    monkeypatch.setenv(audit.ENV, str(path))
    assert len(audit.status(check=True)["replay"]["drifted"]) == drifted


@pytest.mark.parametrize("recorded, drifted", [(5e-12, 1), (5e-13, 0)])
def test_report_and_status_checks_use_an_absolute_tolerance_of_1e12(tmp_path, monkeypatch, recorded, drifted):
    # kills m00910, m00911, m00921, m00922
    args = {"pd": 0.0, "lgd": 0.45, "ead": 1000.0}
    assert tools.call("expected_loss", args) == 0.0
    path = tmp_path / "l.jsonl"
    _write(path, _chain([{"tool": "expected_loss", "arguments": args, "result": recorded}]))
    assert f"{1 - drifted} matched, {drifted} drifted" in audit.report(str(path), check=True)
    monkeypatch.setenv(audit.ENV, str(path))
    assert len(audit.status(check=True)["replay"]["drifted"]) == drifted


def test_status_replay_window_defaults_and_bounds(tmp_path, monkeypatch):
    # kills m00552, m00553, m00913, m00970, m00975, m01054
    import inspect

    params = inspect.signature(audit.status).parameters
    assert params["replay_last"].default == 50 and params["replay_seconds"].default == 10.0
    path = tmp_path / "l.jsonl"
    entries = _chain([{"tool": "expected_loss", "arguments": {"pd": 0.01 * k, "lgd": 0.5, "ead": 1.0},
                       "result": tools.call("expected_loss", {"pd": 0.01 * k, "lgd": 0.5, "ead": 1.0})}
                      for k in range(1, 4)])
    _write(path, entries)
    monkeypatch.setenv(audit.ENV, str(path))
    one = audit.status(check=True, replay_last=1)
    assert one["replay"]["replayed"] == 1
    assert "most recent 1 of 3 entries" in one["how"] and "for about 10 s" in one["how"]
    every = audit.status(check=True, replay_last=None)
    assert every["replay"]["replayed"] == 3 and every["replay"]["matched"] == 3


def test_write_failure_reasons_are_counted_and_clipped(monkeypatch):
    # kills m01000, m01001, m01011, m01049
    def boom(self, *a, **k):
        raise RuntimeError("y" * 500)

    monkeypatch.setattr(audit.Ledger, "append", boom)
    with audit.record() as ledger:
        tools.call("expected_loss", {"pd": 0.1, "lgd": 0.5, "ead": 1.0})
        once = audit._failure_text(ledger)
        tools.call("expected_loss", {"pd": 0.1, "lgd": 0.5, "ead": 1.0})
    (reason,) = ledger.failure_reasons
    assert len(reason) == 201 and reason.endswith("…") and reason.startswith("RuntimeError: yyy")
    assert "×" not in once and f"{reason} ×2" in audit._failure_text(ledger)


def test_a_tool_name_that_is_not_a_name_is_recorded_as_a_repr_of_80_characters():
    # kills m00950, m00951
    name = "é" * 300
    with audit.record() as ledger:
        tools.call(name, {}, errors="return")
    assert ledger.entries[0]["tool"] == repr(name[:200])[:80]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits; Windows reports 0o666 for any writable file")
def test_a_new_ledger_file_gets_the_usual_permissions(tmp_path):
    # kills m00946, m00947 (0o666 before the umask)
    old = os.umask(0o022)
    try:
        path = tmp_path / "l.jsonl"
        audit.Ledger(str(path)).append("t", {}, result=1)
        assert (os.stat(path).st_mode & 0o777) == 0o644
    finally:
        os.umask(old)


def test_a_file_of_only_damaged_lines_chains_from_the_last_line(tmp_path):
    # kills m00867, m00868
    path = tmp_path / "l.jsonl"
    path.write_bytes(b"not json\n")
    entry = audit.Ledger(str(path)).append("t", {}, result=1)
    assert entry["seq"] == 1 and entry["prev"] == hashlib.sha256(b"not json").hexdigest()


def test_a_torn_tail_read_once_still_gets_its_newline_back(tmp_path):
    # kills m00995 (the cached "ends with a newline" flag)
    path = tmp_path / "l.jsonl"
    good = _chain([{"tool": "t", "arguments": {}, "result": 1}])
    path.write_text(json.dumps(good[0]) + "\n" + '{"seq": 2, "tor', encoding="ascii")
    ledger = audit.Ledger(str(path))
    ledger.head                       # reads and caches the tail
    ledger.append("t", {}, result=2)
    last = path.read_bytes().splitlines()[-1]
    assert json.loads(last)["seq"] == 2


def test_a_line_whose_end_falls_one_byte_into_a_block_is_read_whole(tmp_path):
    # kills m00749 (the backwards reader must read the file's first byte)
    path = tmp_path / "l.jsonl"
    pad = 65_000
    for _ in range(5):
        path.unlink(missing_ok=True)
        audit.Ledger(str(path)).append("t", {"pad": "x" * pad}, result=1)
        size = path.stat().st_size
        if size == (1 << 16) + 1:
            break
        pad += (1 << 16) + 1 - size
    assert path.stat().st_size == (1 << 16) + 1
    entry = audit.Ledger(str(path)).append("t", {}, result=2)
    assert entry["seq"] == 2 and audit.verify_chain(str(path)).ok


def test_chain_link_bounds_on_seq():
    # kills m00563, m00565, m00566, m00747, m00748, m00860
    d = "a" * 64
    assert audit._chain_link(json.dumps({"seq": 0, "digest": d}).encode()) == (0, d)
    assert audit._chain_link(json.dumps({"seq": 2 ** 53 - 1, "digest": d}).encode()) == (2 ** 53 - 1, d)
    assert audit._chain_link(json.dumps({"seq": 2 ** 53, "digest": d}).encode()) is None


def test_report_table_counts_and_fences(tmp_path):
    # kills m00545, m00546, m00821, m00822, m00826, m00904/m00905, m00907, m00969
    path = tmp_path / "l.jsonl"
    with audit.record(str(path), note="the note"):
        tools.call("expected_loss", {"pd": 0.1, "lgd": 0.5, "ead": 1.0})
        tools.call("expected_loss", {"pd": -1, "lgd": 0.5, "ead": 1.0}, errors="return")
    text = audit.report(str(path))
    assert "(1 succeeded, 1 refused)" in text
    assert "- **Period:** " in text and "- **Note:** the note" in text
    rows = [line for line in text.splitlines() if line.startswith("| ") and "expected_loss" in line]
    assert rows[0].startswith("| 1 |") and rows[0].split("|")[4].strip() == "ok"
    assert rows[1].startswith("| 2 |") and rows[1].split("|")[4].strip().startswith("refused: RP")
    assert "Refused — **RP" in text
    fences = [line for line in text.splitlines() if line.startswith("``")]
    assert fences and all(f in ("```json", "```", "```bash") for f in fences)


class _Pipe:
    def __init__(self, exc):
        self.exc = exc

    def write(self, text):
        raise self.exc

    def flush(self):
        pass

    def fileno(self):
        import io
        raise io.UnsupportedOperation("no fileno")


def test_the_cli_stops_quietly_on_a_closed_pipe_and_codes_other_os_errors(tmp_path, monkeypatch, capsys):
    # kills m00832, m00833, m00834, m00924, m00925, m00926, m00979, m00980
    import errno

    path = tmp_path / "l.jsonl"
    _write(path, _chain([{"tool": "t", "arguments": {}, "result": 1}]))
    monkeypatch.setattr(sys, "stdout", _Pipe(BrokenPipeError(errno.EPIPE, "Broken pipe")))
    assert audit.main(["report", str(path)]) == 0
    monkeypatch.setattr(sys, "stdout", _Pipe(OSError(errno.EIO, "I/O error")))
    assert audit.main(["report", str(path)]) == 1
    assert "RP900" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# riskpy.tools: limits, the meta-tools' work budget, sensitivity, the CLI trace


def _call(name, arguments):
    return tools.call(name, arguments, errors="return")


def _err(out):
    assert isinstance(out, dict) and set(out) == {"error"}, out
    return out["error"]


def test_limits_are_the_documented_values():
    # kills m01982, m01983, m01986, m01987 (and pins every other LIMITS value)
    assert tools.LIMITS == {
        "trials": 1_000_000, "simulated_claims": 100_000_000, "periods": 100_000, "triangle": 300,
        "list": 1_000, "matrix": 100, "capital_cells": 5_000_000, "scenarios": 100, "string": 200,
        "integer": 2 ** 31 - 1, "depth": 32, "meta_seconds": 20,
    }


def test_names_and_labels_of_exactly_200_characters_are_not_refused_as_too_long():
    # kills m02034, m02067, m02076
    assert _err(_call("x" * 200, {}))["code"] == "RP401"
    assert _err(_call("x" * 201, {}))["code"] == "RP405"
    assert _err(_call("black_scholes", {**_BS, "y" * 200: 1}))["code"] == "RP402"
    assert _err(_call("black_scholes", {**_BS, "y" * 201: 1}))["code"] == "RP405"
    ok = _call("scenario_table", {"tool": "black_scholes", "arguments": _BS, "scenarios": {"l" * 200: {"r": 0.06}}})
    assert "error" not in ok and "l" * 200 in ok["scenarios"]
    long = _call("scenario_table", {"tool": "black_scholes", "arguments": _BS, "scenarios": {"l" * 201: {"r": 0.06}}})
    assert _err(long)["code"] == "RP405"


def test_an_error_param_of_exactly_80_characters_is_not_clipped():
    # kills m02030
    assert _err(_call("black_scholes", {**_BS, "z" * 80: 1}))["param"] == "z" * 80
    assert _err(_call("black_scholes", {**_BS, "z" * 81: 1}))["param"] == "z" * 77 + "..."


def test_five_unknown_names_are_listed_without_an_and_more():
    # kills m02077
    message = _err(_call("black_scholes", {**_BS, **{f"a{k}": 1 for k in range(1, 6)}}))["message"]
    assert "'a5']" in message and "more" not in message


def test_synonym_suggestions_cover_the_first_five_unknown_names():
    # kills m02149, m02150
    extra = {"dividend": 1, "expiry": 1, "risk_free": 1, "spot": 1, "strike": 1, "vol": 1}
    hint = _err(_call("black_scholes", {**_BS, **extra}))["hint"]
    assert "'strike' → 'K'" in hint and "'vol'" not in hint


def test_unknown_tool_suggestions_are_at_most_three_close_matches():
    # kills m02110, m02111, m02113
    assert "Did you mean 'capital_aggregate' or 'capital_allocation' or 'basel_irb_capital'?" in \
        _err(_call("capital_", {}))["hint"]
    assert "Did you mean 'annuity_certain' or 'life_annuity_due' or 'temporary_annuity_due'?" in \
        _err(_call("annuity", {}))["hint"]
    assert "Did you mean 'black_scholes' or 'mack_chain_ladder'?" in _err(_call("black_schole", {}))["hint"]


def test_a_meta_tool_cannot_drive_another_and_says_why():
    # kills m02117
    message = _err(_call("sensitivity", {"tool": "scenario_table", "arguments": {}, "parameter": "x"}))["message"]
    assert "drives other calculators itself" in message
    message = _err(_call("sensitivity", {"tool": "riskpy_status", "arguments": {}, "parameter": "x"}))["message"]
    assert "is an introspection tool" in message


def test_arguments_nested_32_levels_pass_the_depth_check_and_33_do_not():
    # kills m02074
    def run(n):
        return _err(_call("scenario_table", {"tool": "black_scholes", "arguments": {**_BS, "S": _nest(n)},
                                             "scenarios": {"a": {"r": 0.06}}}))["code"]
    assert run(31) == "RP404"          # through the depth check; black_scholes refuses the list
    assert run(32) == "RP405"


_BOND = {"face": 100.0, "coupon": 0.05, "ytm": 0.05}


def _budget_refused(tool, base, scenarios, needle):
    out = _call("scenario_table", {"tool": tool, "arguments": base, "scenarios": scenarios})
    error = _err(out)
    assert error["code"] == "RP405" and needle in error["message"], error["message"]


@pytest.mark.parametrize("tool, base, needle", [
    ("bond_analytics", {**_BOND, "maturity": 100.0, "frequency": 1000}, "maturity × frequency"),
    ("cds_par_spread", {"hazard": 0.02, "recovery": 0.4, "r": 0.03, "maturity": 100.0, "frequency": 1000},
     "maturity × frequency"),
    ("bootstrap_yield_curve", {"par_rates": [0.03] * 10, "maturities": [float(k) for k in range(1, 11)],
                               "frequency": 1000}, "knots × longest maturity"),
    ("chain_ladder", {"triangle": [[1.0] * 300 for _ in range(300)]}, "triangle"),
    ("capital_aggregate", {"standalone": [1.0] * 100,
                           "correlation": [[1.0 if i == j else 0.0 for j in range(100)] for i in range(100)]},
     "100 risks"),
    ("nearest_correlation", {"matrix": [[1.0 if i == j else 0.0 for j in range(100)] for i in range(100)]},
     "100 risks"),
    ("value_at_risk", {"returns": [0.01 * ((k % 7) - 3) for k in range(1000)]}, "1000 returns"),
])
def test_every_sized_calculator_has_a_work_budget(tool, base, needle):
    # kills m01926-m01930, m01961-m01966, m02015, m02119-m02122: each estimator must count
    scenario = {"again": {}} if tool not in ("chain_ladder",) else {"tail": {"tail": 1.01}}
    _budget_refused(tool, base, scenario, needle)


def test_a_low_frequency_simulation_is_budgeted_by_its_trials():
    # kills m01959, m02052 (trials / LIMITS / 20 per run: 21 runs of 960,000 trials is 1.008 calls)
    base = {"expected_frequency": 0.001, "severity_mean": 10.0, "severity_sd": 5.0, "trials": 960_000, "seed": 1}
    _budget_refused("aggregate_loss", base, {f"s{k}": {"seed": k + 2} for k in range(20)}, "trials")


def test_eleven_low_frequency_runs_fit_the_budget():
    # kills m02053
    base = {"expected_frequency": 0.001, "severity_mean": 10.0, "severity_sd": 5.0, "trials": 1_000_000, "seed": 1}
    out = _call("scenario_table", {"tool": "aggregate_loss", "arguments": base,
                                   "scenarios": {f"s{k}": {"seed": k + 2} for k in range(10)}})
    assert "error" not in out and out["failed"] == 0


def test_the_budget_counts_default_arguments():
    # kills m02088: bond_analytics' default frequency (2) makes 30,000 years 60,000 periods a run
    _budget_refused("bond_analytics", {**_BOND, "maturity": 30_000.0}, {"up": {"ytm": 0.06}}, "maturity × frequency")


def test_a_refused_scenario_cannot_buy_budget_for_the_others():
    # kills m02049 (a negative size must count as zero, not as negative work)
    _budget_refused("bond_analytics", {**_BOND, "maturity": 100.0, "frequency": 1000},
                    {"neg": {"maturity": -100.0}, "up": {"ytm": 0.06}}, "maturity × frequency")


def test_runs_that_add_up_to_exactly_one_call_are_allowed():
    # kills m02018, m02056
    out = _call("scenario_table", {"tool": "bond_analytics", "arguments": {**_BOND, "maturity": 25.0, "frequency": 1000},
                                   "scenarios": {f"s{k}": {"ytm": 0.05 + k / 100} for k in range(3)}})
    assert "error" not in out and out["failed"] == 0


def test_the_refusal_names_the_largest_run():
    # kills m02158 (the largest run is chosen by cost, not by its text)
    _budget_refused("bond_analytics", {**_BOND, "maturity": 9.0, "frequency": 1000}, {"big": {"maturity": 100.0}},
                    "= 100000 payment periods")


def test_default_bumps():
    # kills m02022, m02061, m02062, m02125, m02126
    assert _call("sensitivity", {"tool": "black_scholes", "arguments": _BS, "parameter": "r"})["bump"] == 1e-4
    sim = {"expected_frequency": 0.8, "severity_mean": 1000.0, "severity_sd": 500.0, "trials": 2000, "seed": 1}
    assert _call("sensitivity", {"tool": "aggregate_loss", "arguments": sim, "parameter": "severity_mean"})["bump"] == 10.0
    bump = _call("sensitivity", {"tool": "aggregate_loss", "arguments": sim, "parameter": "expected_frequency"})["bump"]
    assert bump == pytest.approx(0.008)


@pytest.mark.parametrize("pd, scheme", [(0.0, "forward"), (1.0, "backward")])
def test_one_sided_differences_of_a_linear_calculator_are_exact(pd, scheme):
    # kills m02140, m02141, m02151, m02152, m02159, m02160, m02173, m02174
    out = _call("sensitivity", {"tool": "expected_loss", "arguments": {"pd": pd, "lgd": 0.5, "ead": 100.0},
                                "parameter": "pd"})
    assert out["scheme"] == scheme
    result = out["outputs"]["result"]
    assert result["derivative"] == pytest.approx(50.0, rel=1e-8)
    assert result["second_derivative"] == pytest.approx(0.0, abs=1e-3)


def test_a_cancellation_between_bumped_runs_is_rp408(monkeypatch):
    # kills m02063, m02064 (sensitivity must not swallow a cancellation)
    import threading

    event = threading.Event()
    real = tools._run_inner
    calls = []

    def run_then_cancel(t, arguments):
        calls.append(1)
        out = real(t, arguments)
        event.set()                   # the client cancels while the base run is computing
        return out

    monkeypatch.setattr(tools, "_run_inner", run_then_cancel)
    with tools._cancellable(event):
        out = _call("sensitivity", {"tool": "black_scholes", "arguments": _BS, "parameter": "S"})
    assert _err(out)["code"] == "RP408" and len(calls) == 1


def test_black_scholes_explain_gives_d1_and_d2():
    # kills m02156, m02157
    S, K, T, r, sigma = 100.0, 90.0, 0.5, 0.03, 0.25
    out = _call("black_scholes", {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "explain": True})
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    assert out["explain"]["d1"] == pytest.approx(d1, rel=1e-12)
    assert out["explain"]["d2"] == pytest.approx(d1 - sigma * math.sqrt(T), rel=1e-12)


def test_trace_text_clips_each_field(tmp_path, capsys):
    # kills m02154, m02161-m02172
    path = tmp_path / "trace.jsonl"
    entry = {"seq": 1, "at": "2026-11-21T10:00:00.000+00:00EXTRA", "tool": "T" * 100, "args": {}, "ms": 1.5,
             "ok": False, "code": "RP101XXXXXXXXXXXX", "param": "P" * 100, "message": "M" * 300, "via": "V" * 20}
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    tools.trace(clear=True)
    assert tools.main(["trace", "--file", str(path)]) == 0
    line = [ln for ln in capsys.readouterr().out.splitlines() if "TTTT" in ln][0]
    assert line.startswith("2026-11-21T10:00:00.000  " + "V" * 12 + " " + "T" * 80 + " ")
    assert ("RP101XXXXXXX " + "P" * 80 + " " + "M" * 200) in line and "M" * 201 not in line


# ---------------------------------------------------------------------------
# riskpy.capital: every sampler path gives uniform marginals and the right Kendall's tau

from riskpy import capital  # noqa: E402

_SAMPLER_PATHS = [
    ("gaussian", lambda: capital.GaussianCopula([[1, 0.5], [0.5, 1]])),
    ("t df=4", lambda: capital.StudentTCopula([[1, 0.5], [0.5, 1]], df=4)),
    ("t df=0.1 (log mixing)", lambda: capital.StudentTCopula([[1, 0.5], [0.5, 1]], df=0.1)),
    ("t df=0.01 (log mixing)", lambda: capital.StudentTCopula([[1, 0.5], [0.5, 1]], df=0.01)),
    ("clayton 2", lambda: capital.ClaytonCopula(2.0)),
    ("clayton 2 dim 3", lambda: capital.ClaytonCopula(2.0, dim=3)),
    ("clayton 20 (log frailty)", lambda: capital.ClaytonCopula(20.0)),
    ("gumbel 2", lambda: capital.GumbelCopula(2.0)),
    ("gumbel 60", lambda: capital.GumbelCopula(60.0)),
    ("frank 0.5", lambda: capital.FrankCopula(0.5)),
    ("frank 5", lambda: capital.FrankCopula(5.0)),
    ("frank -5", lambda: capital.FrankCopula(-5.0)),
    ("frank 40", lambda: capital.FrankCopula(40.0)),
    ("frank 5 dim 3", lambda: capital.FrankCopula(5.0, dim=3)),
    ("frank 30 dim 3 (own log-series)", lambda: capital.FrankCopula(30.0, dim=3)),
    ("frank 200 dim 3 (own log-series)", lambda: capital.FrankCopula(200.0, dim=3)),
]


def _kendall_tau(x, y):
    from scipy import stats

    return stats.kendalltau(x, y)[0]


@pytest.mark.parametrize("name, make", _SAMPLER_PATHS, ids=[n for n, _ in _SAMPLER_PATHS])
def test_sampled_copulas_have_uniform_marginals_and_their_kendall_tau(name, make):
    # kills the sampler-constant mutants of the Clayton (m01281/2, m01328/9, m01330/1, m01350-4, m01389,
    # m01417-9, m01422), t log-mixing (m01276/7, m01324-7) and Frank log-series (m01238, m01296-m01304,
    # m01342, m01364-8, m01397/8) samplers
    from scipy import stats

    cop = make()
    u = np.asarray(cop.sample(20_000, seed=7))
    for j in range(u.shape[1]):
        assert stats.kstest(u[:, j], "uniform").pvalue > 1e-4, (name, j)
    assert _kendall_tau(u[:, 0], u[:, 1]) == pytest.approx(cop.kendall_tau(), abs=0.02)


def test_survival_gumbel_joint_exceedance_is_the_gumbel_cdf_at_the_complement():
    # kills m01133, m01134
    q, th = 0.99, 2.0
    C = math.exp(-((-math.log(1 - q)) ** th * 2) ** (1 / th))
    assert capital.GumbelCopula(th, survival=True).joint_exceedance(q) == pytest.approx(C, rel=1e-12)


@pytest.mark.parametrize("tau", [1e-9, -5e-5, 0.001, 0.5, 0.99, -0.99])
def test_frank_from_kendall_tau_round_trips(tau):
    # kills m01140, m01191, m01235, m01236, m01295, m01340, m01362
    cop = capital.FrankCopula.from_kendall_tau(tau)
    assert cop.kendall_tau() == pytest.approx(tau, rel=1e-10)


def test_frank_from_tiny_tau_uses_the_series():
    # kills m01140 / m01236: theta = 9 tau to rounding for |tau| < 1e-4
    assert capital.FrankCopula.from_kendall_tau(1e-9).theta == pytest.approx(9e-9, rel=1e-12)


def _rp(call):
    with pytest.raises(RiskPyError) as info:
        call()
    return info.value.code


def test_copula_shape_and_index_validation():
    # kills m01109, m01118, m01128, m01166, m01184, m01203, m01252, m01280, m01322
    assert _rp(lambda: capital.ClaytonCopula(2.0, dim=1)) == "RP107"
    assert _rp(lambda: capital.ClaytonCopula(2.0, dim=2.5)) == "RP107"
    assert _rp(lambda: capital.GaussianCopula([[1.0]])) == "RP111"
    assert _rp(lambda: capital.GaussianCopula([[]])) == "RP100"
    assert _rp(lambda: capital.GaussianCopula([[1, 0.5], 0.5])) == "RP404"
    g3 = capital.GaussianCopula([[1, 0.3, 0.2], [0.3, 1, 0.1], [0.2, 0.1, 1]])
    assert _rp(lambda: g3.kendall_tau(1, 1)) == "RP113"
    assert _rp(lambda: g3.kendall_tau(0, 3)) == "RP100"
    assert g3.kendall_tau(0, 2) == pytest.approx(2 / math.pi * math.asin(0.2))


def test_a_two_by_two_matrix_must_be_unit_diagonal_and_symmetric_to_1e12():
    # kills m01257, m01258, m01316-m01319
    assert _rp(lambda: capital.GaussianCopula([[1 + 5e-12, 0.5], [0.5, 1]])) == "RP100"
    assert _rp(lambda: capital.GaussianCopula([[1, 0.5 + 5e-12], [0.5, 1]])) == "RP100"
    capital.GaussianCopula([[1 + 5e-13, 0.5], [0.5, 1]])
    capital.GaussianCopula([[1, 0.5 + 5e-13], [0.5, 1]])
    assert _rp(lambda: capital.GaussianCopula([[1, 0.5], [0.5, 1 - 5e-12]])) == "RP100"     # m01318
    capital.GaussianCopula([[1, 0.5], [0.5, 1 - 5e-13]])                                    # m01319


def test_a_dataframe_correlation_is_read_as_its_values():
    # kills m01153
    pd = pytest.importorskip("pandas")
    cop = capital.GaussianCopula(pd.DataFrame([[1, 0.4], [0.4, 1]]))
    assert cop.correlation.tolist() == [[1.0, 0.4], [0.4, 1.0]]


def test_nearest_correlation_max_iter_is_a_whole_number_of_at_least_one():
    # kills m01102, m01103, m01146, m01195, m01196, m01243, m01244
    m = [[1, 0.5], [0.5, 1]]
    for bad in (0, -1, 2.5, True, "3"):
        assert _rp(lambda: capital.nearest_correlation(m, max_iter=bad)) == "RP107"
    capital.nearest_correlation(m, max_iter=1)


def test_allocate_refuses_losses_whose_totals_overflow_even_when_the_means_do_not():
    # kills m01201
    losses = np.array([[1e308, 1e308], [-1e308, -1e308], [0.0, 0.0]])
    assert _rp(lambda: capital.allocate(losses, level=0.5)) == "RP304"


def _euler_var_oracle(matrix, level):
    """The documented Euler VaR allocation, written out independently."""
    n, k = matrix.shape
    total_losses = matrix.sum(axis=1)
    order = np.argsort(total_losses, kind="mergesort")
    centre = min(max(math.ceil(n * level - 1e-9) - 1, 0), n - 1)
    var = total_losses[order[centre]]
    half = max(int(round(0.5 * math.sqrt(n))), 5)
    half = min(half, centre, n - 1 - centre)
    tied = total_losses == var
    if tied.sum() > 2 * half + 1:
        local = matrix[tied].mean(axis=0)
    else:
        local = matrix[order[centre - half:centre + half + 1]].mean(axis=0)
    centred = total_losses - total_losses.mean()
    beta = matrix.T @ centred / (centred @ centred)
    return var, local + beta * (var - local.sum())


@pytest.mark.parametrize("case", ["continuous", "atom", "atom-at-window-size", "atom-one-over"])
def test_euler_var_contributions_follow_the_documented_window(case):
    # kills m01249, m01344, m01345, m01371, m01372, m01423, m01424
    rng = np.random.default_rng(3)
    n = 400
    if case == "continuous":
        matrix = rng.normal(size=(n, 3)) * [1.0, 2.0, 0.5] + [5, 1, 2]
    else:
        # n = 400 at level 0.5: half = 10, window 21. Put `ties` trials exactly at the VaR.
        ties = {"atom": 60, "atom-at-window-size": 21, "atom-one-over": 22}[case]
        matrix = rng.normal(size=(n, 2)) * 3.0
        matrix[:ties] = [1.0, 2.0]
        totals = matrix.sum(axis=1)
        others = np.flatnonzero(totals != 3.0)
        # 195 trials below the atom, so the sorted window (positions 189..209 around the VaR at 199)
        # straddles the tie group instead of coinciding with it
        below = others[:195]
        above = others[195:]
        matrix[below] = np.abs(matrix[below]) * -1 - 1
        matrix[above] = np.abs(matrix[above]) + 3
    alloc = capital.allocate(matrix, level=0.5, measure="var")
    var, expected = _euler_var_oracle(matrix, 0.5)
    assert alloc.total == pytest.approx(var)
    assert list(alloc.contributions.values()) == pytest.approx(list(expected), rel=1e-12, abs=1e-12)
    assert sum(alloc.contributions.values()) == pytest.approx(var, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# riskpy.mc

from riskpy import mc  # noqa: E402


def test_distribution_parameter_domains():
    # kills m01517, m01520, m01526, m01576, m01577, m01579, m01580, m01592, m01639, m01682, m01684, m01685,
    # m01686, m01710, m01711
    assert _rp(lambda: mc.Normal(0, -1)) == "RP102"
    assert _rp(lambda: mc.Exponential(0.0)) == "RP102"
    assert _rp(lambda: mc.Exponential(math.inf)) in ("RP101", "RP102")
    assert _rp(lambda: mc.Poisson(-1.0)) == "RP102"
    mc.Poisson(0.0)
    mc.Normal(0, 0.0)
    assert _rp(lambda: mc.Uniform(1, 1)) == "RP113"
    mc.Uniform(1, 1 + 1e-9)
    assert _rp(lambda: mc.Bernoulli(1.5)) == "RP103"
    assert _rp(lambda: mc.Bernoulli(-0.1)) == "RP103"
    mc.Bernoulli(1.0)
    mc.Bernoulli(0.0)
    assert _rp(lambda: mc.NegativeBinomial(3, 0.0)) == "RP103"
    mc.NegativeBinomial(3, 1.0)
    mc.NegativeBinomial(3, 0.5)


def test_correlation_entries_validation():
    # kills m01543, m01597, m01599, m01653, m01691, m01692, m01712, m01716, m01717
    model = mc.Model(a=mc.Normal(0, 1), b=mc.Normal(0, 1))
    for rho in (1.0, -1.0):
        model.correlate("a", "b", rho).run(100, seed=1, fn=lambda a, b: a + b)
        model = mc.Model(a=mc.Normal(0, 1), b=mc.Normal(0, 1))
    assert _rp(lambda: mc.Model(a=mc.Normal(0, 1), b=mc.Normal(0, 1)).correlate("a", "b", 1.5)) == "RP100"
    nan = np.array([[1.0, np.nan], [np.nan, 1.0]])
    assert _rp(lambda: mc.iman_conover({"a": np.arange(10.0), "b": np.arange(10.0)}, nan, np.random.default_rng(1))) == "RP101"
    near = np.array([[1.0, 1.0 + 5e-13], [1.0 + 5e-13, 1.0]])
    over = np.array([[1.0, 1.0 + 5e-12], [1.0 + 5e-12, 1.0]])
    mc.iman_conover({"a": np.arange(10.0), "b": np.arange(10.0)}, near, np.random.default_rng(1))
    assert _rp(lambda: mc.iman_conover({"a": np.arange(10.0), "b": np.arange(10.0)}, over, np.random.default_rng(1))) == "RP100"


def test_percentile_and_level_bounds():
    # kills m01606, m01658, m01663, m01664, m01693, m01694, m01700, m01718, m01719
    res = mc.Model(x=mc.Normal(0, 1)).run(500, seed=2, fn=lambda x: x)
    assert res.percentile(0) == res.values.min() and res.percentile(100) == res.values.max()
    assert res.percentile(100.0) == res.values.max()
    for q in (-1e-9, 100.5, 101):
        assert _rp(lambda: res.percentile(q)) == "RP110"
    for level in (0.0, 1.0, 1.5):
        assert _rp(lambda: res.var(level)) == "RP110"


def test_iman_conover_with_a_single_trial_returns_the_samples():
    # kills m01604
    out = mc.iman_conover({"a": np.array([1.0]), "b": np.array([2.0])}, [[1, 0.5], [0.5, 1]], np.random.default_rng(1))
    assert {k: list(v) for k, v in out.items()} == {"a": [1.0], "b": [2.0]}


def test_a_copula_that_returns_the_wrong_shape_is_refused():
    # kills m01627, m01678
    class Bad(capital.GaussianCopula):
        def uniforms(self, n, rng):
            return super().uniforms(n, rng)[:, :1]

    model = mc.Model(a=mc.Normal(0, 1), b=mc.Normal(0, 1))
    assert _rp(lambda: model.run(100, seed=1, copula=Bad(0.5), fn=lambda a, b: a + b)) == "RP109"


def test_lhs_under_a_copula_fills_every_stratum_and_random_sampling_does_not():
    # kills m01673, m01769
    cop = capital.GaussianCopula(0.5)
    model = mc.Model(a=mc.Uniform(0, 1), b=mc.Uniform(0, 1))
    n = 1000
    lhs = model.sample(n, seed=3, sampling="lhs", copula=cop)
    assert len(set(np.floor(lhs["a"] * n).astype(int))) == n
    rnd = model.sample(n, seed=3, sampling="random", copula=cop)
    assert len(set(np.floor(rnd["a"] * n).astype(int))) < 0.8 * n
    st = model.stress({"s": {"a": mc.Uniform(0, 2)}}, trials=n, seed=3, sampling="lhs", copula=cop,
                      fn=lambda a, b: a + b)
    stressed = np.asarray(st.scenarios["s"].inputs["a"])
    assert len(set(np.floor(stressed / 2 * n).astype(int))) == n
    st = model.stress({"s": {"a": mc.Uniform(0, 2)}}, trials=n, seed=3, sampling="random", copula=cop,
                      fn=lambda a, b: a + b)
    stressed = np.asarray(st.scenarios["s"].inputs["a"])
    assert len(set(np.floor(stressed / 2 * n).astype(int))) < 0.8 * n


def test_seeded_stress_and_lhs_copula_runs_are_reproducible_to_the_last_digit():
    # kills the per-variable stream derivation mutants m01656, m01752-m01758, m01764-m01768, m01771-m01777:
    # a ledger replay (riskpy.audit) relies on a seed giving the same numbers in every 0.8.x release
    m = mc.Model(d=mc.Bernoulli(0.02), loss=mc.LogNormal(10.0, 1.0), cnt=mc.Poisson(2.0)).correlate("loss", "cnt", 0.5)
    st = m.stress({"pd up": {"d": mc.Bernoulli(0.05)}, "sev": {"loss": mc.LogNormal(10.2, 1.1)}}, trials=4000,
                  seed=11, fn=lambda d, loss, cnt: d * loss + cnt * 1000.0)
    assert float(st.base.values.sum()) == pytest.approx(11199190.643032493, rel=1e-12)
    assert float(st.scenarios["pd up"].values.sum()) == pytest.approx(16365966.856661104, rel=1e-12)
    assert float(st.scenarios["sev"].values.sum()) == pytest.approx(12246375.845723249, rel=1e-12)
    cop = capital.ClaytonCopula(2.0)
    mm = mc.Model(a=mc.Normal(0, 1), b=mc.Gamma(2.0, 1.0))
    res = mm.run(3000, seed=5, sampling="lhs", copula=cop, fn=lambda a, b: a + b)
    assert float(res.values.sum()) == pytest.approx(5999.242703036966, rel=1e-12)
    st2 = mm.stress({"x": {"b": mc.Gamma(3.0, 1.0)}}, trials=3000, seed=5, sampling="lhs", copula=cop,
                    fn=lambda a, b: a + b)
    assert float(st2.scenarios["x"].values.sum()) == pytest.approx(8999.248429136947, rel=1e-12)


@pytest.mark.parametrize("tail", ["upper", "lower"])
@pytest.mark.parametrize("level", [0.99, 0.95, 0.9])
def test_reverse_stress_tail_mean_is_tvar_with_ties(tail, level):
    # kills m01741, m01742, m01743 (the VaR index) for both tails, with tied outputs
    res = mc.Model(x=mc.Poisson(3.0), y=mc.Normal(0, 1)).run(1000, seed=1, fn=lambda x, y: x)
    rs = res.reverse_stress(level, tail=tail)
    if tail == "upper":
        assert rs.output_mean == pytest.approx(res.tvar(level), rel=1e-12)
    else:
        neg = mc.Model(x=mc.Poisson(3.0), y=mc.Normal(0, 1)).run(1000, seed=1, fn=lambda x, y: -x)
        assert -rs.output_mean == pytest.approx(neg.tvar(level), rel=1e-12)


def test_reverse_stress_with_a_threshold_counts_trials_at_the_threshold():
    # kills m01732, m01733
    res = mc.Model(x=mc.Poisson(3.0)).run(1000, seed=1, fn=lambda x: x)
    lower = res.reverse_stress(threshold=1.0, tail="lower")
    assert lower.tail_trials == int((res.values <= 1.0).sum())
    upper = res.reverse_stress(threshold=6.0, tail="upper")
    assert upper.tail_trials == int((res.values >= 6.0).sum())
    assert upper.output_mean == pytest.approx(float(res.values[res.values >= 6.0].mean()))


def test_convergence_of_a_short_run_starts_at_the_second_trial():
    # kills m01721, m01722
    res = mc.Model(x=mc.Normal(0, 1)).run(5, seed=1, fn=lambda x: x)
    n = np.asarray(res.convergence()[0])
    assert n.tolist() == [2, 3, 4, 5]


# ---------------------------------------------------------------------------
# riskpy.quant, riskpy.rates


def test_var_helpers_check_level_and_refuse_text():
    # kills m01797, m01800, m01816, m01817, m01832, m01833, m01853
    from riskpy import quant

    r = [0.01, -0.02, 0.03, -0.01]
    for level in (0.0, 1.0, 1.5):
        assert _rp(lambda: quant.historical_var(r, level=level)) == "RP110"
        assert _rp(lambda: quant.expected_shortfall(r, level=level)) == "RP110"
    assert _rp(lambda: quant.historical_var("123")) == "RP101"
    assert _rp(lambda: quant.historical_var(b"123")) == "RP101"


def test_a_far_out_of_the_money_put_is_worth_almost_nothing():
    # kills m01831 (the put's floor is 0, not 1)
    from riskpy import quant

    assert quant.black_scholes(S=100, K=50, T=1, r=0.05, sigma=0.2, kind="put") == pytest.approx(3.3334219755660945e-4,
                                                                                                rel=1e-9)


def test_heston_accepts_two_quadrature_nodes():
    # kills m01871
    from riskpy import quant

    assert math.isfinite(quant.heston_price(S=100, K=100, T=1, r=0.03, v0=0.04, kappa=2.0, theta=0.04, xi=0.3,
                                            rho=-0.5, nodes=2))


def test_short_rate_bonds_at_their_maturity_and_near_the_overflow_limit():
    # kills m01889, m01901, m01902, m01906, m01909
    from riskpy import rates

    assert rates.vasicek_zero_bond(0.03, 5.0, 5.0, 0.5, 0.04, 0.01) == 1.0
    assert rates.cir_zero_bond(0.03, 5.0, 5.0, 0.5, 0.04, 0.01) == 1.0
    big = rates.vasicek_zero_bond(r=-1.0, t=0.0, T=400.0, kappa=1e-9, theta=0.0, sigma=0.0)
    assert big == pytest.approx(math.exp(-math.expm1(-1e-9 * 400.0) / 1e-9), rel=1e-9)


# ---------------------------------------------------------------------------
# riskpy.verify, riskpy.errors, riskpy.context: the CLIs' edges


def test_verify_finds_an_unwritable_output_before_running(tmp_path):
    # kills m02178, m02180, m02182, m02187, m02190, m02191, m02204
    from riskpy import verify

    assert isinstance(verify._unwritable(str(tmp_path)), IsADirectoryError)
    assert isinstance(verify._unwritable(str(tmp_path / "missing" / "out.json")), FileNotFoundError)
    locked = tmp_path / "locked.json"
    locked.write_text("{}")
    locked.chmod(0o444)
    try:
        if not os.access(locked, os.W_OK):
            assert isinstance(verify._unwritable(str(locked)), PermissionError)
    finally:
        locked.chmod(0o644)
    assert verify._unwritable(str(tmp_path / "fine.json")) is None


class _FakeReport:
    def __init__(self, ok):
        self.ok = ok

    def to_dict(self):
        return {"ok": self.ok}

    def to_markdown(self):
        return "# report"

    def summary(self, verbose=False):
        return "SUMMARY-LINE"


@pytest.mark.parametrize("ok, status", [(True, 0), (False, 1)])
def test_verify_cli_status_quiet_and_json(tmp_path, monkeypatch, capsys, ok, status):
    # kills m02186, m02195, m02196, m02197, m02205, m02206
    from riskpy import verify

    monkeypatch.setattr(verify, "run", lambda *a, **k: _FakeReport(ok))
    out = tmp_path / "r.json"
    assert verify.main(["--json", str(out)]) == status
    assert "SUMMARY-LINE" in capsys.readouterr().out
    assert out.read_text() == '{\n  "ok": ' + ("true" if ok else "false") + "\n}\n"
    assert verify.main(["--quiet"]) == status
    assert "SUMMARY-LINE" not in capsys.readouterr().out


def test_verify_cli_stops_quietly_on_a_closed_pipe_but_raises_other_os_errors(monkeypatch):
    # kills m02199, m02203
    import errno
    from riskpy import verify

    monkeypatch.setattr(verify, "run", lambda *a, **k: _FakeReport(True))
    monkeypatch.setattr(sys, "stdout", _Pipe(BrokenPipeError(errno.EPIPE, "Broken pipe")))
    assert verify.main([]) == 0
    monkeypatch.setattr(sys, "stdout", _Pipe(OSError(errno.EIO, "I/O error")))
    with pytest.raises(OSError):
        verify.main([])


def test_errors_broken_pipe_and_output_failed(monkeypatch, capsys):
    # kills m01475, m01479, m01489
    from riskpy import errors

    monkeypatch.setattr(sys, "stdout", _Pipe(BrokenPipeError()))
    assert errors._broken_pipe() == 0
    monkeypatch.undo()
    assert errors._output_failed("prog", "x.json", OSError()) == 1
    assert "cannot write 'x.json': OSError" in capsys.readouterr().err


def test_context_summaries_of_long_docstrings():
    # kills m01433-m01435, m01438, m01440, m01442-m01446, m01448, m01449
    from riskpy import context

    exact = "a" * 240
    assert context._summary(exact) == exact
    first = "This is the first sentence of a long paragraph. " + "Then " + "words " * 60
    assert context._summary(first) == "This is the first sentence of a long paragraph."
    no_break = "word " * 80
    out = context._summary(no_break)
    assert out.endswith("...") and len(out) <= 240 and not out.endswith(" ...")
    long_first = ("x" * 250) + ". Second sentence here."
    assert context._summary(long_first).endswith("...")

    class Own:
        """Own summary."""

    class Inherits(Own):
        pass

    import dataclasses as dc

    @dc.dataclass
    class Data:
        a: int

    assert context._doc(Own) == "Own summary."
    assert context._doc(Inherits) == ""
    assert context._doc(Data) == ""
    assert context._doc(context._summary).startswith("The first paragraph of a docstring")


# ---------------------------------------------------------------------------
# Second round: mutants the first round of this file still let through


def test_the_frank_log_series_frailty_follows_its_pmf():
    # kills m01298-m01301, m01303, m01304, m01364-m01367, m01397, m01398 (the 0.8.0 log-series sampler)
    from scipy import stats

    for theta in (30.0, 200.0):
        k = capital._logseries_frailty(theta, 200_000, np.random.default_rng(5))
        p = -math.expm1(-theta)
        pmf = np.array([p ** j / (j * theta) for j in range(1, 6)])
        observed = np.array([(k == j).sum() for j in range(1, 6)] + [(k > 5).sum()])
        expected = np.append(pmf, 1.0 - pmf.sum()) * k.size
        assert np.all(k >= 1) and stats.chisquare(observed, expected).pvalue > 1e-4


def test_t_copula_log_mixing_marginals_with_a_large_sample():
    # kills m01326 (the 2/df scale of the log-space mixing variable), which 20,000 draws cannot see
    from scipy import stats

    u = np.asarray(capital.StudentTCopula(0.5, df=0.1).sample(200_000, seed=11))
    assert min(stats.kstest(u[:, j], "uniform").pvalue for j in range(2)) > 1e-4


def test_report_numbers_a_damaged_line_by_its_position(tmp_path):
    # kills m00821, m00822, m01046, m01047
    path = tmp_path / "l.jsonl"
    entries = _chain([{"tool": "t", "arguments": {}, "result": k} for k in range(2)])
    path.write_text(json.dumps(entries[0]) + "\nnot json\n" + json.dumps(entries[1]) + "\n", encoding="ascii")
    text = audit.report(str(path))
    assert "| 2 |  | *damaged line* | line 2 " in text
    skipped = audit.replay(str(path)).skipped
    assert [s["seq"] for s in skipped if s["tool"] is None] == [2]


def test_replay_skips_an_entry_whose_result_could_not_be_stored():
    # kills m00809
    entries = _chain([{"tool": "expected_loss", "arguments": {"pd": 0.1, "lgd": 0.5, "ead": 1.0},
                       "result": {"$riskpy": "unserialisable", "type": "dict", "reason": "x", "sha256": None}}])
    outcome = audit.replay(entries)
    assert outcome.matched == 0 and "could not be stored" in outcome.skipped[0]["reason"]


def test_the_report_lists_at_most_five_differences_per_drifted_call(tmp_path):
    # kills m01009, m01010
    args = {"standalone": [100.0, 60.0, 40.0], "correlation": [[1, 0.2, 0.1], [0.2, 1, 0.3], [0.1, 0.3, 1]]}
    result = tools.call("capital_aggregate", args)

    def doubled(value):
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return value * 2 + 1
        if isinstance(value, dict):
            return {k: doubled(v) for k, v in value.items()}
        if isinstance(value, list):
            return [doubled(v) for v in value]
        return value

    path = tmp_path / "l.jsonl"
    _write(path, _chain([{"tool": "capital_aggregate", "arguments": args, "result": doubled(result)}]))
    assert len(audit.replay(str(path)).drifted[0]["differences"]) > 5
    lines = [ln for ln in audit.report(str(path), check=True).splitlines() if ln.startswith("- drift, call 1")]
    assert len(lines) == 5


def test_run_with_lhs_under_a_copula_fills_every_stratum():
    # kills m01769 (Model.run, as opposed to Model.sample)
    cop = capital.GaussianCopula(0.5)
    model = mc.Model(a=mc.Uniform(0, 1), b=mc.Uniform(0, 1))
    n = 1000
    lhs = model.run(n, seed=3, sampling="lhs", copula=cop, fn=lambda a, b: a)
    assert len(set(np.floor(np.asarray(lhs.inputs["a"]) * n).astype(int))) == n
    rnd = model.run(n, seed=3, sampling="random", copula=cop, fn=lambda a, b: a)
    assert len(set(np.floor(np.asarray(rnd.inputs["a"]) * n).astype(int))) < 0.8 * n


def test_correlation_refusals_say_which_check_failed():
    # kills m01543, m01599 (the NaN check) and m01597, m01712 (the per-pair range check)
    nan = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(RiskPyError, match="contains NaN") as info:
        mc.iman_conover({"a": np.arange(10.0), "b": np.arange(10.0)}, nan, np.random.default_rng(1))
    assert info.value.code == "RP101"
    with pytest.raises(RiskPyError, match=r"for \('a', 'b'\)"):
        mc.Model(a=mc.Normal(0, 1), b=mc.Normal(0, 1)).correlate("a", "b", 1.5)


@pytest.mark.parametrize("level", [0.99, 0.95])
def test_reverse_stress_of_distinct_outputs_takes_exactly_the_worst_share(level):
    # kills m01742, m01743 (with distinct outputs the VaR index decides which trial is the cut)
    res = mc.Model(x=mc.Normal(0, 1)).run(1000, seed=4, fn=lambda x: x)
    rs = res.reverse_stress(level)
    assert rs.tail_trials == round(1000 * (1 - level))
    assert rs.output_mean == pytest.approx(res.tvar(level), rel=1e-12)
    assert rs.output_mean == pytest.approx(float(np.sort(res.values)[-round(1000 * (1 - level)):].mean()), rel=1e-12)


def test_triangle_and_matrix_budgets_are_quadratic():
    # kills m02120, m02121, m02122
    I = lambda n: [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]    # noqa: E731
    ok = _call("scenario_table", {"tool": "chain_ladder", "arguments": {"triangle": [[1.0] * 100 for _ in range(100)]},
                                  "scenarios": {"t": {"tail": 1.01}}})
    assert "error" not in ok and ok["failed"] == 0
    ok = _call("scenario_table", {"tool": "capital_aggregate",
                                  "arguments": {"standalone": [1.0] * 50, "correlation": I(50)},
                                  "scenarios": {"a": {}, "b": {}}})
    assert "error" not in ok
    _budget_refused("capital_aggregate", {"standalone": [1.0] * 60, "correlation": I(60)}, {"a": {}, "b": {}},
                    "60 risks")


def test_trace_json_is_indented_by_two(tmp_path, capsys):
    # kills m02143, m02144
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"seq": 1, "tool": "t", "ok": True}) + "\n", encoding="utf-8")
    tools.trace(clear=True)
    assert tools.main(["trace", "--json", "--file", str(path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[" and lines[1] == "  {" and lines[2].startswith('    "')


def _cut(paragraph):
    """The documented fallback: the first 237 characters, back to the last space, then '...'."""
    head = paragraph[:237]
    return (head[:head.rfind(" ")] if " " in head else head) + "..."


@pytest.mark.parametrize("space_at", [235, 236, 237])
def test_context_summary_cuts_at_a_word_boundary(space_at):
    # kills m01466, m01467, m01472, m01473: a space just inside, at, or just past the 237-character cut
    from riskpy import context

    text = "y" * space_at + " " + "z" * 30 + " " + "q " * 20
    expected = ("y" * space_at if space_at < 237 else "y" * 237) + "..."
    assert context._summary(text) == expected == _cut(" ".join(text.split()))


def test_context_summary_length_boundaries():
    # kills m01448, m01450, m01457, m01458
    from riskpy import context

    w241 = ("abcd " * 49)[:241]
    assert len(w241) == 241 and context._summary(w241) == _cut(w241)
    s240 = "A" * 239 + "."
    tail = " Second sentence follows here and is long enough. " + "more words " * 10
    assert context._summary(s240 + tail) == s240
    s241 = "A" * 240 + "."
    assert context._summary(s241 + tail) == _cut(" ".join((s241 + tail).split()))


def test_context_class_signature_of_a_builtin_is_empty():
    # kills m01453
    from riskpy import context

    assert context._class_signature(int) == ""


def test_context_cli_stops_quietly_on_a_closed_pipe_but_raises_other_os_errors(monkeypatch):
    # kills m01455, m01456, m01465
    import errno
    from riskpy import context

    monkeypatch.setattr(sys, "stdout", _Pipe(BrokenPipeError(errno.EPIPE, "Broken pipe")))
    assert context.main([]) == 0
    monkeypatch.setattr(sys, "stdout", _Pipe(OSError(errno.EIO, "I/O error")))
    with pytest.raises(OSError):
        context.main([])


def test_seeded_stress_of_an_independent_tied_input_is_reproducible():
    # kills m01767, m01768: without a copula or correlate(), ties in the shocked input are broken by the
    # variable's own tie-break stream, so that stream is part of the seed's contract
    m = mc.Model(d=mc.Bernoulli(0.02), loss=mc.LogNormal(10.0, 1.0))
    st = m.stress({"pd up": {"d": mc.Bernoulli(0.05)}}, trials=4000, seed=11, fn=lambda d, loss: d * loss)
    assert float(st.base.values.sum()) == pytest.approx(2571502.3426220813, rel=1e-12)
    assert float(st.scenarios["pd up"].values.sum()) == pytest.approx(5963882.369002972, rel=1e-12)


@pytest.mark.parametrize("level", [0.95, 0.05])
def test_reverse_stress_of_ten_trials_at_extreme_levels(level):
    # kills m01741, m01742: the VaR index is clipped to [0, n - 1]
    res = mc.Model(x=mc.Normal(0, 1)).run(10, seed=4, fn=lambda x: x)
    assert res.reverse_stress(level).output_mean == pytest.approx(res.tvar(level), rel=1e-12)


def test_a_short_rate_price_just_past_the_largest_double_is_a_coded_rp304():
    # kills m01888 (_MAX_EXP + 1 lets e^710.2 reach math.exp and fail as a bare OverflowError)
    from riskpy import rates

    with pytest.raises(RiskPyError) as info:
        rates.vasicek_zero_bond(r=-1.0, t=0.0, T=710.2, kappa=1e-9, theta=0.0, sigma=0.0)
    assert info.value.code == "RP304" and info.value.param == "r"
