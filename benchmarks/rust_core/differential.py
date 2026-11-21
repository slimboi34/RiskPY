"""Differential test: the Rust core against the C++ core of 0.8.0, call for call.

Each side runs in its own interpreter (one venv with ``open-riskpy==0.8.0``, one with
the build under test), as a worker that reads calls as JSON lines on stdin and answers
on stdout. Hypothesis generates the calls — wide inputs with the edge cases mixed in
(0, -0.0, subnormals, ±1e300, ±inf, NaN, huge and negative ints, empty and ragged
lists, ints, bools, strings, NumPy scalars, Decimals where a number is expected) — and
sends every call to both workers. The answers must agree:

* the same exception type in the same cases, and the same message for ``ValueError``
  and for the ``RuntimeError`` of a bad Excel column;
* bit-identical values where the C++ was pure IEEE arithmetic (+ − × ÷, sqrt, min,
  max: rating, triangles, credibility, burning cost, ...);
* within a stated tolerance where it called the C library's transcendental functions
  (pow, erfc, sin/cos in the FFT), which the Rust core takes from the pure-Rust libm so
  that results no longer depend on the platform; the tolerance is relative to the
  size of the terms, not of a result that cancels;
* the two deferred 0.8.0 fixes are checked against a 50-digit mpmath evaluation:
  ``present_value``/``future_value`` must be accurate where the C++ was too, and every
  difference from the C++ must be explained by the C++'s own error; negative
  periods and NaN/inf payments must now raise ``ValueError``.

Random draws are not compared (the generator changed on purpose); the argument checks
of every ``simulate_*`` method and its deterministic corners are.

    python differential.py --cpp /path/to/venv-cpp/bin/python \\
                           --rust /path/to/venv-rust/bin/python [--examples 400]

Prints a table of functions, cases, mismatches and intended differences, and exits 1
on any unexplained mismatch.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from collections import defaultdict

# ---------------------------------------------------------------------------
# Wire format: floats as hex so nothing is lost; tagged exotic values.


def enc(v):
    if isinstance(v, bool):
        return {"$bool": v}
    if isinstance(v, float):
        return {"$f": v.hex()}
    if isinstance(v, int):
        return {"$i": str(v)}
    if isinstance(v, str):
        return {"$s": v}
    if isinstance(v, bytes):
        return {"$b": base64.b64encode(v).decode()}
    if v is None:
        return {"$none": True}
    if isinstance(v, tuple):
        return {"$t": [enc(x) for x in v]}
    if isinstance(v, list):
        return [enc(x) for x in v]
    if isinstance(v, dict):
        return {"$d": [[enc(k), enc(x)] for k, x in v.items()]}
    if isinstance(v, Tagged):
        return {"$x": [v.kind, v.payload]}
    raise TypeError(type(v))


class Tagged:
    """A value the worker builds itself (NumPy scalar, Decimal, NumPy array)."""

    def __init__(self, kind, payload):
        self.kind, self.payload = kind, payload

    def __repr__(self):
        return f"{self.kind}({self.payload!r})"


def dec(v):
    if isinstance(v, list):
        return [dec(x) for x in v]
    if not isinstance(v, dict):
        return v
    (tag, x), = v.items()
    if tag == "$bool":
        return x
    if tag == "$f":
        return float.fromhex(x)
    if tag == "$i":
        return int(x)
    if tag == "$s":
        return x
    if tag == "$b":
        return base64.b64decode(x)
    if tag == "$none":
        return None
    if tag == "$t":
        return tuple(dec(i) for i in x)
    if tag == "$d":
        return {dec(k): dec(i) for k, i in x}
    if tag == "$x":
        kind, payload = x
        if kind == "np.float64":
            import numpy as np
            return np.float64(float.fromhex(payload))
        if kind == "np.float32":
            import numpy as np
            return np.float32(float.fromhex(payload))
        if kind == "np.int64":
            import numpy as np
            return np.int64(int(payload))
        if kind == "decimal":
            from decimal import Decimal
            return Decimal(payload)
        if kind == "ndarray":
            import numpy as np
            return np.array([float.fromhex(p) for p in payload])
        raise ValueError(kind)
    raise ValueError(tag)


def enc_result(r):
    if isinstance(r, float):
        return {"$f": r.hex()}
    if isinstance(r, bool):
        return {"$bool": r}
    if isinstance(r, int):
        return {"$i": str(r)}
    if isinstance(r, (list, tuple)):
        return {"$t" if isinstance(r, tuple) else "$l": [enc_result(x) for x in r]}
    if r is None:
        return {"$none": True}
    if isinstance(r, str):
        return {"$s": r}
    if isinstance(r, dict):
        return {"$d": [[enc_result(k), enc_result(x)] for k, x in r.items()]}
    return {"$repr": repr(r)}


def dec_result(r):
    if not isinstance(r, dict):
        return r
    (tag, x), = r.items()
    if tag == "$f":
        return float.fromhex(x)
    if tag == "$i":
        return int(x)
    if tag in ("$t", "$l"):
        seq = [dec_result(i) for i in x]
        return tuple(seq) if tag == "$t" else seq
    if tag == "$d":
        return {dec_result(k): dec_result(i) for k, i in x}
    if tag in ("$s", "$bool", "$repr"):
        return x
    if tag == "$none":
        return None
    raise ValueError(tag)


# ---------------------------------------------------------------------------
# Worker: runs one "program" (a list of steps on fresh objects) per request.


def run_program(core, program, tmpdir):
    """A program is a list of steps; objects made by "new" are referenced by index."""
    objs = []
    out = None
    for step in program:
        op = step[0]
        if op == "new":
            _, cls, args = step
            objs.append(getattr(core, cls)(*dec(args)))
            out = None
        elif op == "call":
            _, target, meth, args = step
            fn = getattr(getattr(core, target) if isinstance(target, str) else objs[target], meth)
            out = fn(*dec(args))
        elif op == "logic":            # engine.set_math_logic(<python logic by name>)
            _, target, name = step
            engine = objs[target]
            if name == "factor":
                engine.set_math_logic(objs[step[3]].calculate)
            elif name == "linear":
                engine.set_math_logic(lambda d: sum(v for v in d.values() if isinstance(v, float)) * 1.5 + 7.0)
            elif name == "seen":
                engine.set_math_logic(lambda d: float(len(json.dumps(sorted((k, str(v)) for k, v in d.items())))))
            elif name == "bad":
                engine.set_math_logic(lambda d: "not a number")
            elif name == "none":
                engine.set_math_logic(None)
            out = None
        elif op == "attach":
            _, target, exporter = step
            objs[target].attach_exporter(objs[exporter] if exporter is not None else None)
            out = None
        elif op == "export":           # export, then read the workbook back as cells
            _, target, kind, args = step
            path = os.path.join(tmpdir, "out.xlsx")
            if os.path.exists(path):
                os.remove(path)
            if kind == "single":
                objs[target].export_to_excel(path)
            else:
                objs[target].export_batch_to_excel(path, *dec(args))
            out = read_cells(path) if os.path.exists(path) else "no file"
        elif op == "field":            # engine.add_field(Field(name, label, type))
            _, target, name, label, kind = step
            objs[target].add_field(core.Field(name, label, kind))
            out = None
        elif op == "fields":
            out = [(f.name, f.label, f.type) for f in objs[step[1]].get_fields()]
        else:
            raise ValueError(op)
    return out


def read_cells(path):
    import openpyxl

    wb = openpyxl.load_workbook(path)
    ws = wb[wb.sheetnames[0]]
    cells = {}
    for row in ws.iter_rows():
        for c in row:
            if c.value is not None:
                v = c.value
                cells[c.coordinate] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
    return {"sheets": wb.sheetnames, "cells": cells}


def worker():
    import io
    import riskpy.cpp_underwriter as core

    # The cores print export messages on fd 1, which is the protocol channel: keep the
    # protocol on a duplicate of fd 1 and send fd 1 itself to stderr.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = io.TextIOWrapper(os.fdopen(os.dup(2), "wb"), line_buffering=True)
    tmpdir = tempfile.mkdtemp(prefix="riskpy-diff-")
    for line in sys.stdin:
        req = json.loads(line)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = {"ok": True, "value": enc_result(run_program(core, req["program"], tmpdir))}
        except BaseException as exc:  # noqa: BLE001 - every outcome is data here
            res = {"ok": False, "exc": type(exc).__name__, "msg": str(exc)}
        proto.write(json.dumps(res) + "\n")


def encode_program(program):
    """Arguments go over the wire tagged (see enc); the step structure stays plain JSON."""
    out = []
    for step in program:
        if step[0] == "new":
            step = [step[0], step[1], enc(list(step[2]))]
        elif step[0] == "call":
            step = [step[0], step[1], step[2], enc(list(step[3]))]
        elif step[0] == "export":
            step = [step[0], step[1], step[2], enc(list(step[3]))]
        out.append(step)
    return out


class Worker:
    def __init__(self, python):
        self.p = subprocess.Popen([python, "-I", os.path.abspath(__file__), "--worker"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                  cwd=tempfile.gettempdir())

    def run(self, program):
        self.p.stdin.write(json.dumps({"program": encode_program(program)}) + "\n")
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("worker died")
        return json.loads(line)


# ---------------------------------------------------------------------------
# Comparison


def same_float(a, b):
    return (math.isnan(a) and math.isnan(b)) or a.hex() == b.hex()


def as_float(v):
    """The double the cores' float conversion makes of v, or None if they refuse it."""
    v = dec(enc(v)) if isinstance(v, Tagged) else v
    if isinstance(v, (str, bytes)) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def as_int(v):
    """The C++ int the cores make of v, or None if they refuse it."""
    v = dec(enc(v)) if isinstance(v, Tagged) else v
    if isinstance(v, (float, str, bytes)) or v is None:
        return None
    try:
        i = int(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return i if -2**31 <= i < 2**31 else None


def fma(a, b, c):
    """a*b + c rounded once, as a fused multiply-add computes it."""
    from fractions import Fraction
    if not all(math.isfinite(x) for x in (a, b, c)):
        return a * b + c
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def fused_judge(candidates):
    """An `intended` judge for methods whose single expression contains a*b + c.

    The 0.8.0 macOS arm64 wheel was built by clang, which contracts a*b + c into one
    fused multiply-add (FMA) by default; the Linux and Windows x86_64 wheels were not
    (baseline x86-64 has no FMA instruction). Rust never contracts. So where the C++
    answer equals the fused evaluation and the Rust answer the plain one, the two
    differ by the FMA's rounding, not by a porting error.
    """
    def judge(cpp, rust):
        if not (cpp["ok"] and rust["ok"]):
            return None
        c, r = dec_result(cpp["value"]), dec_result(rust["value"])
        if same_float(c, r):
            return None
        plain, fused = candidates()
        if plain is None:
            return None
        if same_float(r, plain) and any(same_float(c, f) for f in fused):
            return "C++ arm64 wheel fused a*b+c (FMA); Rust, like the x86_64 C++ wheels, does not"
        return None
    return judge


def flatten(v):
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out.extend(flatten(x))
        return out
    return [v]


class Ledger:
    def __init__(self):
        self.cases = defaultdict(int)
        self.mismatch = defaultdict(list)
        self.intended = defaultdict(lambda: defaultdict(int))
        self.bitident = defaultdict(lambda: [0, 0])
        self.maxrel = defaultdict(float)


LEDGER = Ledger()


def classify(fn, cpp, rust, *, mode="exact", tol=None, scale=None, intended=None):
    """Record and return True if the two answers agree under `mode`."""
    LEDGER.cases[fn] += 1
    if intended is not None:
        reason = intended(cpp, rust)
        if reason:
            LEDGER.intended[fn][reason] += 1
            return True
    if cpp["ok"] != rust["ok"]:
        LEDGER.mismatch[fn].append(("raised on one side only", cpp, rust))
        return False
    if not cpp["ok"]:
        if cpp["exc"] != rust["exc"]:
            LEDGER.mismatch[fn].append(("different exception types", cpp, rust))
            return False
        if (cpp["exc"] == "ValueError" or "XLCellReference" in cpp["msg"]) and cpp["msg"] != rust["msg"]:
            LEDGER.mismatch[fn].append(("different ValueError messages", cpp, rust))
            return False
        return True
    a, b = dec_result(cpp["value"]), dec_result(rust["value"])
    fa, fb = flatten(a), flatten(b)
    if len(fa) != len(fb) or any(type(x) is not type(y) for x, y in zip(fa, fb)):
        LEDGER.mismatch[fn].append(("different shapes or types", cpp, rust))
        return False
    for i, (x, y) in enumerate(zip(fa, fb)):
        if not isinstance(x, float):
            if x != y:
                LEDGER.mismatch[fn].append((f"element {i} differs", cpp, rust))
                return False
            continue
        exact = same_float(x, y)
        LEDGER.bitident[fn][0] += exact
        LEDGER.bitident[fn][1] += 1
        if exact:
            continue
        if mode == "exact":
            LEDGER.mismatch[fn].append((f"element {i} not bit-identical", cpp, rust))
            return False
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            LEDGER.mismatch[fn].append((f"element {i}: {x!r} vs {y!r}", cpp, rust))
            return False
        s = scale(i) if scale else max(abs(x), abs(y))
        rel = abs(x - y) / s if s > 0 else abs(x - y)
        LEDGER.maxrel[fn] = max(LEDGER.maxrel[fn], rel)
        if rel > tol:
            LEDGER.mismatch[fn].append((f"element {i}: relative difference {rel:.3g} > {tol:g}", cpp, rust))
            return False
    return True


# ---------------------------------------------------------------------------
# Strategies and properties

def build_tests(cpp_w, rust_w, examples):
    import mpmath as mp
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    mp.mp.dps = 50
    cfg = settings(max_examples=examples, deadline=None, database=None, derandomize=True,
                   suppress_health_check=list(HealthCheck), print_blob=False)

    edge_floats = st.sampled_from([0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 1e-300, -1e-300, 1e-12,
                                   1e-8, 0.5, 1.0, -1.0, -0.5, -0.999999, -1.0 + 1e-15, 2.0, 1e9, 1e12, 1e15,
                                   1e300, -1e300, 1.7976931348623157e308, math.inf, -math.inf, math.nan])
    any_float = st.one_of(edge_floats, st.floats(allow_nan=True, allow_infinity=True),
                          st.floats(min_value=-1e6, max_value=1e6), st.floats(min_value=0, max_value=1))
    pos_float = st.one_of(st.floats(min_value=1e-300, max_value=1e300), st.sampled_from([1.0, 0.5, 1e-9, 1e9, 1e300]))
    ratio = st.one_of(st.floats(min_value=0.0, max_value=1.0), edge_floats)
    any_int = st.one_of(st.integers(min_value=-2**31, max_value=2**31 - 1),
                        st.sampled_from([0, 1, -1, 2**31 - 1, -2**31, 2**31, -2**31 - 1, 10**20, 18, 19, 90, 91]),
                        st.integers(min_value=-5, max_value=200))
    # numbers as other types: must convert (or be refused) identically
    exotic_num = st.one_of(
        st.booleans(), st.none(), st.sampled_from(["1.5", "abc", b"1", 10**400]),
        st.floats(allow_nan=False, allow_infinity=False, width=32).map(lambda f: Tagged("np.float32", float(f).hex())),
        st.floats().map(lambda f: Tagged("np.float64", f.hex())),
        st.integers(min_value=-2**40, max_value=2**40).map(lambda i: Tagged("np.int64", str(i))),
        st.sampled_from(["5.7", "-3", "1e400", "NaN"]).map(lambda s: Tagged("decimal", s)),
    )
    num = st.one_of(any_float, any_float, any_float, exotic_num)
    intish = st.one_of(any_int, any_int, any_int, exotic_num, any_float)
    loss_list = st.one_of(
        st.lists(st.one_of(st.floats(min_value=0, max_value=1e9), edge_floats), max_size=12),
        st.lists(st.floats(min_value=0, max_value=1e7), max_size=40).map(tuple),
        st.lists(st.floats(min_value=0, max_value=1e6), max_size=10).map(lambda xs: Tagged("ndarray", [x.hex() for x in xs])),
        st.sampled_from(["abc", None, 5.0, [1.0, "x"], [[1.0]], {1.0: 2.0}]),
    )
    rate_list = st.one_of(
        st.lists(st.one_of(st.floats(min_value=-0.99, max_value=2.0), edge_floats), max_size=12),
        st.lists(st.floats(min_value=-0.5, max_value=0.5), max_size=30),
    )

    def mostly(valid, wild):
        """Constructor arguments for the method tests: valid ones only, because a refused
        constructor ends a case (and Hypothesis favours the last branch of a one_of).
        The constructors get wild arguments in their own tests (t_constructors)."""
        del wild
        return valid

    def call(target, meth, *args):
        return ["call", target, meth, list(args)]

    def both(program):
        return cpp_w.run(program), rust_w.run(program)

    tests = []

    def register(f):
        tests.append(f)
        return f

    # ---- ActuarialMath ---------------------------------------------------------------
    def annuity_reference(kind, rate, n, pay):
        # 50 significant digits: mpmath's log1p/expm1 stay accurate for any tiny rate,
        # where (1 + r)**n would round 1 + 5e-324 to 1 at any fixed precision.
        r, P = mp.mpf(rate), mp.mpf(pay)
        if r == 0:
            return P * n
        if kind == "present_value":
            return P * (-mp.expm1(-n * mp.log1p(r)) / r)
        return P * (mp.expm1(n * mp.log1p(r)) / r)

    def annuity_intended(kind, args):
        rate_, n_, pay_ = args

        def judge(cpp, rust):
            if rust["ok"] is False and rust["exc"] == "ValueError" and cpp["ok"]:
                if rust["msg"] == "Periods must be non-negative":
                    return "negative periods now refused (deferred 0.8.0 fix)"
                if rust["msg"] == "Payment must be finite":
                    return "NaN/inf payment now refused (deferred 0.8.0 fix)"
            rate_f, pay_f, n_i = as_float(rate_), as_float(pay_), as_int(n_)
            if rate_f is None or pay_f is None or n_i is None:
                return None
            rate, pay, n = rate_f, pay_f, n_i
            if cpp["ok"] and rust["ok"]:
                c, r_ = dec_result(cpp["value"]), dec_result(rust["value"])
                if same_float(c, r_) or not (math.isfinite(rate) and math.isfinite(pay)):
                    return None
                ref = annuity_reference(kind, rate, n, pay)
                if abs(ref) > mp.mpf("1.7976931348623157e308"):
                    return "overflow boundary" if math.isinf(r_) else None
                err_r = abs(mp.mpf(r_) - ref) if math.isfinite(r_) else mp.inf
                err_c = abs(mp.mpf(c) - ref) if math.isfinite(c) else mp.inf
                x = abs(n * math.log1p(rate)) if rate > -1 else 0.0
                # the new forms: within a few ulps times the condition number of exp
                budget = (8 + 2 * x) * 2.0 ** -52 * abs(ref) + mp.mpf("1e-320")
                if err_r <= budget and err_r <= err_c * 1.01 + budget:
                    if abs(c - r_) > 1e-12 * abs(ref):
                        return "expm1/log1p form: C++ lost digits, Rust within ulps of 50-digit value"
                    return "expm1/log1p form: last-bit difference, Rust no worse than C++"
            return None
        return judge

    def make_annuity_test(kind):
        @cfg
        @given(rate=num, n=intish, pay=num)
        def t_annuity(rate, n, pay):
            fn = f"ActuarialMath.{kind}"
            cpp, rust = both([call("ActuarialMath", kind, rate, n, pay)])
            assert classify(fn, cpp, rust, mode="exact", intended=annuity_intended(kind, (rate, n, pay))), fn
        t_annuity.__name__ = f"t_{kind}"
        return t_annuity

    tests.append(make_annuity_test("present_value"))
    tests.append(make_annuity_test("future_value"))

    @register
    @cfg
    @given(rate=st.one_of(st.floats(min_value=-0.5, max_value=1.0), st.floats(min_value=1e-16, max_value=1e-5),
                          st.floats(min_value=-1e-5, max_value=-1e-16)),
           n=st.integers(0, 1200), pay=st.floats(min_value=1.0, max_value=1e7))
    def t_annuity_everyday(rate, n, pay):
        """Ordinary rates, terms and payments, plus tiny rates: where the old forms lose digits."""
        for kind in ("present_value", "future_value"):
            fn = f"ActuarialMath.{kind} (everyday inputs)"
            cpp, rust = both([call("ActuarialMath", kind, rate, n, pay)])
            assert classify(fn, cpp, rust, mode="exact", intended=annuity_intended(kind, (rate, n, pay))), fn

    @register
    @cfg
    @given(a=num, b=num)
    def t_loss_ratio(a, b):
        cpp, rust = both([call("ActuarialMath", "calculate_loss_ratio", a, b)])
        assert classify("ActuarialMath.calculate_loss_ratio", cpp, rust)

    @register
    @cfg
    @given(age=intish)
    def t_mortality(age):
        cpp, rust = both([call("ActuarialMath", "lookup_mortality_rate", age)])
        assert classify("ActuarialMath.lookup_mortality_rate", cpp, rust)

    # ---- FactorModel ----------------------------------------------------------------
    field = st.sampled_from(["state", "age", "zone", "", "Ä", "b"])
    value = st.one_of(st.sampled_from(["NY", "FL", "", "x", "Ä"]), any_float, st.integers(-5, 100), st.booleans(),
                      st.sampled_from([None, b"NY", [1]]))
    rule = st.one_of(
        st.tuples(st.just("add_multiplier"), field, st.sampled_from(["NY", "FL", "", "Ä"]), num),
        st.tuples(st.just("add_numeric_band_multiplier"), field, num, num, num),
        st.tuples(st.just("set_base_rate"), num),
    )

    @register
    @cfg
    @given(base=mostly(st.floats(min_value=-1e6, max_value=1e6), num), rules=st.lists(rule, max_size=8),
           inputs=st.dictionaries(field, value, max_size=4),
           as_proxy=st.booleans())
    def t_factor(base, rules, inputs, as_proxy):
        program = [["new", "FactorModel", [base]]]
        for r in rules:
            program.append(call(0, r[0], *r[1:]))
        program.append(call(0, "calculate", inputs))
        # Run step by step so a refused rule is compared, then the rest still runs.
        cpp, rust = both(program[:1])
        if not classify("FactorModel.__init__", cpp, rust) or not cpp["ok"]:
            return
        kept = program[:1]
        for step in program[1:]:
            cpp, rust = both(kept + [step])
            name = "FactorModel." + step[2]
            assert classify(name, cpp, rust), name
            if cpp["ok"] and step[2] != "calculate":
                kept.append(step)

    # ---- LossTriangle ----------------------------------------------------------------
    payments = st.one_of(
        st.lists(st.floats(min_value=0, max_value=1e9), min_size=1, max_size=8),
        st.lists(st.sampled_from([0.0, 0.0, 1.0, 5.0, 1e300, 1e308]), min_size=1, max_size=6),
        st.lists(st.one_of(st.floats(min_value=0, max_value=1e6), edge_floats), max_size=6),
        st.just([]), st.sampled_from(["abc", None, [1.0, None]]),
    )

    @register
    @cfg
    @given(rows=st.lists(st.tuples(intish, payments), max_size=8))
    def t_triangle(rows):
        kept = [["new", "LossTriangle", []]]
        for year, pay in rows:
            step = call(0, "add_origin_year", year, pay)
            cpp, rust = both(kept + [step])
            assert classify("LossTriangle.add_origin_year", cpp, rust)
            if cpp["ok"]:
                kept.append(step)
        for meth in ("get_development_factors", "get_ultimate_losses", "get_ibnr_reserves", "get_origin_year_count",
                     "get_development_period_count"):
            cpp, rust = both(kept + [call(0, meth)])
            assert classify("LossTriangle." + meth, cpp, rust), meth

    # ---- ExperienceRating -----------------------------------------------------------
    @register
    @cfg
    @given(k=mostly(pos_float, num), ballast=mostly(st.floats(min_value=0, max_value=10), num), a=num, b=num, c=num,
           n=intish)
    def t_experience(k, ballast, a, b, c, n):
        ctor = ["new", "ExperienceRating", [k, ballast]]
        cpp, rust = both([ctor])
        if not classify("ExperienceRating.__init__", cpp, rust) or not cpp["ok"]:
            return
        fa, fb, fc, fbal = as_float(a), as_float(b), as_float(c), as_float(ballast)

        def cwr():
            if None in (fa, fb, fc):
                return None, []
            return fc * fa + (1.0 - fc) * fb, [fma(fc, fa, (1.0 - fc) * fb), fma(1.0 - fc, fb, fc * fa)]

        def emf():
            if None in (fa, fb, fbal):
                return None, []
            den = fb * (1.0 + fbal)
            return (fa + fbal * fb) / den, [fma(fbal, fb, fa) / den]

        for meth, args, judge in (("credibility_weighted_rate", (a, b, c), fused_judge(cwr)),
                                  ("calculate_credibility", (n,), None),
                                  ("experience_mod_factor", (a, b), fused_judge(emf)),
                                  ("buhlmann_straub_credibility", (a, b), None),
                                  ("expected_losses", (a, b, c), None)):
            cpp, rust = both([ctor, call(0, meth, *args)])
            assert classify("ExperienceRating." + meth, cpp, rust, intended=judge), meth

    # ---- ExposureRating -------------------------------------------------------------
    def ilf_scale(base_limit, b, *limits):
        """Size of the ILF terms, so a premium that cancels is judged against them."""
        vals = []
        for L in limits:
            try:
                if isinstance(L, float) and L > 0 and math.isfinite(L) and isinstance(base_limit, float):
                    vals.append(abs(float(mp.mpf(L) / mp.mpf(base_limit)) ** (1 - b)))
            except (OverflowError, ValueError, ZeroDivisionError, TypeError):
                pass
        return max(vals) if vals else 1.0

    @register
    @cfg
    @given(base=mostly(pos_float, num), b=mostly(st.floats(min_value=0, max_value=1), num), x=num, y=num, z=num,
           losses=loss_list, years=intish)
    def t_exposure(base, b, x, y, z, losses, years):
        ctor = ["new", "ExposureRating", [base, b]]
        cpp, rust = both([ctor])
        if not classify("ExposureRating.__init__", cpp, rust) or not cpp["ok"]:
            return
        cpp, rust = both([ctor, call(0, "increased_limits_factor", x)])
        assert classify("ExposureRating.increased_limits_factor", cpp, rust, mode="tol", tol=1e-12)
        cpp, rust = both([ctor, call(0, "layer_premium", x, y, z)])
        if cpp["ok"] and rust["ok"]:
            prem = x if isinstance(x, float) else 1.0
            ilf_base = 1.0
            s = abs(prem) * ilf_scale(base, b, y, (y + z) if isinstance(y, float) and isinstance(z, float) else z) / ilf_base
            assert classify("ExposureRating.layer_premium", cpp, rust, mode="tol", tol=1e-12,
                            scale=lambda i: max(s, abs(dec_result(cpp["value"]))))
        else:
            assert classify("ExposureRating.layer_premium", cpp, rust)
        cpp, rust = both([ctor, call(0, "burning_cost", losses, y, z, years)])
        assert classify("ExposureRating.burning_cost", cpp, rust)
        cpp, rust = both([ctor, call(0, "loss_elimination_ratio", x, y, z)])
        assert classify("ExposureRating.loss_elimination_ratio", cpp, rust, mode="tol", tol=1e-12,
                        scale=lambda i: 1.0)
        cpp, rust = both([ctor, call(0, "excess_risk_load", x, y)])
        assert classify("ExposureRating.excess_risk_load", cpp, rust)

    # ---- RateAnalyzer ---------------------------------------------------------------
    @register
    @cfg
    @given(tcr=mostly(st.floats(min_value=0.01, max_value=3), num), er=mostly(st.floats(min_value=0, max_value=0.99), num),
           a=num, b=num, changes=rate_list, prems=loss_list)
    def t_rate(tcr, er, a, b, changes, prems):
        ctor = ["new", "RateAnalyzer", [tcr, er]]
        cpp, rust = both([ctor])
        if not classify("RateAnalyzer.__init__", cpp, rust) or not cpp["ok"]:
            return
        for meth, args, mode in (("on_level_factor", (changes,), "exact"), ("rate_change_impact", (a, b), "exact"),
                                 ("combined_ratio", (a,), "exact"), ("required_rate_change", (a,), "exact"),
                                 ("on_level_premiums", (prems, changes), "exact"),
                                 ("on_level_premiums", (changes, changes), "exact"),
                                 ("permissible_loss_ratio", (), "exact"), ("trend_factor", (a, b), "tol"),
                                 ("loss_cost_rate", (a, b), "exact")):
            cpp, rust = both([ctor, call(0, meth, *args)])
            assert classify("RateAnalyzer." + meth, cpp, rust, mode=mode, tol=1e-12), meth

    # ---- FourierTransform -------------------------------------------------------------
    vec = st.one_of(st.lists(st.floats(min_value=-1e6, max_value=1e6), max_size=70),
                    st.lists(st.one_of(st.floats(min_value=-1, max_value=1), edge_floats), max_size=9))
    pmf = st.one_of(st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=60),
                    st.lists(st.sampled_from([0.0, 0.0, 1.0, 0.25, 1e-12, 5.0]), min_size=1, max_size=20),
                    st.lists(st.one_of(st.floats(min_value=0, max_value=1), edge_floats), max_size=8))

    def l1(v):
        try:
            return sum(abs(x) for x in v if isinstance(x, float) and math.isfinite(x)) or 1.0
        except TypeError:
            return 1.0

    @register
    @cfg
    @given(re_=vec, im_=vec, same=st.booleans())
    def t_fft(re_, im_, same):
        if same:
            im_ = (im_ + [0.0] * len(re_))[: len(re_)]
        s = l1(re_) + l1(im_)
        for meth in ("fft", "ifft"):
            cpp, rust = both([call("FourierTransform", meth, re_, im_)])
            assert classify("FourierTransform." + meth, cpp, rust, mode="tol", tol=1e-12, scale=lambda i: s), meth

    @register
    @cfg
    @given(a=vec, b=vec)
    def t_convolve(a, b):
        s = l1(a) * l1(b)
        cpp, rust = both([call("FourierTransform", "convolve", a, b)])
        assert classify("FourierTransform.convolve", cpp, rust, mode="tol", tol=1e-12, scale=lambda i: s)

    def fft_wrap_borderline(cpp, rust):
        # The wrap-around check compares a mean error with a noise floor; inputs whose
        # error sits at the floor can land either side by the last bit of a sin/cos.
        msgs = [x.get("msg", "") for x in (cpp, rust) if not x["ok"]]
        if cpp["ok"] != rust["ok"] and msgs and "FFT grid too small" in msgs[0]:
            return "wrap-around check at its noise floor (one ulp of sin/cos decides)"
        if not cpp["ok"] and not rust["ok"] and cpp["msg"] != rust["msg"] and "FFT grid too small" in cpp["msg"] \
                and "FFT grid too small" in rust["msg"]:
            return "same refusal, the quoted mean differs in its sixth decimal"
        return None

    @register
    @cfg
    @given(sev=st.one_of(pmf, st.sampled_from([[], "x", [1.0, "x"]])),
           lam=st.one_of(st.floats(min_value=0, max_value=200), num),
           grid=st.one_of(st.just(0), st.sampled_from([1, 2, 8, 16, 64, 1024, 3, -4, 2**24, 2**25, 2**30]), intish))
    def t_compound(sev, lam, grid):
        cpp, rust = both([call("FourierTransform", "compound_poisson_pmf", sev, lam, grid)])
        assert classify("FourierTransform.compound_poisson_pmf", cpp, rust, mode="tol", tol=1e-12,
                        scale=lambda i: 1.0, intended=fft_wrap_borderline)

    # ---- RiskEngine / Field / ExcelExporter ---------------------------------------------
    inputs_st = st.dictionaries(st.sampled_from(["age", "state", "value", "missing", "b"]),
                                st.one_of(st.floats(min_value=-1e6, max_value=1e6), edge_floats,
                                          st.sampled_from(["NY", "FL", "", "Ä <&>", "a\nb"]), st.integers(-3, 10**6)),
                                max_size=4)
    col = st.sampled_from(["A", "B", "C", "Z", "AA", "XFD", "a", "XFE", "", "1", "A1", "AAAA", "A$"])

    @register
    @cfg
    @given(fields=st.lists(st.tuples(st.text(max_size=5), st.text(max_size=5), st.text(max_size=5)), max_size=4),
           maps=st.lists(st.tuples(col, st.sampled_from(["age", "state", "value", "premium_output", "nope"]),
                                   st.sampled_from(["Age", "", "Premium", "État"])), max_size=5),
           logic=st.sampled_from(["linear", "factor", "seen", "bad", "none"]),
           quote=inputs_st, batch=st.lists(inputs_st, max_size=4),
           prems=st.lists(st.floats(min_value=-1e9, max_value=1e9), max_size=5))
    def t_engine(fields, maps, logic, quote, batch, prems):
        program = [["new", "RiskEngine", []], ["new", "ExcelExporter", ["template.xlsx"]],
                   ["new", "FactorModel", [100.0]]]
        program.append(call(2, "add_multiplier", "state", "NY", 1.25))
        program.append(call(2, "add_numeric_band_multiplier", "age", 18.0, 25.0, 1.4))
        for f in fields:
            program.append(["field", 0, *f])
        for m in maps:
            program.append(call(1, "map_column", *m))
        program.append(["logic", 0, logic, 2])
        program.append(["attach", 0, 1])
        cpp, rust = both(program + [["fields", 0]])
        assert classify("RiskEngine.get_fields", cpp, rust)
        cpp, rust = both(program + [call(0, "execute", quote)])
        assert classify("RiskEngine.execute", cpp, rust)
        if cpp["ok"]:
            program.append(call(0, "execute", quote))
        cpp, rust = both(program + [["export", 0, "single", []]])
        assert classify("RiskEngine.export_to_excel", cpp, rust, mode="exact"), (cpp, rust)
        cpp, rust = both(program + [["export", 0, "batch", [batch, prems]]])
        assert classify("RiskEngine.export_batch_to_excel", cpp, rust, mode="exact"), (cpp, rust)

    # ---- every constructor with anything as arguments --------------------------------
    @register
    @cfg
    @given(a=num, b=num, n=intish, m=intish, s1=st.one_of(st.text(max_size=4), st.binary(max_size=4), num))
    def t_constructors(a, b, n, m, s1):
        for cls, args in (("FactorModel", [a]), ("FactorModel", []), ("ExperienceRating", [a, b]),
                          ("ExperienceRating", [a]), ("ExposureRating", [a, b]), ("ExposureRating", []),
                          ("RateAnalyzer", [a, b]), ("RateAnalyzer", [b]), ("MonteCarloSimulator", [n, m]),
                          ("MonteCarloSimulator", [n]), ("ExcelExporter", [s1]), ("Field", [s1, s1, s1]),
                          ("Field", [s1])):
            cpp, rust = both([["new", cls, args]])
            assert classify(cls + ".__init__", cpp, rust), cls

    # ---- MonteCarloSimulator: checks and deterministic corners --------------------------------
    @register
    @cfg
    @given(trials=mostly(st.integers(1, 50), st.one_of(st.integers(-3, 0), intish)),
           seed=mostly(st.integers(0, 2**32 - 1), intish),
           a=num, b=num, c=num, d=num, n=intish)
    def t_simulator_checks(trials, seed, a, b, c, d, n):
        ctor = ["new", "MonteCarloSimulator", [trials, seed]]
        cpp, rust = both([ctor])
        if not classify("MonteCarloSimulator.__init__", cpp, rust) or not cpp["ok"]:
            return
        if isinstance(trials, int) and trials > 2000:
            return
        calls = (("simulate_aggregate_loss", (a, b, c)), ("simulate_life_portfolio", (n, a, b, c)),
                 ("simulate_economic_path", (a, b, c, n)), ("simulate_health_claims", (a, b, c, d)),
                 ("simulate_catastrophe_loss", (a, b, c)))
        for meth, args in calls:
            # only the refusals and the trial count are comparable: the draws differ by design
            if meth == "simulate_aggregate_loss" and isinstance(a, float) and a > 1e4:
                continue
            if meth == "simulate_life_portfolio" and isinstance(n, int) and n > 10**6:
                continue
            if meth == "simulate_economic_path" and isinstance(n, int) and n > 1000:
                continue
            if meth == "simulate_health_claims" and isinstance(a, float) and a > 1e4:
                continue
            cpp, rust = both([ctor, call(0, meth, *args), ])
            name = "MonteCarloSimulator." + meth + " (checks)"
            LEDGER.cases[name] += 1
            if cpp["ok"] != rust["ok"] or (not cpp["ok"] and (cpp["exc"], cpp["msg"]) != (rust["exc"], rust["msg"])):
                if not cpp["ok"] and not rust["ok"] and cpp["exc"] == rust["exc"] == "TypeError":
                    continue
                LEDGER.mismatch[name].append(("checks differ", cpp, rust))
                raise AssertionError(name)
            if cpp["ok"] and len(dec_result(cpp["value"])) != len(dec_result(rust["value"])):
                LEDGER.mismatch[name].append(("trial counts differ", cpp, rust))
                raise AssertionError(name)

    @register
    @cfg
    @given(trials=st.integers(1, 200), seed=st.integers(1, 2**32 - 1), price=pos_float,
           drift=st.floats(min_value=-1, max_value=1), steps=st.integers(1, 30), xm=pos_float,
           alpha=pos_float, pol=st.integers(0, 10**6), benefit=st.floats(min_value=0, max_value=1e9),
           mu=st.floats(min_value=-50, max_value=50), sigma=st.floats(min_value=1e-12, max_value=5))
    def t_simulator_deterministic(trials, seed, price, drift, steps, xm, alpha, pol, benefit, mu, sigma):
        ctor = ["new", "MonteCarloSimulator", [trials, seed]]
        for meth, args in (("simulate_economic_path", (price, drift, 0.0, steps)),      # volatility 0
                           ("simulate_catastrophe_loss", (0.0, xm, alpha)),             # no event
                           ("simulate_life_portfolio", (pol, 0.0, 0.3, benefit)),       # no mortality
                           ("simulate_life_portfolio", (pol, 1.0, 0.0, benefit)),       # certain death
                           ("simulate_aggregate_loss", (0.0, mu, sigma)),               # no claims
                           ("simulate_health_claims", (0.0, 0.5, 2.0, 3.0))):          # no claims
            cpp, rust = both([ctor, call(0, meth, *args)])
            mode = "tol" if meth == "simulate_economic_path" else "exact"
            assert classify("MonteCarloSimulator." + meth + " (deterministic corner)", cpp, rust, mode=mode,
                            tol=1e-12), meth

    return tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--cpp")
    ap.add_argument("--rust")
    ap.add_argument("--examples", type=int, default=300)
    ap.add_argument("--json")
    ns = ap.parse_args()
    if ns.worker:
        worker()
        return 0
    cpp_w, rust_w = Worker(ns.cpp), Worker(ns.rust)
    failures = []
    for t in build_tests(cpp_w, rust_w, ns.examples):
        name = t.__name__
        try:
            t()
            print(f"ok    {name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL  {name}: {exc!r}", flush=True)
            traceback.print_exc(limit=1)
    print()
    print(f"{'function':<62} {'cases':>7} {'mismatch':>9} {'bit-identical floats':>22} {'max rel diff':>13}")
    total = 0
    for fn in sorted(LEDGER.cases):
        n = LEDGER.cases[fn]
        total += n
        same, count = LEDGER.bitident.get(fn, [0, 0])
        bi = f"{same}/{count}" if count else "-"
        mr = f"{LEDGER.maxrel[fn]:.2e}" if LEDGER.maxrel.get(fn) else "0"
        print(f"{fn:<62} {n:>7} {len(LEDGER.mismatch.get(fn, [])):>9} {bi:>22} {mr:>13}")
    print(f"{'total':<62} {total:>7}")
    if LEDGER.intended:
        print("\nIntended differences:")
        for fn, reasons in sorted(LEDGER.intended.items()):
            for reason, n in sorted(reasons.items()):
                print(f"  {fn}: {reason} — {n} case(s)")
    if any(LEDGER.mismatch.values()):
        print("\nMismatches (first of each function):")
        for fn, items in LEDGER.mismatch.items():
            if items:
                why, cpp, rust = items[0]
                print(f"  {fn}: {why}\n    C++:  {json.dumps(cpp)[:400]}\n    Rust: {json.dumps(rust)[:400]}")
    if ns.json:
        with open(ns.json, "w") as f:
            json.dump({"cases": LEDGER.cases, "mismatch": {k: len(v) for k, v in LEDGER.mismatch.items()},
                       "intended": {k: dict(v) for k, v in LEDGER.intended.items()},
                       "bit_identical": {k: v for k, v in LEDGER.bitident.items()},
                       "max_rel": LEDGER.maxrel, "failures": failures}, f, indent=1)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
