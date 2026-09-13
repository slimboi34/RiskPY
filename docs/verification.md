# Verification

A test suite tells you whether the code still does what it did yesterday.
`riskpy.verify` answers a different question: whether what it does is *right*.
It computes a value with the library and compares it to something the library
had no hand in — a closed form, a published table, a mathematical identity, or
the same quantity by an unrelated route.

```python
from riskpy import verify

report = verify.run()
print(report.summary())
assert report                    # False if any check failed
```

```bash
python -m riskpy.verify              # the same from the command line
riskpy-verify --bench                # with timings, for speed regressions
riskpy-verify --json report.json     # for a dashboard
riskpy-verify -m life -m reserving   # a subset
```

## What a check is

Every check is a name, the value the library produced, the reference it should
match, and an **absolute tolerance**. The tolerance is chosen per check and
stated: machine precision for exact identities, the printed precision for
values transcribed from a book, and a few standard errors for anything Monte
Carlo. Every stochastic check runs on a fixed seed, so it is reproducible, and
is loose enough never to fail by chance.

Some of what is checked, by module:

| module | examples |
|---|---|
| `core` | `FactorModel` against hand arithmetic; the hand-checked chain ladder; the compound Poisson FFT with unit severity is `Poisson(λ)`; the C++ aggregate-loss mean is `λ·E[X]` |
| `special` | `Φ(0) = ½`; `Φ⁻¹(Φ(x)) = x`; `P(1, x) = 1 − e⁻ˣ`; `I_x(a, b) + I_{1−x}(b, a) = 1`; the t with one degree of freedom is Cauchy; `ψ(1) = −γ`; `ψ₁(1) = π²/6`; and, when SciPy is installed, agreement with it to 1e-13 (1e-8 for the Student-t quantile) |
| `mc` | `cdf(ppf(q)) = q` for every family; sample means within four standard errors of the analytic mean; Latin hypercube within one; Iman–Conover recovers the target Spearman and leaves the marginals untouched; a truncated density integrates to one |
| `quant` | the canonical Black–Scholes value; put–call parity; delta as a central difference; implied vol round trip; Heston → Black–Scholes as vol-of-vol → 0; Heston put–call parity; the GBM terminal mean |
| `life` | the SULT's `l_100`, `ä_40` and `A_40` against AMLCR Appendix D; `A_x = 1 − d·ä_x`; `A_{x:n} = A¹_{x:n} + ₙE_x`; Makeham's closed form against numerical integration; `M_x / D_x = A_x`; `ä_xy + ä_x̄ȳ = ä_x + ä_y` |
| `reserving` | GenIns factors, total reserve and Mack standard errors against Mack (1993); `se² = process² + parameter²`; agreement with the compiled `LossTriangle`; Bornhuetter–Ferguson collapsing to the chain ladder |
| `rates` | par at par; `ytm(price(y)) = y`; the bootstrap round trip; Nelson–Siegel limits; both short-rate bonds at `σ = 0`; Vasicek against Monte Carlo |
| `credit` | `equity + debt = assets`; equity as a Black–Scholes call; the ASRF at `ρ = 0`; the credit triangle; the Basel correlation limits; the BCBS risk-weight table |
| `viz` | every palette colour clears 3:1 contrast on its surface; the sequential ramps are monotone in lightness; eight categorical slots and no more |

Without SciPy the suite runs 111 checks; with it, the 14 oracle checks bring it
to 125. NumPy is required for the `mc`, `quant` and `reserving` checks.

## Where it runs

- **CI**, on every push and pull request — a wrong number fails the build.
- **Nightly**, with the SciPy oracle switched on and the benchmarks included,
  keeping the JSON report as an artifact so a slow drift in accuracy or speed
  leaves a trail even on days nobody pushes.
- **Here**, regenerated on every docs build. The table below is the report
  from the commit this site was built from.

## Adding a check

`life`, `reserving`, `rates` and `credit` contribute checks through a private
hook in their own module. Checks for the core, `special`, `mc`, `quant` and
`viz` are written directly in `riskpy/verify.py`.

```python
def _verification_checks():
    """Return a list of (name, value, reference, tolerance) tuples."""
    return [
        ("whole life A_x = 1 - d * ä_x", A, 1 - d * a, 1e-12),
    ]
```

The rule for what belongs here: pick things that would catch a *class* of
error, not a single typo. Put–call parity catches every sign error in a
pricing formula; one reference price catches one.

## Latest report

--8<-- "assets/verification.md"
