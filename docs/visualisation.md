# Visualisation

`riskpy.viz` is six charts that answer six questions. Each takes a
[`Result`](monte-carlo.md), returns a Matplotlib `Figure`, and never calls
`show()` — so the same call works in a notebook, in a script that saves a PNG,
and in a test.

```bash
pip install "open-riskpy[viz]"
```

```python
from riskpy import viz

fig = viz.distribution(result)
viz.save(fig, "loss.png")
```

Or straight off the result:

```python
result.plot()                  # distribution
result.plot("convergence")
result.plot("tornado")
result.plot("exceedance")
```

---

## Where is the risk?

```python
viz.distribution(result, levels=(0.95, 0.995))
```

![Distribution](assets/distribution.png)

Histogram with VaR and TVaR marked at each level you ask for. The shape tells
the story; the rules are what anyone acts on. The y-axis is deliberately
unlabelled — trial counts carry no decision, and numbering them invites people
to read a simulation artefact as a quantity.

## How likely is something worse?

```python
viz.exceedance(result, log_y=True)
```

![Exceedance curve](assets/exceedance.png)

P(loss > x) plotted against x. Far easier to read off than a histogram tail, and
the natural chart for a reinsurance conversation. Log scale by default, because
every interesting probability here is near zero.

## Did I run enough trials?

```python
viz.convergence(result)
```

![Convergence](assets/convergence.png)

Running mean with a 95% band. If the band is still narrowing at the right-hand
edge, run more trials.

## What is driving it?

```python
viz.tornado(result, top=12)
```

![Tornado](assets/tornado.png)

Rank correlation of each input with the output, strongest first, signed. Every
bar is the same colour on purpose — the length already encodes the magnitude,
and tinting by size would say it twice while burning the only free channel.

## What do the paths look like?

```python
from riskpy import quant

paths = quant.gbm_paths(S0=100, mu=0.07, sigma=0.22, T=1.0, trials=20_000)
viz.fan(paths, levels=(0.5, 0.8, 0.95), ylabel="price")
```

![Fan chart](assets/gbm_fan.png)

Nested percentile bands with the median as a line. Twenty thousand spaghetti
lines communicate nothing; three bands communicate the distribution.

`paths` is any `(trials, steps)` array, so this works for reserve run-off and
projections as well as prices.

## How do scenarios compare?

```python
viz.compare({
    "no cover":        baseline,
    "quota share 15%": quota,
    "negotiated":      negotiated,
})
```

![Scenario comparison](assets/compare.png)

One axis, one colour per scenario, always a legend — colour here is identity, so
it can never be the only channel.

Past eight scenarios this raises rather than inventing a ninth hue. Nine
categorical colours cannot be told apart reliably; group the tail or facet into
separate charts instead.

---

## Theme

Dark by default, because that is the surface the charts were designed against.

```python
viz.theme("light")     # or "dark"
```

Both palettes are validated: every series colour clears 3:1 contrast against its
own background, and adjacent slots stay separable under the common forms of
colour vision deficiency. `viz.PALETTE` holds both if you want to match other
charts to them.

## Rules worth not undoing by accident

These are choices, not defaults, and each one is easy to break without noticing:

- **Single-series charts use a single hue.** Colouring bars by height
  double-encodes what the length already says.
- **Direction never rests on colour alone.** Signed values ship with a sign and
  a marker, so the chart survives greyscale printing and colour blindness.
- **No dual axes.** Two quantities of different scale get two charts, or index
  both to a common base. A second y-scale invents a correlation that is not in
  the data.
- **Gridlines are solid hairlines.** Dashed grid reads as a threshold when it is
  just a grid.
- **Nothing is gated behind a tooltip.** Every value is reachable as text.

## Saving

```python
viz.save(fig, "out.png", dpi=160)
```

In a headless environment — CI, a server — select the Agg backend before
importing:

```python
import matplotlib
matplotlib.use("Agg")
```

Every chart on this site is produced by
[`docs/generate_gallery.py`](https://github.com/slimboi34/RiskPY/blob/main/docs/generate_gallery.py),
which runs on each docs build. It is the most complete worked example in the
repository.
