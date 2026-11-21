# How It Works

`riskpy` is a hybrid framework designed to give actuaries the ease of **Python** with the speed of compiled **Rust**.

## The Architecture
The compiled core is a **Rust extension module** (`riskpy.cpp_underwriter`, a name kept from the C++ core it replaced in 0.9.0) built with `PyO3` and `maturin`. The modelling layers — `riskpy.mc`, `viz`, `quant`, `life`, `reserving`, `rates` and `credit` — are readable Python on top of NumPy.

1. **The Python Frontend**: `UnderwritingApp` is Python. It renders the Tkinter GUI, collects user inputs and reads CSV files.
2. **The Rust Backend Engine**: `FactorModel`, `RiskEngine`, `MonteCarloSimulator`, `LossTriangle` and the other core classes are Rust exposed through PyO3. `FactorModel` stores its rules and evaluates them in Rust, and inputs cross into Rust as a map of text-or-number values, so text (like "FL" for State) and numbers (like 25 for Age) share one input map.
3. **Excel Output**: `ExcelExporter` writes `.xlsx` files from Rust, on the `zip` crate. Every export starts from a blank workbook ("Sheet1"); the `template` argument is accepted but, as in every earlier release, not read.

`app.calculate_batch()` reads the CSV and loops over its rows in Python, calling the compiled engine once per row, then hands the whole book to the Excel writer in a single call. The rating and export calls hold Python's Global Interpreter Lock (GIL): the speed comes from compiled arithmetic, not from parallel threads. The `MonteCarloSimulator.simulate_*` methods and the FFT functions do release the GIL while they compute, so other Python threads keep running during a long simulation.

## Why This Matters
If you try to loop through 1,000,000 rows of an Excel spreadsheet in pure Python using `if/else` ladders to calculate premiums, it takes minutes. If you try to run 100,000 Monte Carlo simulations using standard Python generators, it takes even longer.

By compiling the maths into Rust, `riskpy` handles millions of iterations quickly. You construct the model in Python, but you execute it in compiled code. The simulations draw from a PCG64-DXSM stream with samplers written in the core, so a seed gives the same numbers on every platform (seeded results changed in 0.9.0).
