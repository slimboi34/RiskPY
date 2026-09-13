# How It Works

`riskpy` is a hybrid framework designed to give actuaries the ease of **Python** with the speed of **C++**.

## The Architecture
The compiled core is a **C++ extension module** (`riskpy.cpp_underwriter`) built with `pybind11` and `scikit-build-core`. The modelling layers — `riskpy.mc`, `viz`, `quant`, `life`, `reserving`, `rates` and `credit` — are readable Python on top of NumPy.

1. **The Python Frontend**: `UnderwritingApp` is Python. It renders the Tkinter GUI, collects user inputs and reads CSV files.
2. **The C++ Backend Engine**: `FactorModel`, `RiskEngine`, `MonteCarloSimulator`, `LossTriangle` and the other core classes are C++ exposed through pybind11. `FactorModel` stores its rules and evaluates them in C++, and inputs cross into C++ as `std::variant` values, so text (like "FL" for State) and numbers (like 25 for Age) share one input map.
3. **Excel Output**: `ExcelExporter` writes `.xlsx` binaries with the `OpenXLSX` C++ library.

`app.calculate_batch()` reads the CSV and loops over its rows in Python, calling the C++ engine once per row, then hands the whole book to the C++ Excel writer in a single call. The C++ code does not release Python's Global Interpreter Lock (GIL): the speed comes from compiled arithmetic, not from parallel threads.

## Why This Matters
If you try to loop through 1,000,000 rows of an Excel spreadsheet in pure Python using `if/else` ladders to calculate premiums, it takes minutes. If you try to run 100,000 Monte Carlo simulations using standard Python generators, it takes even longer.

By compiling the math into standard C++ `<random>` and `<cmath>` libraries, `riskpy` handles millions of iterations quickly. You construct the model in Python, but you execute it in C++.
