# How It Works

`riskpy` is a hybrid framework designed to give actuaries the ease of **Python** with the blistering speed of **C++**.

## The Architecture
At its core, the framework is a **C++ Extension Module** compiled dynamically into Python via `pybind11` and `scikit-build-core`. 

1. **The Python Frontend**: When you instantiate `UnderwritingApp` or define rules in `FactorModel`, you are working in pure, readable Python. This frontend handles the Tkinter GUI rendering, user inputs, and file paths (like CSVs).
2. **The C++ Backend Engine**: When you call `app.run()` or `app.calculate_batch()`, the Python tier completely hands off execution to the compiled C++ binary (`cpp_underwriter.so`).
3. **Zero-Cost Abstraction**: The C++ engine uses `std::variant` to seamlessly handle both text (like "FL" for State) and numbers (like 25 for Age) without Python's dynamic typing overhead. The C++ `RiskEngine` iterates through the rules, applies the `FactorModel` math, and uses the highly optimized `OpenXLSX` C++ library to write results directly to Excel binaries—bypassing Python's Global Interpreter Lock (GIL).

## Why This Matters
If you try to loop through 1,000,000 rows of an Excel spreadsheet in pure Python using `if/else` ladders to calculate premiums, it takes minutes. If you try to run 100,000 Monte Carlo simulations using standard Python generators, it takes even longer.

By compiling the math into standard C++ `<random>` and `<cmath>` libraries, `riskpy` handles millions of iterations in milliseconds. You construct the model in Python, but you execute it in C++.
