"""The mistakes an agent makes when it calls RiskPY's tools.

Each one is a real slip seen in practice: a percentage where a decimal belongs, a name from
another library, a string where JSON should carry a number, an impossible option price.
``param`` is the name (or value) a good error must point at; ``fix`` says what the agent
should do instead and is documentation only — the scorer never reads it.
"""
from __future__ import annotations

BS = {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}
TRIANGLE = [[100.0, 150.0, 160.0], [110.0, 170.0, None], [120.0, None, None]]

MISTAKES = [
    {"id": "negative_vol", "tool": "black_scholes", "args": {**BS, "sigma": -0.2}, "param": "sigma",
     "what": "a negative volatility", "fix": "sigma=0.2"},
    {"id": "vol_as_percent", "tool": "black_scholes", "args": {**BS, "sigma": 20}, "param": "sigma",
     "what": "volatility given as a percentage (20 for 20%)", "fix": "sigma=0.2"},
    {"id": "rate_as_percent", "tool": "black_scholes", "args": {**BS, "r": 5}, "param": "r",
     "what": "interest rate given as a percentage (5 for 5%)", "fix": "r=0.05"},
    {"id": "coupon_as_percent", "tool": "bond_analytics",
     "args": {"face": 100, "coupon": 5, "maturity": 10, "ytm": 0.045}, "param": "coupon",
     "what": "bond coupon given as a percentage", "fix": "coupon=0.05"},
    {"id": "maturity_zero", "tool": "bond_analytics",
     "args": {"face": 100, "coupon": 0.05, "maturity": 0, "ytm": 0.045}, "param": "maturity",
     "what": "a bond with maturity 0", "fix": "maturity > 0, in years"},
    {"id": "implied_vol_at_expiry", "tool": "implied_vol",
     "args": {"price": 10, "S": 100, "K": 100, "T": 0, "r": 0.05}, "param": "T",
     "what": "implied volatility asked for at expiry (T = 0)", "fix": "T > 0"},
    {"id": "probability_above_1", "tool": "expected_loss", "args": {"pd": 1.5, "lgd": 0.45, "ead": 1e6},
     "param": "pd", "what": "a default probability above 1", "fix": "pd=0.015"},
    {"id": "lgd_as_percent", "tool": "basel_irb_capital", "args": {"pd": 0.01, "lgd": 45, "ead": 1e6},
     "param": "lgd", "what": "loss given default as a percentage (45 for 45%)", "fix": "lgd=0.45"},
    {"id": "recovery_as_percent", "tool": "cds_par_spread",
     "args": {"hazard": 0.02, "recovery": 40, "r": 0.03, "maturity": 5}, "param": "recovery",
     "what": "recovery rate as a percentage", "fix": "recovery=0.4"},
    {"id": "implied_vol_impossible_price", "tool": "implied_vol",
     "args": {"price": 150, "S": 100, "K": 100, "T": 1, "r": 0.05}, "param": "price",
     "what": "a call priced above the spot (no volatility can produce it)", "fix": "check the quote"},
    {"id": "implied_vol_below_intrinsic", "tool": "implied_vol",
     "args": {"price": 1, "S": 100, "K": 50, "T": 1, "r": 0.05}, "param": "price",
     "what": "a call priced below its intrinsic value", "fix": "check the quote and kind"},
    {"id": "unknown_tool", "tool": "blackscholes", "args": BS, "param": "blackscholes",
     "what": "a tool name that does not exist (misspelt)", "fix": "black_scholes"},
    {"id": "wrong_argument_name", "tool": "black_scholes", "args": {"S": 100, "K": 100, "T": 1, "r": 0.05, "vol": 0.2},
     "param": "vol", "what": "another library's argument name (vol for sigma)", "fix": "sigma"},
    {"id": "missing_argument", "tool": "black_scholes", "args": {"S": 100, "K": 100, "T": 1, "r": 0.05},
     "param": "sigma", "what": "a required argument left out", "fix": "pass sigma"},
    {"id": "number_as_string", "tool": "black_scholes", "args": {**BS, "S": "100"}, "param": "S",
     "what": "a number sent as a JSON string", "fix": "S=100"},
    {"id": "unknown_choice", "tool": "basel_irb_capital",
     "args": {"pd": 0.01, "lgd": 0.45, "ead": 1e6, "asset_class": "mortgage"}, "param": "asset_class",
     "what": "an asset class spelt as the 0.4.0 tool description suggested", "fix": "retail_mortgage"},
    {"id": "ragged_triangle", "tool": "chain_ladder", "args": {"triangle": [[100.0, 150.0], [110.0, 170.0, 180.0]]},
     "param": "triangle", "what": "a claims triangle whose rows have mismatched lengths", "fix": "pad with null",
     "needs": "numpy"},
    {"id": "age_outside_table", "tool": "life_annuity_due", "args": {"x": 10, "i": 0.05}, "param": "x",
     "what": "an age below the life table's first age", "fix": "x >= 20"},
    {"id": "missing_extra", "tool": "chain_ladder", "args": {"triangle": TRIANGLE}, "param": "numpy",
     "what": "a tool whose optional dependency (NumPy) is not installed", "fix": 'pip install "open-riskpy[sim]"',
     "block_numpy": True},
]
