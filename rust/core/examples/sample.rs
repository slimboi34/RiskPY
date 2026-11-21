//! Prints raw draws from the core's samplers, one per line, for checking them against
//! closed forms outside Rust (tests/test_rust_core.py drives it when cargo is present;
//! benchmarks/rust_core/ uses it for the distribution report).
//!
//!     cargo run --release --example sample -- raw SEED N          # state, inc, then N u64
//!     cargo run --release --example sample -- uniform SEED N
//!     cargo run --release --example sample -- normal SEED N
//!     cargo run --release --example sample -- gamma SEED N SHAPE
//!     cargo run --release --example sample -- poisson SEED N LAMBDA
//!     cargo run --release --example sample -- binomial SEED N TRIALS P
//!     cargo run --release --example sample -- lognormal SEED N MU SIGMA

use std::io::{BufWriter, Write};

use riskpy_core::dist::Stream;
use riskpy_core::rng::Pcg64Dxsm;

fn arg<T: std::str::FromStr>(args: &[String], i: usize, what: &str) -> T {
    args.get(i)
        .unwrap_or_else(|| panic!("missing {what}"))
        .parse()
        .unwrap_or_else(|_| panic!("bad {what}"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = args.get(1).map(String::as_str).unwrap_or("raw");
    let seed: u32 = arg(&args, 2, "seed");
    let n: usize = arg(&args, 3, "count");
    let out = std::io::stdout();
    let mut w = BufWriter::new(out.lock());
    if kind == "raw" {
        let mut g = Pcg64Dxsm::from_seed(seed);
        let (state, inc) = g.raw_state();
        writeln!(w, "{state}").unwrap();
        writeln!(w, "{inc}").unwrap();
        for _ in 0..n {
            writeln!(w, "{}", g.next_u64()).unwrap();
        }
        return;
    }
    let mut s = Stream::new(seed);
    for _ in 0..n {
        let x: f64 = match kind {
            "uniform" => s.uniform(),
            "normal" => s.standard_normal(),
            "gamma" => s.standard_gamma(arg(&args, 4, "shape")),
            "poisson" => s.poisson(arg(&args, 4, "lambda")) as f64,
            "binomial" => s.binomial(arg(&args, 4, "trials"), arg(&args, 5, "p")) as f64,
            "lognormal" => s.lognormal(arg(&args, 4, "mu"), arg(&args, 5, "sigma")),
            other => panic!("unknown distribution {other}"),
        };
        writeln!(w, "{x:e}").unwrap();
    }
}
