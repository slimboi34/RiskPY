//! The random stream: PCG64-DXSM (O'Neill's 128-bit LCG with the "double xorshift
//! multiply" output, the generator NumPy recommends as `PCG64DXSM`).
//!
//! It is written out here rather than taken from a crate so that the stream is fixed
//! by this file alone. `tests/` in the repository checks it against NumPy's
//! `PCG64DXSM`, given the same internal state, to the last bit.
//!
//! Seeding: a non-zero 32-bit seed is expanded by SplitMix64 into the 128-bit initial
//! state and the 128-bit stream selector, then fed through PCG's standard
//! `srandom` procedure (as `pcg_cm_srandom_r` in NumPy). Seed 0 means "fresh": the
//! state and selector come from the operating system's entropy source, as the C++ core
//! used `std::random_device` for seed 0.

/// PCG's "cheap multiplier" for the 128-bit LCG, also used inside the DXSM output.
const CHEAP_MULTIPLIER: u64 = 0xda94_2042_e4dd_58b5;

/// Domain separation for the SplitMix64 expansion of a user seed, so the RiskPY stream
/// for seed `s` is not SplitMix64's own stream for `s`.
const SEED_DOMAIN: u64 = 0x5249_534b_5059_0001; // "RISKPY" 0001

#[derive(Debug, Clone)]
pub struct Pcg64Dxsm {
    state: u128,
    inc: u128,
}

impl Pcg64Dxsm {
    /// A generator in exactly this internal state. `inc` is forced odd, as PCG requires.
    pub fn from_raw(state: u128, inc: u128) -> Self {
        Pcg64Dxsm { state, inc: inc | 1 }
    }

    /// PCG's `srandom(initstate, initseq)`.
    pub fn from_parts(initstate: u128, initseq: u128) -> Self {
        let mut g = Pcg64Dxsm { state: 0, inc: (initseq << 1) | 1 };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    /// The stream for a 32-bit seed: 0 draws fresh entropy, anything else is reproducible.
    pub fn from_seed(seed: u32) -> Self {
        if seed == 0 {
            let (s, q) = entropy();
            return Self::from_parts(s, q);
        }
        let mut sm = SplitMix64(u64::from(seed) ^ SEED_DOMAIN);
        let s = (u128::from(sm.next()) << 64) | u128::from(sm.next());
        let q = (u128::from(sm.next()) << 64) | u128::from(sm.next());
        Self::from_parts(s, q)
    }

    /// The internal state and increment (for checking against other implementations).
    pub fn raw_state(&self) -> (u128, u128) {
        (self.state, self.inc)
    }

    #[inline]
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(u128::from(CHEAP_MULTIPLIER))
            .wrapping_add(self.inc);
    }

    /// The next 64 random bits: DXSM output of the current state, then one LCG step.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut hi = (self.state >> 64) as u64;
        let lo = (self.state as u64) | 1;
        hi ^= hi >> 32;
        hi = hi.wrapping_mul(CHEAP_MULTIPLIER);
        hi ^= hi >> 48;
        hi = hi.wrapping_mul(lo);
        self.step();
        hi
    }
}

/// Steele, Lea and Flood's SplitMix64, used only to spread a seed over 256 bits.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// 256 bits from the OS. If the OS source fails (it essentially never does), fall back
/// to the standard library's per-process random hash keys mixed with the clock.
fn entropy() -> (u128, u128) {
    let mut buf = [0u8; 32];
    if getrandom::fill(&mut buf).is_err() {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        for (i, chunk) in buf.chunks_mut(8).enumerate() {
            let mut h = RandomState::new().build_hasher();
            h.write_u128(nanos);
            h.write_usize(i);
            chunk.copy_from_slice(&h.finish().to_le_bytes());
        }
    }
    let mut a = [0u8; 16];
    let mut b = [0u8; 16];
    a.copy_from_slice(&buf[..16]);
    b.copy_from_slice(&buf[16..]);
    (u128::from_le_bytes(a), u128::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_stream() {
        let mut a = Pcg64Dxsm::from_seed(42);
        let mut b = Pcg64Dxsm::from_seed(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_and_seed_zero_differ() {
        let x: Vec<u64> = (0..4).scan(Pcg64Dxsm::from_seed(1), |g, _| Some(g.next_u64())).collect();
        let y: Vec<u64> = (0..4).scan(Pcg64Dxsm::from_seed(2), |g, _| Some(g.next_u64())).collect();
        let z: Vec<u64> = (0..4).scan(Pcg64Dxsm::from_seed(0), |g, _| Some(g.next_u64())).collect();
        let w: Vec<u64> = (0..4).scan(Pcg64Dxsm::from_seed(0), |g, _| Some(g.next_u64())).collect();
        assert_ne!(x, y);
        assert_ne!(z, w);
    }

    /// Pins the stream: any change to the generator or the seeding moves these numbers,
    /// and with them every seeded result RiskPY publishes. The state and increment are
    /// this file's seeding of 12345; the three outputs were checked against NumPy's
    /// `PCG64DXSM` set to that state (1,000 outputs each for seeds 1, 12345 and
    /// 4294967295 agreed to the bit when 0.9.0 was written).
    #[test]
    fn seed_expansion_and_stream_are_pinned() {
        let mut g = Pcg64Dxsm::from_seed(12345);
        let (state, inc) = g.raw_state();
        assert_eq!(state, 0x670c_b3a8_bb6f_2eed_0d0d_517d_18d5_ed1b);
        assert_eq!(inc, 0xf2f1_70f9_a84a_2611_9c10_1f20_03ef_c4d3);
        assert_eq!(g.next_u64(), 6_022_038_901_905_743_034);
        assert_eq!(g.next_u64(), 4_405_933_682_800_833_314);
        assert_eq!(g.next_u64(), 17_650_291_280_362_227_660);
    }
}
