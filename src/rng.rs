//! A small, dependency-free deterministic RNG.
//!
//! `match-learn`'s v0 core is built from scratch, including randomness, so that
//! results are reproducible from a single seed without pulling in `rand`.
//! The generator is [splitmix64], which is fast and has good statistical
//! quality for simulation; Gaussian samples use the Box-Muller transform.
//!
//! [splitmix64]: https://prng.di.unimi.it/splitmix64.c

/// A reproducible pseudo-random generator seeded by a single `u64`.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
    /// The spare Box-Muller normal sample, kept until the next `gaussian` call.
    spare_gaussian: Option<f64>,
}

impl Rng {
    /// Create a generator from `seed`. Equal seeds yield identical streams.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
            spare_gaussian: None,
        }
    }

    /// Next raw 64-bit value (splitmix64).
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        // Use the top 53 bits for full f64 mantissa precision.
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Standard normal sample (mean 0, variance 1) via Box-Muller.
    ///
    /// Box-Muller yields *two* independent normals per pair of uniforms; we
    /// return one and cache the other, so amortized cost is one `ln`/`sqrt`/
    /// `sin_cos` per two samples instead of per sample.
    pub fn gaussian(&mut self) -> f64 {
        if let Some(g) = self.spare_gaussian.take() {
            return g;
        }
        let u1 = self.uniform().max(1e-12); // avoid ln(0)
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let (sin, cos) = (std::f64::consts::TAU * u2).sin_cos();
        self.spare_gaussian = Some(r * sin);
        r * cos
    }

    /// Normal sample with the given `mean` and standard deviation `std`.
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        mean + std * self.gaussian()
    }

    /// Uniform integer in `[0, n)`. Panics if `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0) is undefined");
        (self.next_u64() % n as u64) as usize
    }

    /// A Poisson sample with mean `lambda >= 0`, via Knuth's algorithm.
    ///
    /// Fine for the small-to-moderate rates used in the marketplace model; cost
    /// grows linearly with `lambda`.
    pub fn poisson(&mut self, lambda: f64) -> usize {
        assert!(lambda >= 0.0, "lambda must be non-negative");
        if lambda == 0.0 {
            return 0;
        }
        let l = (-lambda).exp();
        let mut k = 0usize;
        let mut p = 1.0;
        loop {
            p *= self.uniform();
            if p <= l {
                return k;
            }
            k += 1;
        }
    }

    /// In-place Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            slice.swap(i, j);
        }
    }

    /// A uniformly random permutation of `0..n`.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        self.shuffle(&mut v);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn uniform_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn gaussian_mean_and_variance_roughly_correct() {
        let mut r = Rng::new(1234);
        let n = 200_000;
        let mut sum = 0.0;
        let mut sq = 0.0;
        for _ in 0..n {
            let x = r.gaussian();
            sum += x;
            sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.05, "var = {var}");
    }

    #[test]
    fn poisson_mean_and_variance_roughly_correct() {
        // For Poisson, both mean and variance equal lambda.
        let mut r = Rng::new(2024);
        let lambda = 4.0;
        let n = 200_000;
        let mut sum = 0.0;
        let mut sq = 0.0;
        for _ in 0..n {
            let x = r.poisson(lambda) as f64;
            sum += x;
            sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!((mean - lambda).abs() < 0.05, "mean = {mean}");
        assert!((var - lambda).abs() < 0.1, "var = {var}");
        assert_eq!(r.poisson(0.0), 0);
    }

    #[test]
    fn permutation_is_a_permutation() {
        let mut r = Rng::new(99);
        let p = r.permutation(50);
        let mut seen = p.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..50).collect::<Vec<_>>());
    }
}
