//! Linear contextual bandit (Thompson Sampling).
//!
//! Where [`GaussianThompson`](crate::learner::GaussianThompson) learns one mean
//! per arm, a contextual learner generalizes across arms through *features*. At
//! each round an arm is described by a feature vector `x`, and its expected
//! reward is `theta . x` for an unknown weight vector `theta` shared across arms.
//! This lets preferences transfer: a never-seen arm is scored from its features.
//!
//! We use Bayesian linear regression with a Gaussian prior `theta ~ N(0, I/lambda)`
//! and known observation noise `obs_var`. The posterior precision and
//! information vector are
//!
//! ```text
//! A = lambda*I + sum_t x_t x_t^T / obs_var,   b = sum_t r_t x_t / obs_var,
//! ```
//!
//! giving posterior mean `mu = A^{-1} b` and covariance `A^{-1}`. Thompson
//! Sampling draws `theta~ ~ N(mu, A^{-1})` each round and scores arms by
//! `theta~ . x`.

use crate::linalg::{cholesky, dot, inverse, matvec};
use crate::prefs::rank_by_scores;
use crate::rng::Rng;

/// A linear contextual bandit using Thompson Sampling.
#[derive(Debug, Clone)]
pub struct LinearThompson {
    dim: usize,
    obs_var: f64,
    /// Posterior precision `A` (symmetric positive-definite).
    a: Vec<Vec<f64>>,
    /// Information vector `b`.
    b: Vec<f64>,
    rng: Rng,
}

impl LinearThompson {
    /// Create a learner over `dim`-dimensional features. `lambda` is the prior
    /// precision (ridge strength, `> 0`); `obs_var` is the reward noise variance
    /// (`> 0`).
    pub fn new(dim: usize, lambda: f64, obs_var: f64, seed: u64) -> Self {
        assert!(dim > 0, "dimension must be positive");
        assert!(
            lambda > 0.0 && obs_var > 0.0,
            "lambda and obs_var must be positive"
        );
        let a = (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| if i == j { lambda } else { 0.0 })
                    .collect()
            })
            .collect();
        Self {
            dim,
            obs_var,
            a,
            b: vec![0.0; dim],
            rng: Rng::new(seed),
        }
    }

    /// Feature dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Posterior mean weight vector `mu = A^{-1} b`.
    pub fn theta_mean(&self) -> Vec<f64> {
        matvec(&inverse(&self.a), &self.b)
    }

    /// Incorporate one observation: features `x` produced reward `r`.
    pub fn update(&mut self, x: &[f64], r: f64) {
        assert_eq!(x.len(), self.dim, "feature dimension mismatch");
        let inv = 1.0 / self.obs_var;
        for i in 0..self.dim {
            self.b[i] += r * x[i] * inv;
            for j in 0..self.dim {
                self.a[i][j] += x[i] * x[j] * inv;
            }
        }
    }

    /// Draw a posterior sample `theta~ ~ N(mu, A^{-1})`.
    fn sample_theta(&mut self) -> Vec<f64> {
        let cov = inverse(&self.a);
        let mu = matvec(&cov, &self.b);
        let l = cholesky(&cov); // l l^T = cov
        let z: Vec<f64> = (0..self.dim).map(|_| self.rng.gaussian()).collect();
        // theta = mu + l * z
        (0..self.dim).map(|i| mu[i] + dot(&l[i], &z)).collect()
    }

    /// Score each arm by `theta~ . x` for a freshly sampled `theta~`.
    ///
    /// `contexts[a]` is arm `a`'s feature vector this round; higher score means
    /// more preferred.
    pub fn scores(&mut self, contexts: &[Vec<f64>]) -> Vec<f64> {
        let theta = self.sample_theta();
        contexts.iter().map(|x| dot(&theta, x)).collect()
    }

    /// Preference ranking over the arms described by `contexts`, most preferred
    /// first.
    pub fn ranking(&mut self, contexts: &[Vec<f64>]) -> Vec<usize> {
        rank_by_scores(&self.scores(contexts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random feature vector in `[-1, 1]^d`.
    fn random_context(rng: &mut Rng, d: usize) -> Vec<f64> {
        (0..d).map(|_| 2.0 * rng.uniform() - 1.0).collect()
    }

    #[test]
    fn recovers_the_true_weight_vector() {
        let theta_true = [1.5, -2.0, 0.5];
        let d = 3;
        let mut learner = LinearThompson::new(d, 1.0, 0.04, 1);
        let mut env = Rng::new(7);
        for _ in 0..3000 {
            let x = random_context(&mut env, d);
            let r = dot(&theta_true, &x) + env.normal(0.0, 0.2);
            learner.update(&x, r);
        }
        let est = learner.theta_mean();
        for k in 0..d {
            assert!(
                (est[k] - theta_true[k]).abs() < 0.1,
                "theta[{k}] = {} vs true {}",
                est[k],
                theta_true[k]
            );
        }
    }

    #[test]
    fn contextual_choice_beats_random_and_regret_flattens() {
        // Each round presents K arms with fresh random features; reward is
        // theta.x. The learner should pick near-best arms, so its tail regret
        // rate is far below picking a fixed arm.
        let theta_true = [1.0, -1.0, 0.5, 2.0];
        let d = 4;
        let k = 6;
        let rounds = 3000;

        let mut learner = LinearThompson::new(d, 1.0, 0.04, 3);
        let mut env = Rng::new(11);
        let mut cumulative = Vec::with_capacity(rounds);
        let mut acc = 0.0;
        for _ in 0..rounds {
            let contexts: Vec<Vec<f64>> = (0..k).map(|_| random_context(&mut env, d)).collect();
            let values: Vec<f64> = contexts.iter().map(|x| dot(&theta_true, x)).collect();
            let best = values.iter().cloned().fold(f64::MIN, f64::max);

            let chosen = learner.ranking(&contexts)[0];
            let reward = values[chosen] + env.normal(0.0, 0.2);
            learner.update(&contexts[chosen], reward);

            acc += best - values[chosen];
            cumulative.push(acc);
        }

        // Tail regret rate is much smaller than the early rate: learning works.
        let early_rate = cumulative[299] / 300.0;
        let tail_rate = (cumulative[rounds - 1] - cumulative[rounds - 301]) / 300.0;
        assert!(
            tail_rate < 0.4 * early_rate,
            "early_rate={early_rate}, tail_rate={tail_rate}"
        );
    }
}
