//! Online preference learners (from scratch).
//!
//! Each agent in a market treats the other side as a multi-armed bandit: pulling
//! arm `a` (being matched to partner `a`) yields a noisy reward whose mean is the
//! agent's unknown utility for `a`. A [`PreferenceLearner`] turns the rewards
//! seen so far into a per-round ranking of the arms, which the matching layer
//! feeds to Gale-Shapley.
//!
//! Strategies provided, all for Gaussian rewards:
//! - [`GaussianThompson`] — Bayesian posterior sampling (Thompson Sampling),
//!   with an exploration-scale knob and exposed posterior mean / std / credible
//!   intervals.
//! - [`Ucb1`] — optimism in the face of uncertainty (UCB1).
//! - [`DiscountedThompson`] — Thompson Sampling that forgets, for non-stationary
//!   preferences.
//! - [`ForcedExploreThompson`] — Thompson Sampling with vanishing forced
//!   exploration (and optional annealing) that beats the greedy-Thompson
//!   matching stall.

use crate::rng::Rng;

/// An online estimator of an agent's preferences over a fixed set of arms.
///
/// `Send + Sync` so a `Market` of boxed learners can cross threads and satisfy
/// PyO3's thread-safety requirements; every learner here meets it trivially.
pub trait PreferenceLearner: Send + Sync {
    /// Number of arms (candidate partners).
    fn n_arms(&self) -> usize;

    /// Per-arm score for the current round; higher means more preferred.
    ///
    /// May be stochastic (Thompson Sampling draws a fresh posterior sample), so
    /// it takes `&mut self`.
    fn scores(&mut self) -> Vec<f64>;

    /// Record an observed `reward` for `arm`.
    fn update(&mut self, arm: usize, reward: f64);

    /// Deterministic point estimate of each arm's mean utility (diagnostics).
    fn means(&self) -> Vec<f64>;

    /// A full preference ranking for this round, most preferred first.
    ///
    /// Ties are broken by arm index, keeping the ranking deterministic given the
    /// scores.
    fn ranking(&mut self) -> Vec<usize> {
        crate::prefs::rank_by_scores(&self.scores())
    }
}

/// Thompson Sampling for Gaussian rewards with known observation noise.
///
/// Each arm's mean utility has a Gaussian prior `N(prior_mean, prior_var)`. With
/// a Gaussian likelihood of known variance `obs_var`, the posterior stays
/// Gaussian and is updated in closed form. Each round, [`scores`] draws one
/// sample per arm from its posterior.
///
/// [`scores`]: PreferenceLearner::scores
#[derive(Debug, Clone)]
pub struct GaussianThompson {
    obs_var: f64,
    prior_mean: f64,
    prior_var: f64,
    explore_scale: f64,
    count: Vec<f64>,
    sum: Vec<f64>,
    rng: Rng,
}

impl GaussianThompson {
    /// Create a learner over `n_arms` with the given prior and observation noise.
    ///
    /// `prior_var` and `obs_var` must be positive.
    pub fn new(n_arms: usize, prior_mean: f64, prior_var: f64, obs_var: f64, seed: u64) -> Self {
        assert!(
            prior_var > 0.0 && obs_var > 0.0,
            "variances must be positive"
        );
        Self {
            obs_var,
            prior_mean,
            prior_var,
            explore_scale: 1.0,
            count: vec![0.0; n_arms],
            sum: vec![0.0; n_arms],
            rng: Rng::new(seed),
        }
    }

    /// Set the exploration scale (a builder method). Posterior samples are drawn
    /// with their standard deviation multiplied by `scale`: `scale > 1` explores
    /// more, `scale < 1` is greedier, `scale == 0` is pure exploitation of the
    /// posterior mean. Default `1.0` is standard Thompson Sampling.
    pub fn with_exploration(mut self, scale: f64) -> Self {
        assert!(scale >= 0.0, "exploration scale must be non-negative");
        self.explore_scale = scale;
        self
    }

    /// Posterior (mean, variance) of arm `a`'s mean utility.
    fn posterior(&self, a: usize) -> (f64, f64) {
        let precision = 1.0 / self.prior_var + self.count[a] / self.obs_var;
        let var = 1.0 / precision;
        let mean = (self.prior_mean / self.prior_var + self.sum[a] / self.obs_var) * var;
        (mean, var)
    }

    /// Posterior mean of arm `a`'s utility (the Bayesian point estimate).
    pub fn posterior_mean(&self, a: usize) -> f64 {
        self.posterior(a).0
    }

    /// Posterior standard deviation of arm `a`'s utility (its uncertainty).
    pub fn posterior_std(&self, a: usize) -> f64 {
        self.posterior(a).1.sqrt()
    }

    /// A `z`-sigma credible interval `(low, high)` for arm `a`'s utility.
    pub fn credible_interval(&self, a: usize, z: f64) -> (f64, f64) {
        let (mean, var) = self.posterior(a);
        let half = z * var.sqrt();
        (mean - half, mean + half)
    }
}

impl PreferenceLearner for GaussianThompson {
    fn n_arms(&self) -> usize {
        self.count.len()
    }

    fn scores(&mut self) -> Vec<f64> {
        (0..self.count.len())
            .map(|a| {
                let (mean, var) = self.posterior(a);
                self.rng.normal(mean, self.explore_scale * var.sqrt())
            })
            .collect()
    }

    fn update(&mut self, arm: usize, reward: f64) {
        self.count[arm] += 1.0;
        self.sum[arm] += reward;
    }

    fn means(&self) -> Vec<f64> {
        (0..self.count.len()).map(|a| self.posterior(a).0).collect()
    }
}

/// UCB1 for bounded/Gaussian rewards.
///
/// The score of arm `a` is its empirical mean plus an exploration bonus
/// `c * sqrt(ln(total) / n_a)`. Arms never pulled score `+inf`, so each is tried
/// once before exploitation begins.
#[derive(Debug, Clone)]
pub struct Ucb1 {
    c: f64,
    total: f64,
    count: Vec<f64>,
    sum: Vec<f64>,
}

impl Ucb1 {
    /// Create a UCB1 learner over `n_arms` with exploration constant `c`.
    pub fn new(n_arms: usize, c: f64) -> Self {
        Self {
            c,
            total: 0.0,
            count: vec![0.0; n_arms],
            sum: vec![0.0; n_arms],
        }
    }

    fn mean(&self, a: usize) -> f64 {
        if self.count[a] == 0.0 {
            0.0
        } else {
            self.sum[a] / self.count[a]
        }
    }
}

impl PreferenceLearner for Ucb1 {
    fn n_arms(&self) -> usize {
        self.count.len()
    }

    fn scores(&mut self) -> Vec<f64> {
        let ln_total = (self.total.max(1.0)).ln();
        (0..self.count.len())
            .map(|a| {
                if self.count[a] == 0.0 {
                    f64::INFINITY
                } else {
                    self.mean(a) + self.c * (ln_total / self.count[a]).sqrt()
                }
            })
            .collect()
    }

    fn update(&mut self, arm: usize, reward: f64) {
        self.count[arm] += 1.0;
        self.sum[arm] += reward;
        self.total += 1.0;
    }

    fn means(&self) -> Vec<f64> {
        (0..self.count.len()).map(|a| self.mean(a)).collect()
    }
}

/// Thompson Sampling that forgets, for *non-stationary* preferences.
///
/// Identical to [`GaussianThompson`] except each arm's statistics are discounted
/// by `gamma` on every update of that arm: `count <- gamma*count + 1`,
/// `sum <- gamma*sum + reward`. Old evidence decays, so the effective sample
/// size saturates at `1/(1 - gamma)` and the posterior never fully hardens —
/// letting the learner track a moving target (e.g. an arm whose true value
/// drifts or whose ranking flips). `gamma == 1.0` recovers the stationary
/// learner.
#[derive(Debug, Clone)]
pub struct DiscountedThompson {
    obs_var: f64,
    prior_mean: f64,
    prior_var: f64,
    gamma: f64,
    count: Vec<f64>,
    sum: Vec<f64>,
    rng: Rng,
}

impl DiscountedThompson {
    /// Create a discounting learner over `n_arms`. `gamma` is the per-update
    /// discount in `(0, 1]`; smaller forgets faster.
    pub fn new(
        n_arms: usize,
        prior_mean: f64,
        prior_var: f64,
        obs_var: f64,
        gamma: f64,
        seed: u64,
    ) -> Self {
        assert!(
            prior_var > 0.0 && obs_var > 0.0,
            "variances must be positive"
        );
        assert!(gamma > 0.0 && gamma <= 1.0, "gamma must be in (0, 1]");
        Self {
            obs_var,
            prior_mean,
            prior_var,
            gamma,
            count: vec![0.0; n_arms],
            sum: vec![0.0; n_arms],
            rng: Rng::new(seed),
        }
    }

    fn posterior(&self, a: usize) -> (f64, f64) {
        let precision = 1.0 / self.prior_var + self.count[a] / self.obs_var;
        let var = 1.0 / precision;
        let mean = (self.prior_mean / self.prior_var + self.sum[a] / self.obs_var) * var;
        (mean, var)
    }
}

impl PreferenceLearner for DiscountedThompson {
    fn n_arms(&self) -> usize {
        self.count.len()
    }

    fn scores(&mut self) -> Vec<f64> {
        (0..self.count.len())
            .map(|a| {
                let (mean, var) = self.posterior(a);
                self.rng.normal(mean, var.sqrt())
            })
            .collect()
    }

    fn update(&mut self, arm: usize, reward: f64) {
        self.count[arm] = self.gamma * self.count[arm] + 1.0;
        self.sum[arm] = self.gamma * self.sum[arm] + reward;
    }

    fn means(&self) -> Vec<f64> {
        (0..self.count.len()).map(|a| self.posterior(a).0).collect()
    }
}

/// Thompson Sampling with vanishing forced exploration, to beat the matching
/// *stall* (the research-track cure for greedy Thompson's frozen-arm failure).
///
/// Greedy Thompson can lock a market onto a wrong stable matching: once an agent
/// stops being matched to an arm, that arm freezes and is never re-checked. This
/// learner adds two knobs:
/// - **forcing**: with probability `eps_t = min(1, c/t)` it forces a probe of
///   the least-sampled arm, so no arm stays frozen — `c == 0` is plain greedy
///   Thompson;
/// - **annealing** (optional, via [`with_anneal`]): the posterior-sample std is
///   scaled by `sqrt(tau / (tau + t))`, cooling toward exploitation to stop the
///   near-tie *churn* that keeps a matching from settling.
///
/// [`with_anneal`]: ForcedExploreThompson::with_anneal
#[derive(Debug, Clone)]
pub struct ForcedExploreThompson {
    obs_var: f64,
    prior_mean: f64,
    prior_var: f64,
    /// Forced-exploration constant: `eps_t = min(1, c / t)`.
    c: f64,
    /// Annealing timescale: Thompson sample std is scaled by
    /// `sqrt(tau / (tau + t))`. `inf` disables annealing (full Thompson forever).
    anneal_tau: f64,
    /// Rounds elapsed (the `t` in `eps_t`); incremented once per `scores` call.
    t: u64,
    count: Vec<f64>,
    sum: Vec<f64>,
    rng: Rng,
}

impl ForcedExploreThompson {
    /// Create a forced-exploration learner over `n_arms`.
    ///
    /// `prior_var` and `obs_var` must be positive; `c` (the forced-exploration
    /// constant) must be non-negative — `c == 0` recovers plain greedy Thompson
    /// Sampling. Larger `c` probes frozen arms more aggressively.
    pub fn new(
        n_arms: usize,
        prior_mean: f64,
        prior_var: f64,
        obs_var: f64,
        c: f64,
        seed: u64,
    ) -> Self {
        assert!(
            prior_var > 0.0 && obs_var > 0.0,
            "variances must be positive"
        );
        assert!(c >= 0.0, "forced-exploration constant must be non-negative");
        Self {
            obs_var,
            prior_mean,
            prior_var,
            c,
            anneal_tau: f64::INFINITY,
            t: 0,
            count: vec![0.0; n_arms],
            sum: vec![0.0; n_arms],
            rng: Rng::new(seed),
        }
    }

    /// Anneal the Thompson sampling temperature (a builder method).
    ///
    /// The posterior-sample standard deviation is multiplied by
    /// `sqrt(tau / (tau + t))`, decaying from full Thompson exploration toward
    /// pure posterior-mean exploitation. Smaller `tau` cools faster. This
    /// suppresses the near-tie churn that keeps a matching from settling. `tau`
    /// must be positive; the default (no call) leaves annealing off.
    pub fn with_anneal(mut self, tau: f64) -> Self {
        assert!(tau > 0.0, "annealing timescale must be positive");
        self.anneal_tau = tau;
        self
    }

    /// The annealing factor applied to the Thompson sample std this round.
    fn anneal_factor(&self) -> f64 {
        if self.anneal_tau.is_infinite() {
            1.0
        } else {
            (self.anneal_tau / (self.anneal_tau + self.t as f64)).sqrt()
        }
    }

    /// Posterior (mean, variance) of arm `a`'s mean utility.
    fn posterior(&self, a: usize) -> (f64, f64) {
        let precision = 1.0 / self.prior_var + self.count[a] / self.obs_var;
        let var = 1.0 / precision;
        let mean = (self.prior_mean / self.prior_var + self.sum[a] / self.obs_var) * var;
        (mean, var)
    }

    /// Index of the least-sampled arm (ties broken by lowest index): the frozen
    /// arm a forced round targets.
    fn least_sampled(&self) -> usize {
        let mut best = 0;
        for a in 1..self.count.len() {
            if self.count[a] < self.count[best] {
                best = a;
            }
        }
        best
    }
}

impl PreferenceLearner for ForcedExploreThompson {
    fn n_arms(&self) -> usize {
        self.count.len()
    }

    fn scores(&mut self) -> Vec<f64> {
        self.t += 1;
        let eps = (self.c / self.t as f64).min(1.0);
        if self.c > 0.0 && self.rng.uniform() < eps {
            // Forced round: the least-sampled arm dominates the ranking; the rest
            // fall back to their posterior means, so if a receiver rejects the
            // forced proposal the agent still proposes sensibly down its list.
            let forced = self.least_sampled();
            (0..self.count.len())
                .map(|a| {
                    if a == forced {
                        f64::INFINITY
                    } else {
                        self.posterior(a).0
                    }
                })
                .collect()
        } else {
            // Ordinary Thompson round: one posterior sample per arm, with the
            // sampling std optionally annealed toward zero to stop near-tie churn.
            let a_factor = self.anneal_factor();
            (0..self.count.len())
                .map(|a| {
                    let (mean, var) = self.posterior(a);
                    self.rng.normal(mean, a_factor * var.sqrt())
                })
                .collect()
        }
    }

    fn update(&mut self, arm: usize, reward: f64) {
        self.count[arm] += 1.0;
        self.sum[arm] += reward;
    }

    fn means(&self) -> Vec<f64> {
        (0..self.count.len()).map(|a| self.posterior(a).0).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a learner as a plain bandit: each round pull the top-ranked arm,
    /// observe a noisy reward, update. Return the cumulative pseudo-regret.
    fn run_bandit<L: PreferenceLearner>(
        learner: &mut L,
        true_means: &[f64],
        rounds: usize,
        noise: f64,
        seed: u64,
    ) -> f64 {
        let mut env = Rng::new(seed);
        let best = true_means.iter().cloned().fold(f64::MIN, f64::max);
        let mut regret = 0.0;
        for _ in 0..rounds {
            let arm = learner.ranking()[0];
            let reward = env.normal(true_means[arm], noise);
            learner.update(arm, reward);
            regret += best - true_means[arm];
        }
        regret
    }

    #[test]
    fn thompson_identifies_best_arm() {
        let true_means = [0.1, 0.5, 0.9, 0.3];
        let mut ts = GaussianThompson::new(4, 0.0, 1.0, 0.25, 1);
        run_bandit(&mut ts, &true_means, 2000, 0.5, 99);
        let means = ts.means();
        let best = (0..4).max_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap());
        assert_eq!(best, Some(2));
    }

    #[test]
    fn ucb1_identifies_best_arm() {
        let true_means = [0.1, 0.5, 0.9, 0.3];
        let mut ucb = Ucb1::new(4, 1.0);
        run_bandit(&mut ucb, &true_means, 2000, 0.5, 99);
        let means = ucb.means();
        let best = (0..4).max_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap());
        assert_eq!(best, Some(2));
    }

    #[test]
    fn thompson_regret_is_sublinear() {
        // Regret over 2T rounds should be well below 2x the regret over T rounds
        // if growth is sublinear; we check it grows much slower than linearly.
        let true_means = [0.0, 0.4, 0.8];
        let regret_t = {
            let mut ts = GaussianThompson::new(3, 0.0, 1.0, 0.25, 5);
            run_bandit(&mut ts, &true_means, 1000, 0.5, 7)
        };
        let regret_2t = {
            let mut ts = GaussianThompson::new(3, 0.0, 1.0, 0.25, 5);
            run_bandit(&mut ts, &true_means, 2000, 0.5, 7)
        };
        // Linear growth would give regret_2t ~= 2 * regret_t. Sublinear is well under.
        assert!(
            regret_2t < 1.6 * regret_t,
            "regret_t={regret_t}, regret_2t={regret_2t} (not clearly sublinear)"
        );
    }

    #[test]
    fn ranking_is_a_permutation_and_deterministic_given_scores() {
        let mut ucb = Ucb1::new(5, 1.0);
        for a in 0..5 {
            ucb.update(a, a as f64);
        }
        let r = ucb.ranking();
        let mut sorted = r.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
        // Highest empirical mean (arm 4) ranks first.
        assert_eq!(r[0], 4);
    }

    /// Run a learner on a bandit whose arm means switch halfway, returning the
    /// regret accumulated in the *second* half (after the switch).
    fn second_half_regret<L: PreferenceLearner>(
        learner: &mut L,
        means_a: &[f64],
        means_b: &[f64],
        half: usize,
        noise: f64,
        seed: u64,
    ) -> f64 {
        let mut env = Rng::new(seed);
        let mut regret = 0.0;
        for t in 0..2 * half {
            let means = if t < half { means_a } else { means_b };
            let best = means.iter().cloned().fold(f64::MIN, f64::max);
            let arm = learner.ranking()[0];
            let reward = env.normal(means[arm], noise);
            learner.update(arm, reward);
            if t >= half {
                regret += best - means[arm];
            }
        }
        regret
    }

    #[test]
    fn discounting_tracks_a_switch_better_than_stationary() {
        // Arm 0 is best in the first half; arms swap in the second half.
        let means_a = [1.0, 0.0];
        let means_b = [0.0, 1.0];
        let half = 1500;
        let noise = 0.2;

        let mut discounted = DiscountedThompson::new(2, 0.5, 1.0, 0.04, 0.95, 1);
        let dr = second_half_regret(&mut discounted, &means_a, &means_b, half, noise, 7);

        let mut stationary = GaussianThompson::new(2, 0.5, 1.0, 0.04, 1);
        let sr = second_half_regret(&mut stationary, &means_a, &means_b, half, noise, 7);

        // The forgetting learner recovers after the switch; the stationary one,
        // having hardened onto arm 0, pays far more regret in the second half.
        assert!(
            dr < 0.5 * sr,
            "discounted second-half regret {dr} not << stationary {sr}"
        );
    }

    #[test]
    fn discounting_with_gamma_one_matches_stationary() {
        // gamma = 1.0 means no forgetting: same best-arm identification.
        let true_means = [0.1, 0.5, 0.9, 0.3];
        let mut d = DiscountedThompson::new(4, 0.0, 1.0, 0.25, 1.0, 1);
        run_bandit(&mut d, &true_means, 2000, 0.5, 99);
        let means = d.means();
        let best = (0..4).max_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap());
        assert_eq!(best, Some(2));
    }

    #[test]
    fn posterior_uncertainty_shrinks_with_evidence() {
        let mut ts = GaussianThompson::new(2, 0.0, 1.0, 0.25, 1);
        let std_prior = ts.posterior_std(0);
        for _ in 0..100 {
            ts.update(0, 0.7);
        }
        let std_after = ts.posterior_std(0);
        assert!(
            std_after < std_prior,
            "posterior std did not shrink: {std_prior} -> {std_after}"
        );
        // The mean is pulled toward the observed reward, and a 2-sigma interval
        // brackets it.
        let (lo, hi) = ts.credible_interval(0, 2.0);
        let mean = ts.posterior_mean(0);
        assert!(lo < mean && mean < hi);
        assert!(mean > 0.5, "mean did not move toward observed 0.7: {mean}");
    }

    #[test]
    fn exploration_scale_controls_how_much_it_explores() {
        // On the same bandit, a larger exploration scale pulls the suboptimal
        // arms more often than a greedier (small-scale) learner.
        let true_means = [0.9, 0.5, 0.3, 0.1];
        let suboptimal_pulls = |scale: f64| {
            let mut ts = GaussianThompson::new(4, 0.0, 1.0, 0.25, 5).with_exploration(scale);
            let mut env = Rng::new(7);
            let mut count = 0;
            for _ in 0..600 {
                let arm = ts.ranking()[0];
                if arm != 0 {
                    count += 1;
                }
                ts.update(arm, env.normal(true_means[arm], 0.5));
            }
            count
        };
        let greedy = suboptimal_pulls(0.2);
        let exploratory = suboptimal_pulls(2.5);
        assert!(
            exploratory > greedy,
            "more exploration should mean more suboptimal pulls: greedy={greedy}, exploratory={exploratory}"
        );
    }

    #[test]
    fn forced_exploration_probes_every_arm() {
        // Even an arm whose mean is far worse keeps getting probed under forcing,
        // so it never freezes — the property that escapes the matching stall.
        let true_means = [1.0, 0.2, 0.0, 0.0];
        let mut fe = ForcedExploreThompson::new(4, 0.0, 1.0, 0.25, 1.0, 3);
        let mut env = Rng::new(9);
        for _ in 0..3000 {
            let arm = fe.ranking()[0];
            fe.update(arm, env.normal(true_means[arm], 0.5));
        }
        // Every arm has been pulled at least once.
        for a in 0..4 {
            assert!(fe.count[a] > 0.0, "arm {a} never probed under forcing");
        }
        // Still identifies the best arm.
        let means = fe.means();
        let best = (0..4).max_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap());
        assert_eq!(best, Some(0));
    }

    #[test]
    fn forced_explore_with_c_zero_is_greedy_thompson() {
        // c = 0 disables forcing: behaves like plain Thompson, identifying the
        // best arm without the forced probes.
        let true_means = [0.1, 0.5, 0.9, 0.3];
        let mut fe = ForcedExploreThompson::new(4, 0.0, 1.0, 0.25, 0.0, 1);
        let mut env = Rng::new(99);
        for _ in 0..2000 {
            let arm = fe.ranking()[0];
            fe.update(arm, env.normal(true_means[arm], 0.5));
        }
        let means = fe.means();
        let best = (0..4).max_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap());
        assert_eq!(best, Some(2));
    }
}
