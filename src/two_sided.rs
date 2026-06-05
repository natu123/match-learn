//! Two-sided unknown preferences: both sides learn.
//!
//! In Phase 1 only proposers learned; receivers had known, fixed preferences.
//! Here *both* sides are bandits. Each round both sides rank from their beliefs,
//! Gale-Shapley matches the two belief-rankings, and every matched pair produces
//! a noisy reward for *each* partner — so proposers and receivers update
//! simultaneously. The learning target is the stable matching of the true market
//! under both sides' true preferences.
//!
//! Two-sided learning is harder: the ranking each side matches against is itself
//! moving, so early rounds are noisier than the one-sided case. It still
//! converges, as the tests show.

use crate::eval::LearningMarket;
use crate::learner::{GaussianThompson, PreferenceLearner};
use crate::matching::{Matching, gale_shapley};
use crate::prefs::rank_by_scores;
use crate::rng::Rng;

/// A two-sided market where both proposers and receivers learn their
/// preferences online.
pub struct TwoSidedMarket {
    /// `util_p[p][r]` is proposer `p`'s true mean utility for receiver `r`.
    util_p: Vec<Vec<f64>>,
    /// `util_r[r][p]` is receiver `r`'s true mean utility for proposer `p`.
    util_r: Vec<Vec<f64>>,
    proposer_learners: Vec<Box<dyn PreferenceLearner>>,
    receiver_learners: Vec<Box<dyn PreferenceLearner>>,
    noise: f64,
    rng: Rng,
    round: usize,
}

/// Derive strict preference rankings from a utility matrix (one row per agent).
fn prefs_from_util(util: &[Vec<f64>]) -> Vec<Vec<usize>> {
    util.iter().map(|row| rank_by_scores(row)).collect()
}

impl TwoSidedMarket {
    /// Build a two-sided market where every agent on both sides uses Gaussian
    /// Thompson Sampling.
    ///
    /// `util_p` is `[n_proposers][n_receivers]` and `util_r` is
    /// `[n_receivers][n_proposers]`.
    pub fn with_thompson(
        util_p: Vec<Vec<f64>>,
        util_r: Vec<Vec<f64>>,
        prior_mean: f64,
        prior_var: f64,
        obs_var: f64,
        noise: f64,
        seed: u64,
    ) -> Self {
        let n_p = util_p.len();
        let n_r = util_r.len();
        assert!(n_p > 0 && n_r > 0, "market must be non-empty");
        for row in &util_p {
            assert_eq!(row.len(), n_r, "util_p row must cover all receivers");
        }
        for row in &util_r {
            assert_eq!(row.len(), n_p, "util_r row must cover all proposers");
        }
        let proposer_learners = (0..n_p)
            .map(|p| {
                Box::new(GaussianThompson::new(
                    n_r,
                    prior_mean,
                    prior_var,
                    obs_var,
                    seed ^ (0x2000 + p as u64),
                )) as Box<dyn PreferenceLearner>
            })
            .collect();
        let receiver_learners = (0..n_r)
            .map(|r| {
                Box::new(GaussianThompson::new(
                    n_p,
                    prior_mean,
                    prior_var,
                    obs_var,
                    seed ^ (0x4000 + r as u64),
                )) as Box<dyn PreferenceLearner>
            })
            .collect();
        Self {
            util_p,
            util_r,
            proposer_learners,
            receiver_learners,
            noise,
            rng: Rng::new(seed),
            round: 0,
        }
    }

    /// Number of receivers.
    pub fn n_receivers(&self) -> usize {
        self.util_r.len()
    }

    /// Rounds played so far.
    pub fn round(&self) -> usize {
        self.round
    }

    /// Play one round and return the realized matching. Both sides rank from
    /// beliefs, Gale-Shapley matches them, and every matched pair updates both
    /// partners on a noisy reward.
    pub fn step(&mut self) -> Matching {
        let proposer_prefs: Vec<Vec<usize>> = self
            .proposer_learners
            .iter_mut()
            .map(|l| l.ranking())
            .collect();
        let receiver_prefs: Vec<Vec<usize>> = self
            .receiver_learners
            .iter_mut()
            .map(|l| l.ranking())
            .collect();
        let matching = gale_shapley(&proposer_prefs, &receiver_prefs);
        for (p, &slot) in matching.proposer.iter().enumerate() {
            if let Some(r) = slot {
                let rp = self.rng.normal(self.util_p[p][r], self.noise);
                self.proposer_learners[p].update(r, rp);
                let rr = self.rng.normal(self.util_r[r][p], self.noise);
                self.receiver_learners[r].update(p, rr);
            }
        }
        self.round += 1;
        matching
    }
}

impl LearningMarket for TwoSidedMarket {
    fn step(&mut self) -> Matching {
        TwoSidedMarket::step(self)
    }
    fn n_proposers(&self) -> usize {
        self.util_p.len()
    }
    fn proposer_util(&self, p: usize, r: usize) -> f64 {
        self.util_p[p][r]
    }
    fn true_proposer_prefs(&self) -> Vec<Vec<usize>> {
        prefs_from_util(&self.util_p)
    }
    fn true_receiver_prefs(&self) -> Vec<Vec<usize>> {
        prefs_from_util(&self.util_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::simulate;
    use crate::matching::gale_shapley;

    /// An aligned 3x3 market: proposer p and receiver p value each other most, so
    /// the unique stable matching is the identity.
    fn aligned(seed: u64) -> TwoSidedMarket {
        let util_p = vec![
            vec![1.0, 0.4, 0.1],
            vec![0.2, 1.0, 0.5],
            vec![0.1, 0.3, 1.0],
        ];
        let util_r = vec![
            vec![1.0, 0.3, 0.2],
            vec![0.4, 1.0, 0.3],
            vec![0.2, 0.5, 1.0],
        ];
        TwoSidedMarket::with_thompson(util_p, util_r, 0.5, 1.0, 0.04, 0.2, seed)
    }

    #[test]
    fn both_sides_converge_to_true_stable_matching() {
        let mut m = aligned(42);
        let target = gale_shapley(&m.true_proposer_prefs(), &m.true_receiver_prefs());
        assert_eq!(target.proposer, vec![Some(0), Some(1), Some(2)]);

        let rep = simulate(&mut m, 4000);
        assert!(
            rep.tail_stable_fraction(800) > 0.9,
            "tail stable fraction = {}",
            rep.tail_stable_fraction(800)
        );
        assert!(
            rep.tail_mean_regret(800).abs() < 0.02,
            "tail mean regret = {}",
            rep.tail_mean_regret(800)
        );
    }

    #[test]
    fn two_sided_learning_beats_no_learning_on_random_markets() {
        let mut seedgen = Rng::new(2026_0606);
        let markets = 20;
        let n = 4;
        let rounds = 2000;
        let noise = 0.15;

        let mut ratio_sum = 0.0;
        let mut learn_sum = 0.0;
        let mut base_sum = 0.0;

        for _ in 0..markets {
            let seed = (seedgen.below(1_000_000_000) as u64) + 1;
            let mut g = Rng::new(seed);
            let util_p: Vec<Vec<f64>> = (0..n)
                .map(|_| (0..n).map(|_| g.uniform()).collect())
                .collect();
            let util_r: Vec<Vec<f64>> = (0..n)
                .map(|_| (0..n).map(|_| g.uniform()).collect())
                .collect();

            let mut m = TwoSidedMarket::with_thompson(
                util_p.clone(),
                util_r.clone(),
                0.5,
                1.0,
                noise * noise,
                noise,
                seed ^ 0xBEEF,
            );
            let rep = simulate(&mut m, rounds);

            let r_t = rep.cumulative_regret[rounds / 2 - 1].max(1e-9);
            let r_2t = rep.cumulative_regret[rounds - 1].max(1e-9);
            ratio_sum += r_2t / r_t;
            learn_sum += rep.total_regret();

            // No-learning baseline: both sides rank by index forever.
            let true_pp = prefs_from_util(&util_p);
            let true_rp = prefs_from_util(&util_r);
            let baseline = gale_shapley(&true_pp, &true_rp);
            let fixed: Vec<Vec<usize>> = (0..n).map(|_| (0..n).collect()).collect();
            let played = gale_shapley(&fixed, &fixed);
            let mut per_round = 0.0;
            for (p, up) in util_p.iter().enumerate() {
                let b = baseline.proposer[p].map_or(0.0, |r| up[r]);
                let gp = played.proposer[p].map_or(0.0, |r| up[r]);
                per_round += b - gp;
            }
            base_sum += per_round * rounds as f64;
        }

        let mean_ratio = ratio_sum / markets as f64;
        // Sublinear on aggregate (two-sided is noisier than one-sided).
        assert!(mean_ratio < 1.5, "mean R(2T)/R(T) = {mean_ratio}");
        // And still clearly beats the no-learning baseline.
        assert!(
            learn_sum < 0.6 * base_sum,
            "learn {learn_sum} vs no-learn {base_sum}"
        );
    }
}
