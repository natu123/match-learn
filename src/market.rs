//! The integration loop: learn -> match -> reward -> update.
//!
//! A [`Market`] couples self-built preference learners with Gale-Shapley. This
//! is the Phase 1 setting: **one-sided unknown preferences**. Proposers (agents)
//! do not know their own utilities and learn them online; receivers (arms) have
//! fixed, *known* preferences over proposers.
//!
//! Each round:
//! 1. every proposer ranks the receivers from its current beliefs;
//! 2. Gale-Shapley computes a stable matching of beliefs against known receiver
//!    preferences;
//! 3. each matched proposer pulls its partner, observing a noisy reward;
//! 4. learners update on what they saw.
//!
//! Over time the belief-rankings approach the true preferences, so the played
//! matching approaches the stable matching of the *true* market.

use crate::learner::{GaussianThompson, PreferenceLearner, Ucb1};
use crate::matching::{gale_shapley, Matching};
use crate::rng::Rng;

/// A two-sided market with learning proposers and known-preference receivers.
pub struct Market {
    /// `true_util[p][r]` is proposer `p`'s true mean utility for receiver `r`.
    true_util: Vec<Vec<f64>>,
    /// Known receiver preference rankings over proposers (most preferred first).
    receiver_prefs: Vec<Vec<usize>>,
    /// One online learner per proposer.
    learners: Vec<Box<dyn PreferenceLearner>>,
    /// Standard deviation of the reward noise.
    noise: f64,
    rng: Rng,
    round: usize,
}

impl Market {
    /// Build a market from explicit learners.
    ///
    /// `true_util` is `[n_proposers][n_receivers]`; `receiver_prefs` has one
    /// ranking per receiver; `learners` has one per proposer, each over
    /// `n_receivers` arms.
    pub fn new(
        true_util: Vec<Vec<f64>>,
        receiver_prefs: Vec<Vec<usize>>,
        learners: Vec<Box<dyn PreferenceLearner>>,
        noise: f64,
        seed: u64,
    ) -> Self {
        let n_r = receiver_prefs.len();
        assert_eq!(true_util.len(), learners.len(), "one learner per proposer");
        for row in &true_util {
            assert_eq!(row.len(), n_r, "true_util row must cover all receivers");
        }
        for l in &learners {
            assert_eq!(l.n_arms(), n_r, "learner must have one arm per receiver");
        }
        Self {
            true_util,
            receiver_prefs,
            learners,
            noise,
            rng: Rng::new(seed),
            round: 0,
        }
    }

    /// Build a market whose proposers all use Gaussian Thompson Sampling.
    pub fn with_thompson(
        true_util: Vec<Vec<f64>>,
        receiver_prefs: Vec<Vec<usize>>,
        prior_mean: f64,
        prior_var: f64,
        obs_var: f64,
        noise: f64,
        seed: u64,
    ) -> Self {
        let n_r = receiver_prefs.len();
        let learners: Vec<Box<dyn PreferenceLearner>> = (0..true_util.len())
            .map(|p| {
                Box::new(GaussianThompson::new(
                    n_r,
                    prior_mean,
                    prior_var,
                    obs_var,
                    seed ^ (0x1000 + p as u64),
                )) as Box<dyn PreferenceLearner>
            })
            .collect();
        Self::new(true_util, receiver_prefs, learners, noise, seed)
    }

    /// Build a market whose proposers all use UCB1.
    pub fn with_ucb(
        true_util: Vec<Vec<f64>>,
        receiver_prefs: Vec<Vec<usize>>,
        c: f64,
        noise: f64,
        seed: u64,
    ) -> Self {
        let n_r = receiver_prefs.len();
        let learners: Vec<Box<dyn PreferenceLearner>> = (0..true_util.len())
            .map(|_| Box::new(Ucb1::new(n_r, c)) as Box<dyn PreferenceLearner>)
            .collect();
        Self::new(true_util, receiver_prefs, learners, noise, seed)
    }

    /// Number of proposers (learning agents).
    pub fn n_proposers(&self) -> usize {
        self.true_util.len()
    }

    /// Number of receivers (arms).
    pub fn n_receivers(&self) -> usize {
        self.receiver_prefs.len()
    }

    /// Rounds played so far.
    pub fn round(&self) -> usize {
        self.round
    }

    /// True utility of matching proposer `p` with receiver `r`.
    pub fn true_util(&self, p: usize, r: usize) -> f64 {
        self.true_util[p][r]
    }

    /// Receiver preferences (known, fixed).
    pub fn receiver_prefs(&self) -> &[Vec<usize>] {
        &self.receiver_prefs
    }

    /// Each proposer's *true* preference ranking, derived from `true_util`
    /// (descending, ties broken by receiver index). Used by evaluation to find
    /// the stable matching of the true market.
    pub fn true_proposer_prefs(&self) -> Vec<Vec<usize>> {
        self.true_util
            .iter()
            .map(|row| {
                let mut idx: Vec<usize> = (0..row.len()).collect();
                idx.sort_by(|&a, &b| {
                    row[b]
                        .partial_cmp(&row[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });
                idx
            })
            .collect()
    }

    /// The proposer-optimal stable matching of the *true* market. This is the
    /// learning target and the baseline for regret.
    pub fn true_stable_matching(&self) -> Matching {
        gale_shapley(&self.true_proposer_prefs(), &self.receiver_prefs)
    }

    /// Each proposer's belief-ranking of receivers for the current round.
    pub fn belief_rankings(&mut self) -> Vec<Vec<usize>> {
        self.learners.iter_mut().map(|l| l.ranking()).collect()
    }

    /// Play one round and return the matching that was realized.
    ///
    /// Proposers rank receivers from their beliefs, Gale-Shapley matches against
    /// known receiver preferences, each matched proposer observes a noisy reward,
    /// and learners update.
    pub fn step(&mut self) -> Matching {
        let proposer_prefs = self.belief_rankings();
        let matching = gale_shapley(&proposer_prefs, &self.receiver_prefs);
        for p in 0..self.n_proposers() {
            if let Some(r) = matching.proposer[p] {
                let reward = self.rng.normal(self.true_util[p][r], self.noise);
                self.learners[p].update(r, reward);
            }
        }
        self.round += 1;
        matching
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::is_stable;

    /// A small market where the stable matching of the true market is unique and
    /// known, so we can check that learning converges onto it.
    fn aligned_market_thompson(seed: u64) -> Market {
        // 3 proposers, 3 receivers. Proposer p most values receiver p.
        let true_util = vec![
            vec![1.0, 0.4, 0.1],
            vec![0.2, 1.0, 0.5],
            vec![0.1, 0.3, 1.0],
        ];
        // Receivers reciprocate: receiver r most values proposer r.
        let receiver_prefs = vec![vec![0, 1, 2], vec![1, 0, 2], vec![2, 1, 0]];
        Market::with_thompson(true_util, receiver_prefs, 0.0, 1.0, 0.25, 0.3, seed)
    }

    #[test]
    fn step_produces_a_stable_matching_of_current_beliefs() {
        // UCB rankings are a deterministic function of learner state, so the
        // prefs we read equal the prefs step() uses internally. (Thompson
        // Sampling re-draws each call, so its realized matching is stable w.r.t.
        // a sample we cannot re-observe — hence we check the invariant on UCB.)
        let true_util = vec![
            vec![1.0, 0.4, 0.1],
            vec![0.2, 1.0, 0.5],
            vec![0.1, 0.3, 1.0],
        ];
        let receiver_prefs = vec![vec![0, 1, 2], vec![1, 0, 2], vec![2, 1, 0]];
        let mut m = Market::with_ucb(true_util, receiver_prefs, 0.5, 0.3, 1);
        for _ in 0..10 {
            let prefs = m.belief_rankings();
            let played = m.step();
            assert!(is_stable(&prefs, m.receiver_prefs(), &played));
        }
        assert_eq!(m.round(), 10);
    }

    #[test]
    fn converges_to_true_stable_matching_thompson() {
        let mut m = aligned_market_thompson(42);
        let target = m.true_stable_matching();
        // Here the aligned market's stable matching is the identity.
        assert_eq!(target.proposer, vec![Some(0), Some(1), Some(2)]);

        let mut last = m.step();
        for _ in 0..3000 {
            last = m.step();
        }
        // After enough learning, the played matching is the true stable one.
        assert_eq!(last.proposer, target.proposer);
    }

    #[test]
    fn converges_to_true_stable_matching_ucb() {
        let true_util = vec![
            vec![1.0, 0.4, 0.1],
            vec![0.2, 1.0, 0.5],
            vec![0.1, 0.3, 1.0],
        ];
        let receiver_prefs = vec![vec![0, 1, 2], vec![1, 0, 2], vec![2, 1, 0]];
        let mut m = Market::with_ucb(true_util, receiver_prefs, 0.5, 0.3, 7);
        let target = m.true_stable_matching();
        let mut last = m.step();
        for _ in 0..5000 {
            last = m.step();
        }
        assert_eq!(last.proposer, target.proposer);
    }
}
