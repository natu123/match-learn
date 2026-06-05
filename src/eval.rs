//! Evaluation harness: regret and stability over a run.
//!
//! Runs a [`Market`] for a number of rounds and records two things per round:
//!
//! - **Regret** against the agent-optimal stable matching of the *true* market
//!   (the proposer-optimal stable matching under true preferences). For each
//!   proposer it is the gap between the utility of its stable partner and the
//!   utility of the partner it was actually matched to this round, summed over
//!   proposers. The Phase 1 gate asks for *sublinear* cumulative regret.
//!
//! - **Stability**: whether the realized matching is stable in the true market
//!   (no blocking pair under true preferences). The gate asks the matching to
//!   *stabilize* — to become and stay stable.

use crate::market::Market;
use crate::matching::{Matching, gale_shapley, is_stable};

/// A learning matching loop the harness can drive and score.
///
/// Implemented by both the one-sided [`Market`](crate::market::Market) and the
/// two-sided market, so they share the same regret/stability harness. The "true"
/// preferences are the ground truth the learners are converging toward; on a
/// one-sided market the receiver side is simply the known fixed preferences.
pub trait LearningMarket {
    /// Play one round and return the realized matching.
    fn step(&mut self) -> Matching;
    /// Number of proposers (the side whose regret is measured).
    fn n_proposers(&self) -> usize;
    /// Proposer `p`'s true utility for receiver `r`.
    fn proposer_util(&self, p: usize, r: usize) -> f64;
    /// Proposers' true preference rankings.
    fn true_proposer_prefs(&self) -> Vec<Vec<usize>>;
    /// Receivers' true preference rankings.
    fn true_receiver_prefs(&self) -> Vec<Vec<usize>>;
}

/// Per-round record of a run, plus convenience summaries.
#[derive(Debug, Clone)]
pub struct Report {
    /// Number of rounds played.
    pub rounds: usize,
    /// `cumulative_regret[t]` is total regret through round `t` (inclusive).
    pub cumulative_regret: Vec<f64>,
    /// `stable[t]` is whether round `t`'s matching was stable in the true market.
    pub stable: Vec<bool>,
}

impl Report {
    /// Total regret over the whole run.
    pub fn total_regret(&self) -> f64 {
        self.cumulative_regret.last().copied().unwrap_or(0.0)
    }

    /// Fraction of the final `k` rounds whose matching was stable.
    pub fn tail_stable_fraction(&self, k: usize) -> f64 {
        let k = k.min(self.rounds);
        if k == 0 {
            return 1.0;
        }
        let start = self.rounds - k;
        let n = self.stable[start..].iter().filter(|&&s| s).count();
        n as f64 / k as f64
    }

    /// The first round from which *every* later round is stable, i.e. the round
    /// at which the market settles for good. `None` if the last round is unstable.
    pub fn settled_round(&self) -> Option<usize> {
        if !self.stable.last().copied().unwrap_or(false) {
            return None;
        }
        // Walk back over the trailing run of stable rounds.
        let mut t = self.rounds;
        while t > 0 && self.stable[t - 1] {
            t -= 1;
        }
        Some(t)
    }

    /// Average per-round regret over the final `k` rounds. A market that has
    /// settled onto the stable matching has a tail average at (or below) zero.
    pub fn tail_mean_regret(&self, k: usize) -> f64 {
        let k = k.min(self.rounds);
        if k == 0 {
            return 0.0;
        }
        // Regret accumulated across the final k rounds, divided by k.
        let end = self.cumulative_regret[self.rounds - 1];
        let start = if self.rounds - k == 0 {
            0.0
        } else {
            self.cumulative_regret[self.rounds - k - 1]
        };
        (end - start) / k as f64
    }
}

/// Proposer-side regret of one realized `matching` against the `baseline`.
fn instantaneous_regret<M: LearningMarket>(
    market: &M,
    baseline: &Matching,
    matching: &Matching,
) -> f64 {
    let mut r = 0.0;
    for p in 0..market.n_proposers() {
        let base = baseline.proposer[p].map_or(0.0, |r| market.proposer_util(p, r));
        let got = matching.proposer[p].map_or(0.0, |r| market.proposer_util(p, r));
        r += base - got;
    }
    r
}

/// Run `market` for `rounds` rounds, returning a [`Report`].
///
/// Regret is measured against the proposer-optimal stable matching of the true
/// market; stability is checked against both sides' true preferences.
pub fn simulate<M: LearningMarket>(market: &mut M, rounds: usize) -> Report {
    let true_pp = market.true_proposer_prefs();
    let true_rp = market.true_receiver_prefs();
    let baseline = gale_shapley(&true_pp, &true_rp);

    let mut cumulative_regret = Vec::with_capacity(rounds);
    let mut stable = Vec::with_capacity(rounds);
    let mut acc = 0.0;

    for _ in 0..rounds {
        let matching = market.step();
        acc += instantaneous_regret(market, &baseline, &matching);
        cumulative_regret.push(acc);
        stable.push(is_stable(&true_pp, &true_rp, &matching));
    }

    Report {
        rounds,
        cumulative_regret,
        stable,
    }
}

impl LearningMarket for crate::market::Market {
    fn step(&mut self) -> Matching {
        Market::step(self)
    }
    fn n_proposers(&self) -> usize {
        Market::n_proposers(self)
    }
    fn proposer_util(&self, p: usize, r: usize) -> f64 {
        Market::true_util(self, p, r)
    }
    fn true_proposer_prefs(&self) -> Vec<Vec<usize>> {
        Market::true_proposer_prefs(self)
    }
    fn true_receiver_prefs(&self) -> Vec<Vec<usize>> {
        Market::receiver_prefs(self).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aligned_market(seed: u64) -> Market {
        let true_util = vec![
            vec![1.0, 0.4, 0.1],
            vec![0.2, 1.0, 0.5],
            vec![0.1, 0.3, 1.0],
        ];
        let receiver_prefs = vec![vec![0, 1, 2], vec![1, 0, 2], vec![2, 1, 0]];
        Market::with_thompson(true_util, receiver_prefs, 0.0, 1.0, 0.25, 0.3, seed)
    }

    #[test]
    fn report_lengths_match_rounds() {
        let mut m = aligned_market(1);
        let rep = simulate(&mut m, 500);
        assert_eq!(rep.cumulative_regret.len(), 500);
        assert_eq!(rep.stable.len(), 500);
        assert_eq!(rep.rounds, 500);
    }

    #[test]
    fn tail_is_mostly_stable() {
        // Thompson Sampling keeps a small residual exploration probability (arms
        // it stops pulling have frozen, non-zero posterior variance), so the
        // matching is stable in the vast majority of late rounds, not literally
        // all. The strict horizon-doubling sublinearity gate lives in tests/.
        let mut m = aligned_market(42);
        let rep = simulate(&mut m, 3000);
        assert!(
            rep.tail_stable_fraction(500) > 0.9,
            "tail stable fraction = {}",
            rep.tail_stable_fraction(500)
        );
    }

    #[test]
    fn tail_regret_rate_collapses_after_learning() {
        // Per-round regret late in the run is far below the early exploration
        // rate: learning has driven the regret rate down by a large factor.
        let mut m = aligned_market(7);
        let rep = simulate(&mut m, 3000);
        let early_rate = rep.cumulative_regret[199] / 200.0; // mean over first 200
        let tail_rate = rep.tail_mean_regret(500);
        // The late regret rate is well under half the early rate. (The strict
        // multi-seed, horizon-doubling sublinearity gate lives in tests/.)
        assert!(
            tail_rate < 0.5 * early_rate,
            "early_rate={early_rate}, tail_rate={tail_rate}"
        );
    }

    #[test]
    fn settled_round_is_none_when_last_round_unstable() {
        // A report whose final round is unstable has not settled.
        let rep = Report {
            rounds: 3,
            cumulative_regret: vec![1.0, 1.0, 2.0],
            stable: vec![true, true, false],
        };
        assert_eq!(rep.settled_round(), None);
        // ...and one with a stable tail reports the start of that run.
        let rep = Report {
            rounds: 4,
            cumulative_regret: vec![1.0, 2.0, 2.0, 2.0],
            stable: vec![false, false, true, true],
        };
        assert_eq!(rep.settled_round(), Some(2));
    }
}
