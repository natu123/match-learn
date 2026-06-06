//! Online (dynamic) matching: agents arrive and depart over time.
//!
//! The [`Market`](crate::market::Market) repeats a *static* market; real
//! platforms are dynamic — riders, drivers, patients, applicants arrive over
//! time and leave when matched (or give up). The central question is no longer
//! *whom* to match but *when*: match an agent the moment it arrives, against a
//! thin pool and a likely-poor partner, or wait for the pool to thicken and a
//! better partner to appear — at the risk that someone abandons first.
//!
//! [`OnlineMarket`] models this on the plane (match value is proximity, as in
//! ride-hailing) and lets a [`Policy`] choose the timing. The greedy policy
//! matches every tick; a batched policy accumulates the pool and matches
//! periodically, trading shorter waits for better matches.

use crate::matching::gale_shapley;
use crate::prefs::rank_by_scores;
use crate::rng::Rng;

/// When the platform runs a matching round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Match every tick, against whoever is waiting.
    Greedy,
    /// Match once every `k` ticks, letting the pool accumulate first.
    Batched(usize),
}

/// Summary statistics of an online run.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OnlineStats {
    /// Pairs matched over the run.
    pub matched: usize,
    /// Agents that abandoned their queue unmatched (summed over both sides).
    pub abandoned: usize,
    /// Sum of match distances (lower is better quality).
    pub total_distance: f64,
}

impl OnlineStats {
    /// Mean distance per matched pair (match quality; lower is better). `0.0` if
    /// nothing matched.
    pub fn mean_distance(&self) -> f64 {
        if self.matched == 0 {
            0.0
        } else {
            self.total_distance / self.matched as f64
        }
    }
}

/// A dynamic two-sided market on the unit plane.
#[derive(Debug, Clone)]
pub struct OnlineMarket {
    arrivals: f64,
    abandon: f64,
    waiting_p: Vec<(f64, f64)>,
    waiting_r: Vec<(f64, f64)>,
    rng: Rng,
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

impl OnlineMarket {
    /// Create a market with `arrivals` mean Poisson arrivals per side per tick
    /// and per-tick abandonment probability `abandon`.
    pub fn new(arrivals: f64, abandon: f64, seed: u64) -> Self {
        assert!(arrivals >= 0.0, "arrivals must be non-negative");
        assert!((0.0..1.0).contains(&abandon), "abandon must be in [0, 1)");
        Self {
            arrivals,
            abandon,
            waiting_p: Vec::new(),
            waiting_r: Vec::new(),
            rng: Rng::new(seed),
        }
    }

    /// Current number of waiting proposers and receivers.
    pub fn waiting(&self) -> (usize, usize) {
        (self.waiting_p.len(), self.waiting_r.len())
    }

    /// Run a matching round on the current pools: match by proximity (stable
    /// matching on distance preferences), remove matched agents, and accumulate
    /// match count and distance into `stats`.
    fn match_round(&mut self, stats: &mut OnlineStats) {
        let n_p = self.waiting_p.len();
        let n_r = self.waiting_r.len();
        if n_p == 0 || n_r == 0 {
            return;
        }
        let prop_prefs: Vec<Vec<usize>> = (0..n_p)
            .map(|i| {
                let scores: Vec<f64> = (0..n_r)
                    .map(|j| -dist(self.waiting_p[i], self.waiting_r[j]))
                    .collect();
                rank_by_scores(&scores)
            })
            .collect();
        let recv_prefs: Vec<Vec<usize>> = (0..n_r)
            .map(|j| {
                let scores: Vec<f64> = (0..n_p)
                    .map(|i| -dist(self.waiting_r[j], self.waiting_p[i]))
                    .collect();
                rank_by_scores(&scores)
            })
            .collect();

        let m = gale_shapley(&prop_prefs, &recv_prefs);
        let mut matched_p = vec![false; n_p];
        let mut matched_r = vec![false; n_r];
        for (i, &slot) in m.proposer.iter().enumerate() {
            if let Some(j) = slot {
                stats.matched += 1;
                stats.total_distance += dist(self.waiting_p[i], self.waiting_r[j]);
                matched_p[i] = true;
                matched_r[j] = true;
            }
        }
        // Keep only the unmatched.
        self.waiting_p = (0..n_p)
            .filter(|&i| !matched_p[i])
            .map(|i| self.waiting_p[i])
            .collect();
        self.waiting_r = (0..n_r)
            .filter(|&j| !matched_r[j])
            .map(|j| self.waiting_r[j])
            .collect();
    }

    /// Drop waiting agents who abandon this tick, counting them in `stats`.
    fn abandon_round(&mut self, stats: &mut OnlineStats) {
        if self.abandon == 0.0 {
            return;
        }
        let before = self.waiting_p.len() + self.waiting_r.len();
        let a = self.abandon;
        let rng = &mut self.rng;
        self.waiting_p.retain(|_| rng.uniform() >= a);
        self.waiting_r.retain(|_| rng.uniform() >= a);
        stats.abandoned += before - (self.waiting_p.len() + self.waiting_r.len());
    }

    /// Run the market for `ticks` ticks under `policy`, returning summary stats.
    pub fn run(&mut self, ticks: usize, policy: Policy) -> OnlineStats {
        let mut stats = OnlineStats {
            matched: 0,
            abandoned: 0,
            total_distance: 0.0,
        };
        for tick in 0..ticks {
            // Arrivals.
            for _ in 0..self.rng.poisson(self.arrivals) {
                let p = (self.rng.uniform(), self.rng.uniform());
                self.waiting_p.push(p);
            }
            for _ in 0..self.rng.poisson(self.arrivals) {
                let r = (self.rng.uniform(), self.rng.uniform());
                self.waiting_r.push(r);
            }
            // Match, per policy.
            let do_match = match policy {
                Policy::Greedy => true,
                Policy::Batched(k) => k > 0 && tick % k == k - 1,
            };
            if do_match {
                self.match_round(&mut stats);
            }
            // Abandonment.
            self.abandon_round(&mut stats);
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_matches_and_leaves_small_pools() {
        let mut m = OnlineMarket::new(3.0, 0.0, 1);
        let stats = m.run(500, Policy::Greedy);
        assert!(stats.matched > 0);
        // With balanced arrivals and immediate matching, the pool stays small.
        let (wp, wr) = m.waiting();
        assert!(wp.min(wr) < 5, "greedy left a large pool: {wp}, {wr}");
    }

    #[test]
    fn batching_improves_match_quality() {
        // Batching accumulates a thicker pool, so the stable matching pairs
        // closer partners: lower mean distance than matching every tick.
        let greedy = OnlineMarket::new(3.0, 0.0, 7).run(4000, Policy::Greedy);
        let batched = OnlineMarket::new(3.0, 0.0, 7).run(4000, Policy::Batched(8));
        assert!(
            batched.mean_distance() < greedy.mean_distance(),
            "batched {} not better than greedy {}",
            batched.mean_distance(),
            greedy.mean_distance()
        );
    }

    #[test]
    fn batching_costs_more_abandonment_under_impatience() {
        // When agents are impatient, waiting to batch loses more of them.
        let greedy = OnlineMarket::new(3.0, 0.05, 11).run(4000, Policy::Greedy);
        let batched = OnlineMarket::new(3.0, 0.05, 11).run(4000, Policy::Batched(8));
        assert!(
            batched.abandoned > greedy.abandoned,
            "batched abandoned {} not more than greedy {}",
            batched.abandoned,
            greedy.abandoned
        );
    }
}
