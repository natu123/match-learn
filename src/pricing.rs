//! Learning a dynamic pricing policy (Phase 7).
//!
//! In [`marketplace`](crate::marketplace) the clearing price is computed from
//! known demand and supply curves. A real platform does *not* know those curves
//! — it must **learn** the right price online from what it observes (matches,
//! revenue). This closes the loop with the rest of the library: pricing becomes
//! a bandit over a price grid, reusing the very same [`PreferenceLearner`]s that
//! drive the matching markets. The "preference" being learned is the platform's
//! payoff as a function of price.

use crate::learner::{PreferenceLearner, Ucb1};
use crate::marketplace::{Marketplace, RoundOutcome};

/// What the platform is optimizing for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Objective {
    /// Matched volume per round (maximized at the clearing price).
    Throughput,
    /// Revenue per round (`matched * price`; favours higher prices).
    Revenue,
}

/// An evenly spaced price grid of `n` points over `[min, max]` (inclusive).
pub fn price_grid(min: f64, max: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2 && max > min, "need n >= 2 and max > min");
    (0..n)
        .map(|i| min + (max - min) * i as f64 / (n - 1) as f64)
        .collect()
}

/// A pricing policy that learns the best grid price online.
pub struct LearnedPricer {
    grid: Vec<f64>,
    learner: Box<dyn PreferenceLearner>,
    objective: Objective,
}

impl LearnedPricer {
    /// Build a pricer over `grid` using an explicit learner.
    pub fn new(grid: Vec<f64>, learner: Box<dyn PreferenceLearner>, objective: Objective) -> Self {
        assert_eq!(
            learner.n_arms(),
            grid.len(),
            "learner must have one arm per grid price"
        );
        Self {
            grid,
            learner,
            objective,
        }
    }

    /// Build a pricer that uses UCB1 over the grid.
    pub fn with_ucb(grid: Vec<f64>, c: f64, objective: Objective) -> Self {
        let learner = Box::new(Ucb1::new(grid.len(), c)) as Box<dyn PreferenceLearner>;
        Self::new(grid, learner, objective)
    }

    /// Post a price chosen by the learner, advance the market, and update on the
    /// objective. Returns the round outcome.
    pub fn step(&mut self, market: &mut Marketplace) -> RoundOutcome {
        let arm = self.learner.ranking()[0];
        let price = self.grid[arm];
        let outcome = market.step(price);
        let reward = match self.objective {
            Objective::Throughput => outcome.matched as f64,
            Objective::Revenue => outcome.revenue,
        };
        self.learner.update(arm, reward);
        outcome
    }

    /// The grid price the policy currently believes is best (greedy estimate).
    pub fn best_price(&self) -> f64 {
        let means = self.learner.means();
        let arm = (0..means.len())
            .max_by(|&a, &b| {
                means[a]
                    .partial_cmp(&means[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        self.grid[arm]
    }

    /// The price grid.
    pub fn grid(&self) -> &[f64] {
        &self.grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marketplace::{Demand, Marketplace, Supply};

    fn market(seed: u64) -> Marketplace {
        let demand = Demand {
            base: 12.0,
            max_price: 20.0,
        };
        let supply = Supply {
            base: 12.0,
            ref_price: 10.0,
        };
        // Light abandonment keeps queues finite during exploration.
        Marketplace::new(demand, supply, 0.02, seed)
    }

    #[test]
    fn grid_is_evenly_spaced() {
        let g = price_grid(0.0, 10.0, 6);
        assert_eq!(g, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn learns_a_price_near_the_clearing_price() {
        let clearing = market(0).clearing_price();
        let grid = price_grid(1.0, 18.0, 18);
        let step = grid[1] - grid[0];

        let mut pricer = LearnedPricer::with_ucb(grid, 3.0, Objective::Throughput);
        let mut m = market(123);
        for _ in 0..20_000 {
            pricer.step(&mut m);
        }
        let learned = pricer.best_price();
        assert!(
            (learned - clearing).abs() <= 1.5 * step,
            "learned price {learned} not near clearing {clearing} (grid step {step})"
        );
    }

    #[test]
    fn throughput_beats_a_naive_fixed_price() {
        // Compare learned throughput to a naive policy that always posts the
        // highest price (intuitive "maximize per-match revenue" mistake).
        let grid = price_grid(1.0, 18.0, 18);
        let naive_price = *grid.last().unwrap();

        let mut pricer = LearnedPricer::with_ucb(grid, 3.0, Objective::Throughput);
        let mut learned_market = market(7);
        let mut learned_matched = 0usize;
        for _ in 0..20_000 {
            learned_matched += pricer.step(&mut learned_market).matched;
        }

        let mut naive_market = market(7);
        let mut naive_matched = 0usize;
        for _ in 0..20_000 {
            naive_matched += naive_market.step(naive_price).matched;
        }

        assert!(
            learned_matched > naive_matched * 2,
            "learned matched {learned_matched} not >> naive {naive_matched}"
        );
    }

    #[test]
    fn learned_throughput_approaches_best_fixed_price() {
        // The learned policy should match nearly as much as the best fixed grid
        // price chosen with hindsight (an oracle).
        let grid = price_grid(1.0, 18.0, 18);

        let oracle_best: usize = grid
            .iter()
            .map(|&p| {
                let mut m = market(7);
                (0..8000).map(|_| m.step(p).matched).sum()
            })
            .max()
            .unwrap();

        let mut pricer = LearnedPricer::with_ucb(grid, 3.0, Objective::Throughput);
        let mut m = market(7);
        let learned: usize = (0..8000).map(|_| pricer.step(&mut m).matched).sum();

        // Within ~20% of the hindsight-best fixed price.
        assert!(
            learned as f64 > 0.8 * oracle_best as f64,
            "learned {learned} vs oracle-best fixed {oracle_best}"
        );
    }

    /// Run a UCB pricer and return (total matched, mean absolute queue imbalance).
    fn run_pricer(c: f64, rounds: usize) -> (usize, f64) {
        let grid = price_grid(1.0, 18.0, 18);
        let mut pricer = LearnedPricer::with_ucb(grid, c, Objective::Throughput);
        let mut m = market(7);
        let mut matched = 0usize;
        let mut imbalance = 0.0;
        for _ in 0..rounds {
            let o = pricer.step(&mut m);
            matched += o.matched;
            imbalance += (o.demand_queue as f64 - o.supply_queue as f64).abs();
        }
        (matched, imbalance / rounds as f64)
    }

    #[test]
    fn moderate_exploration_beats_under_exploration_on_both_costs() {
        // The regret-queue tradeoff: too little exploration locks onto a wrong
        // price, which hurts *both* matched volume (regret) and queue balance.
        // A moderate exploration constant matches more and keeps queues calmer.
        let (matched_low, imbalance_low) = run_pricer(0.1, 12_000);
        let (matched_mid, imbalance_mid) = run_pricer(0.7, 12_000);

        assert!(
            matched_mid > matched_low,
            "moderate exploration should match more: mid={matched_mid}, low={matched_low}"
        );
        assert!(
            imbalance_mid < imbalance_low,
            "moderate exploration should balance queues better: mid={imbalance_mid}, low={imbalance_low}"
        );
    }
}
