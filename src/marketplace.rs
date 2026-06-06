//! Dynamic pricing over a supply-demand queue (Phase 7).
//!
//! The path toward the project's long-term goal: matching is no longer just
//! pairing fixed agents, it is *clearing a market*. Each round a platform posts
//! a price; demand and supply arrive at rates that depend on that price (riders
//! request when the price is below their value, drivers work when it is above
//! their cost); arrivals join queues; the platform matches as many pairs as it
//! can. Too cheap and demand piles up; too dear and supply idles. The
//! market-clearing price balances the two — and a later piece will *learn* it
//! online when the response curves are unknown.
//!
//! This module is the deterministic-mechanics core: the price-response model,
//! the queueing dynamics, and the closed-form clearing price.

use crate::rng::Rng;

/// Price-responsive demand: mean arrivals fall linearly to zero at `max_price`.
///
/// `mean(p) = base * max(0, 1 - p / max_price)`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Demand {
    /// Arrival intensity at price zero.
    pub base: f64,
    /// Price at which demand vanishes (the highest willingness-to-pay).
    pub max_price: f64,
}

impl Demand {
    /// Mean demand arrivals at `price`.
    pub fn mean(&self, price: f64) -> f64 {
        (self.base * (1.0 - price / self.max_price)).max(0.0)
    }
}

/// Price-responsive supply: mean arrivals rise linearly, saturating at
/// `ref_price`.
///
/// `mean(p) = base * min(1, p / ref_price)` for `p >= 0`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Supply {
    /// Arrival intensity once price reaches `ref_price`.
    pub base: f64,
    /// Price at which supply saturates.
    pub ref_price: f64,
}

impl Supply {
    /// Mean supply arrivals at `price`.
    pub fn mean(&self, price: f64) -> f64 {
        (self.base * (price / self.ref_price)).clamp(0.0, self.base)
    }
}

/// What happened in one round.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RoundOutcome {
    /// Price posted this round.
    pub price: f64,
    /// Demand arrivals this round.
    pub demand_arrivals: usize,
    /// Supply arrivals this round.
    pub supply_arrivals: usize,
    /// Pairs matched this round.
    pub matched: usize,
    /// Demand left waiting after matching.
    pub demand_queue: usize,
    /// Supply left waiting after matching.
    pub supply_queue: usize,
    /// Revenue collected this round (`matched * price`).
    pub revenue: f64,
}

/// A single-good marketplace with price-responsive arrivals and two queues.
#[derive(Debug, Clone)]
pub struct Marketplace {
    demand: Demand,
    supply: Supply,
    /// Per-round probability that each waiting agent abandons its queue.
    abandon: f64,
    demand_queue: usize,
    supply_queue: usize,
    rng: Rng,
}

impl Marketplace {
    /// Create a marketplace. `abandon` in `[0, 1)` is the per-round chance a
    /// waiting agent leaves (set `0.0` for patient queues).
    pub fn new(demand: Demand, supply: Supply, abandon: f64, seed: u64) -> Self {
        assert!((0.0..1.0).contains(&abandon), "abandon must be in [0, 1)");
        Self {
            demand,
            supply,
            abandon,
            demand_queue: 0,
            supply_queue: 0,
            rng: Rng::new(seed),
        }
    }

    /// Current demand-side queue length.
    pub fn demand_queue(&self) -> usize {
        self.demand_queue
    }

    /// Current supply-side queue length.
    pub fn supply_queue(&self) -> usize {
        self.supply_queue
    }

    /// The market-clearing price, where mean demand equals mean supply.
    ///
    /// Solving `D(1 - p/Pd) = S(p/Ps)` gives `p* = D / (S/Ps + D/Pd)`. Within the
    /// model's price range this is where the queues stay balanced.
    pub fn clearing_price(&self) -> f64 {
        let d = self.demand.base;
        let s = self.supply.base;
        d / (s / self.supply.ref_price + d / self.demand.max_price)
    }

    /// Drop waiting agents who abandon this round.
    fn apply_abandonment(&mut self) {
        if self.abandon == 0.0 {
            return;
        }
        let leave = |queue: usize, rng: &mut Rng| -> usize {
            (0..queue).filter(|_| rng.uniform() < self.abandon).count()
        };
        self.demand_queue -= leave(self.demand_queue, &mut self.rng);
        self.supply_queue -= leave(self.supply_queue, &mut self.rng);
    }

    /// Post `price`, take arrivals, match, and return the round outcome.
    pub fn step(&mut self, price: f64) -> RoundOutcome {
        let d = self.rng.poisson(self.demand.mean(price));
        let s = self.rng.poisson(self.supply.mean(price));
        self.demand_queue += d;
        self.supply_queue += s;

        let matched = self.demand_queue.min(self.supply_queue);
        self.demand_queue -= matched;
        self.supply_queue -= matched;

        self.apply_abandonment();

        RoundOutcome {
            price,
            demand_arrivals: d,
            supply_arrivals: s,
            matched,
            demand_queue: self.demand_queue,
            supply_queue: self.supply_queue,
            revenue: matched as f64 * price,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn market(seed: u64) -> Marketplace {
        let demand = Demand {
            base: 10.0,
            max_price: 20.0,
        };
        let supply = Supply {
            base: 10.0,
            ref_price: 10.0,
        };
        Marketplace::new(demand, supply, 0.0, seed)
    }

    #[test]
    fn clearing_price_balances_mean_arrivals() {
        let m = market(1);
        let p = m.clearing_price();
        let md = m.demand.mean(p);
        let ms = m.supply.mean(p);
        assert!(
            (md - ms).abs() < 1e-9,
            "demand {md} != supply {ms} at p*={p}"
        );
    }

    #[test]
    fn at_clearing_price_queues_stay_bounded() {
        let mut m = market(7);
        let p = m.clearing_price();
        let mut max_queue = 0;
        for _ in 0..5000 {
            let o = m.step(p);
            max_queue = max_queue.max(o.demand_queue.max(o.supply_queue));
        }
        // Balanced arrivals: neither queue runs away (random walk, not drift).
        // 5000 rounds at this scale stays well under a few hundred.
        assert!(
            max_queue < 400,
            "queue grew to {max_queue} at clearing price"
        );
    }

    #[test]
    fn too_cheap_floods_the_demand_queue() {
        let mut m = market(3);
        // Price well below clearing: demand >> supply.
        let cheap = m.clearing_price() * 0.3;
        for _ in 0..2000 {
            m.step(cheap);
        }
        assert!(
            m.demand_queue() > 500,
            "demand queue only {} when underpriced",
            m.demand_queue()
        );
        assert_eq!(m.supply_queue(), 0);
    }

    #[test]
    fn too_dear_idles_the_supply_queue() {
        let mut m = market(4);
        // Price well above clearing: supply >> demand.
        let dear = m.clearing_price() * 1.7;
        for _ in 0..2000 {
            m.step(dear);
        }
        assert!(
            m.supply_queue() > 500,
            "supply queue only {} when overpriced",
            m.supply_queue()
        );
        assert_eq!(m.demand_queue(), 0);
    }

    #[test]
    fn clearing_price_maximizes_matched_volume() {
        // Sweep prices; the clearing price should match the most over a run.
        let star = market(11).clearing_price();
        let throughput = |price: f64| -> usize {
            let mut m = market(11);
            (0..3000).map(|_| m.step(price).matched).sum()
        };
        let at_star = throughput(star);
        for mult in [0.5, 0.7, 1.3, 1.6] {
            let other = throughput(star * mult);
            assert!(
                at_star >= other,
                "throughput at p*={star:.2} ({at_star}) < at {:.2} ({other})",
                star * mult
            );
        }
    }

    #[test]
    fn abandonment_drains_a_starved_queue() {
        // With abandonment, an over-supplied queue does not grow without bound.
        let demand = Demand {
            base: 2.0,
            max_price: 20.0,
        };
        let supply = Supply {
            base: 10.0,
            ref_price: 10.0,
        };
        let mut m = Marketplace::new(demand, supply, 0.1, 5);
        let mut last = 0;
        for _ in 0..3000 {
            last = m.step(8.0).supply_queue;
        }
        // Abandonment caps the queue near arrival_rate/abandon, not infinity.
        assert!(last < 200, "supply queue {last} not bounded by abandonment");
    }
}
