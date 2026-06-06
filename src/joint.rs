//! Joint pricing x matching (Phase 7).
//!
//! Where [`marketplace`](crate::marketplace) treats the good as homogeneous
//! (matched volume is just `min(demand, supply)`), here the two sides are
//! *heterogeneous*: agents have preferences over each other. A price gates
//! *participation* — a demand agent enters only if the price is below its value,
//! a supply agent only if the price covers its cost — and the entrants are then
//! matched by preference with [Gale-Shapley](crate::matching). This is where the
//! two halves of the project meet: pricing decides *who* is in the market, and
//! stable matching decides *who pairs with whom*.
//!
//! Raising the price thins demand but thickens supply (and vice versa), so the
//! matched volume is largest at an interior price — which a bandit can learn
//! over a stream of markets, exactly as the pricing policy does.

use crate::matching::gale_shapley;
use crate::prefs::restrict_to_acceptable;
use crate::rng::Rng;

/// One realized two-sided market with values, costs, and preferences.
#[derive(Debug, Clone)]
pub struct JointInstance {
    /// Each demand agent's value (its maximum willingness to pay).
    pub demand_values: Vec<f64>,
    /// Each supply agent's cost (the minimum price it will accept).
    pub supply_costs: Vec<f64>,
    /// Each demand agent's ranking over supply agents.
    pub demand_prefs: Vec<Vec<usize>>,
    /// Each supply agent's ranking over demand agents.
    pub supply_prefs: Vec<Vec<usize>>,
}

impl JointInstance {
    /// Number of pairs matched at `price`: gate participation by value/cost, then
    /// run Gale-Shapley among the entrants.
    pub fn matched_at(&self, price: f64) -> usize {
        let n_d = self.demand_values.len();
        let n_s = self.supply_costs.len();

        // Who clears the price on each side.
        let demand_in: Vec<bool> = self.demand_values.iter().map(|&v| v >= price).collect();
        let supply_in: Vec<bool> = self.supply_costs.iter().map(|&c| c <= price).collect();

        // Restrict each entrant's list to entrants on the other side; non-entrants
        // get empty lists and so match no one.
        let demand_prefs: Vec<Vec<usize>> = (0..n_d)
            .map(|i| {
                if demand_in[i] {
                    restrict_to_acceptable(&self.demand_prefs[i], &supply_in)
                } else {
                    Vec::new()
                }
            })
            .collect();
        let supply_prefs: Vec<Vec<usize>> = (0..n_s)
            .map(|j| {
                if supply_in[j] {
                    restrict_to_acceptable(&self.supply_prefs[j], &demand_in)
                } else {
                    Vec::new()
                }
            })
            .collect();

        gale_shapley(&demand_prefs, &supply_prefs).pairs()
    }

    /// Total gains-from-trade at `price`: the sum of `value - cost` over matched
    /// pairs (always non-negative, since a matched pair clears the price on both
    /// sides).
    pub fn welfare_at(&self, price: f64) -> f64 {
        let n_d = self.demand_values.len();
        let n_s = self.supply_costs.len();
        let demand_in: Vec<bool> = self.demand_values.iter().map(|&v| v >= price).collect();
        let supply_in: Vec<bool> = self.supply_costs.iter().map(|&c| c <= price).collect();

        let demand_prefs: Vec<Vec<usize>> = (0..n_d)
            .map(|i| {
                if demand_in[i] {
                    restrict_to_acceptable(&self.demand_prefs[i], &supply_in)
                } else {
                    Vec::new()
                }
            })
            .collect();
        let supply_prefs: Vec<Vec<usize>> = (0..n_s)
            .map(|j| {
                if supply_in[j] {
                    restrict_to_acceptable(&self.supply_prefs[j], &demand_in)
                } else {
                    Vec::new()
                }
            })
            .collect();

        let m = gale_shapley(&demand_prefs, &supply_prefs);
        m.proposer
            .iter()
            .enumerate()
            .filter_map(|(i, &slot)| slot.map(|j| self.demand_values[i] - self.supply_costs[j]))
            .sum()
    }
}

/// Draw a random `n_d x n_s` market: values and costs uniform in `[0, 1]`,
/// preferences uniformly random.
pub fn random_joint_instance(rng: &mut Rng, n_d: usize, n_s: usize) -> JointInstance {
    JointInstance {
        demand_values: (0..n_d).map(|_| rng.uniform()).collect(),
        supply_costs: (0..n_s).map(|_| rng.uniform()).collect(),
        demand_prefs: (0..n_d).map(|_| rng.permutation(n_s)).collect(),
        supply_prefs: (0..n_s).map(|_| rng.permutation(n_d)).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{PreferenceLearner, Ucb1};
    use crate::pricing::price_grid;

    #[test]
    fn extreme_prices_match_little() {
        // A balanced market: values and costs both uniform on [0,1].
        let mut rng = Rng::new(1);
        let inst = random_joint_instance(&mut rng, 30, 30);
        // Price ~1.0: almost no demand enters. Price ~0.0: almost no supply.
        assert!(inst.matched_at(0.98) <= 2, "too many matched at high price");
        assert!(inst.matched_at(0.02) <= 2, "too many matched at low price");
        // An interior price matches many more.
        assert!(
            inst.matched_at(0.5) > 8,
            "interior price matched only {}",
            inst.matched_at(0.5)
        );
    }

    #[test]
    fn an_interior_price_maximizes_matched_volume() {
        let mut rng = Rng::new(7);
        let inst = random_joint_instance(&mut rng, 40, 40);
        let grid = price_grid(0.05, 0.95, 19);
        let best = grid
            .iter()
            .max_by_key(|&&p| inst.matched_at(p))
            .copied()
            .unwrap();
        // The optimum is interior, not at either boundary.
        assert!(
            best > 0.2 && best < 0.8,
            "best price {best} is not interior"
        );
    }

    #[test]
    fn welfare_is_nonnegative_and_peaks_interior() {
        let mut rng = Rng::new(11);
        let inst = random_joint_instance(&mut rng, 40, 40);
        let grid = price_grid(0.05, 0.95, 19);
        for &p in &grid {
            assert!(inst.welfare_at(p) >= 0.0, "negative welfare at {p}");
        }
        let best = grid
            .iter()
            .max_by(|&&a, &&b| inst.welfare_at(a).partial_cmp(&inst.welfare_at(b)).unwrap())
            .copied()
            .unwrap();
        assert!(
            best > 0.2 && best < 0.8,
            "welfare-best price {best} not interior"
        );
    }

    #[test]
    fn a_bandit_learns_the_joint_optimal_price() {
        // Each round a fresh random market arrives; the platform posts a grid
        // price and observes matched volume. UCB should converge to a price near
        // the one that maximizes expected matched volume.
        let grid = price_grid(0.05, 0.95, 19);

        // Monte-Carlo oracle: expected matched per price over many markets.
        let mut oracle_rng = Rng::new(100);
        let mut totals = vec![0u64; grid.len()];
        for _ in 0..400 {
            let inst = random_joint_instance(&mut oracle_rng, 20, 20);
            for (k, &p) in grid.iter().enumerate() {
                totals[k] += inst.matched_at(p) as u64;
            }
        }
        let oracle_arm = (0..grid.len()).max_by_key(|&k| totals[k]).unwrap();
        let oracle_price = grid[oracle_arm];

        // Learn online.
        let mut learner = Ucb1::new(grid.len(), 3.0);
        let mut rng = Rng::new(500);
        for _ in 0..15_000 {
            let inst = random_joint_instance(&mut rng, 20, 20);
            let arm = learner.ranking()[0];
            let matched = inst.matched_at(grid[arm]);
            learner.update(arm, matched as f64);
        }
        let means = learner.means();
        let learned_arm = (0..grid.len())
            .max_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap())
            .unwrap();
        let learned_price = grid[learned_arm];

        let step = grid[1] - grid[0];
        assert!(
            (learned_price - oracle_price).abs() <= 2.0 * step,
            "learned joint price {learned_price} not near oracle {oracle_price}"
        );
    }
}
