//! Application adapters: mapping real platforms onto the core (Phase 8).
//!
//! The library's abstractions ([`JointInstance`](crate::joint::JointInstance),
//! [`Marketplace`](crate::marketplace::Marketplace), the learners) are
//! domain-agnostic. This module shows how a concrete platform maps onto them, so
//! the path from "matching library" to "ride-hailing / delivery engine" is
//! short and explicit.
//!
//! The flagship example is **ride-hailing**: riders and drivers sit on a plane,
//! each prefers nearer counterparts (shorter ETA), and a surge price gates who
//! participates (riders ride only if the fare is below their value, drivers work
//! only if it covers their cost). That is exactly a [`JointInstance`] — proximity
//! preferences plus value/cost gating — so surge pricing is learned with the
//! same bandit that learns any other price.

use crate::joint::JointInstance;
use crate::prefs::rank_by_scores;
use crate::rng::Rng;

/// Euclidean distance between two points on the plane.
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// A snapshot of a ride-hailing market: riders and drivers with positions,
/// values, and costs.
#[derive(Debug, Clone)]
pub struct RideHailing {
    /// Each rider's value for a ride (maximum fare they will pay).
    pub rider_values: Vec<f64>,
    /// Each driver's cost to serve (minimum fare they will accept).
    pub driver_costs: Vec<f64>,
    /// Rider positions on the unit plane.
    pub rider_pos: Vec<(f64, f64)>,
    /// Driver positions on the unit plane.
    pub driver_pos: Vec<(f64, f64)>,
}

impl RideHailing {
    /// Convert to a [`JointInstance`]: riders are demand, drivers are supply, and
    /// each side prefers nearer counterparts (shorter pickup distance).
    pub fn to_joint(&self) -> JointInstance {
        let n_r = self.rider_pos.len();
        let n_d = self.driver_pos.len();

        // Riders prefer the closest drivers (highest score = smallest distance).
        let demand_prefs: Vec<Vec<usize>> = (0..n_r)
            .map(|i| {
                let scores: Vec<f64> = (0..n_d)
                    .map(|j| -dist(self.rider_pos[i], self.driver_pos[j]))
                    .collect();
                rank_by_scores(&scores)
            })
            .collect();
        // Drivers likewise prefer the closest riders.
        let supply_prefs: Vec<Vec<usize>> = (0..n_d)
            .map(|j| {
                let scores: Vec<f64> = (0..n_r)
                    .map(|i| -dist(self.driver_pos[j], self.rider_pos[i]))
                    .collect();
                rank_by_scores(&scores)
            })
            .collect();

        JointInstance {
            demand_values: self.rider_values.clone(),
            supply_costs: self.driver_costs.clone(),
            demand_prefs,
            supply_prefs,
        }
    }

    /// Rides matched at fare `price` (riders below their value and drivers above
    /// their cost enter, then are matched by proximity).
    pub fn rides_at(&self, price: f64) -> usize {
        self.to_joint().matched_at(price)
    }
}

/// A random ride-hailing snapshot: uniform positions on the unit square, rider
/// values and driver costs uniform in `[0, 1]`.
pub fn random_ride_hailing(rng: &mut Rng, n_riders: usize, n_drivers: usize) -> RideHailing {
    let point = |rng: &mut Rng| (rng.uniform(), rng.uniform());
    RideHailing {
        rider_values: (0..n_riders).map(|_| rng.uniform()).collect(),
        driver_costs: (0..n_drivers).map(|_| rng.uniform()).collect(),
        rider_pos: (0..n_riders).map(|_| point(rng)).collect(),
        driver_pos: (0..n_drivers).map(|_| point(rng)).collect(),
    }
}

/// A snapshot of a delivery market: orders (pickup + dropoff) and couriers.
///
/// Orders are demand, couriers are supply. An order prefers couriers near its
/// pickup (fast collection); a courier prefers orders with the least total
/// effort (deadhead to pickup plus the delivery leg). A delivery fee gates
/// participation, just like ride-hailing's surge.
#[derive(Debug, Clone)]
pub struct Delivery {
    /// Each order's value (maximum delivery fee it will pay).
    pub order_values: Vec<f64>,
    /// Each courier's cost (minimum fee it will accept).
    pub courier_costs: Vec<f64>,
    /// Pickup positions, one per order.
    pub pickup: Vec<(f64, f64)>,
    /// Dropoff positions, one per order.
    pub dropoff: Vec<(f64, f64)>,
    /// Courier positions.
    pub courier_pos: Vec<(f64, f64)>,
}

impl Delivery {
    /// Convert to a [`JointInstance`]. Orders rank couriers by pickup proximity;
    /// couriers rank orders by total effort (deadhead + delivery distance).
    pub fn to_joint(&self) -> JointInstance {
        let n_o = self.pickup.len();
        let n_c = self.courier_pos.len();

        let demand_prefs: Vec<Vec<usize>> = (0..n_o)
            .map(|i| {
                let scores: Vec<f64> = (0..n_c)
                    .map(|j| -dist(self.courier_pos[j], self.pickup[i]))
                    .collect();
                rank_by_scores(&scores)
            })
            .collect();
        let supply_prefs: Vec<Vec<usize>> = (0..n_c)
            .map(|j| {
                let scores: Vec<f64> = (0..n_o)
                    .map(|i| {
                        let deadhead = dist(self.courier_pos[j], self.pickup[i]);
                        let leg = dist(self.pickup[i], self.dropoff[i]);
                        -(deadhead + leg)
                    })
                    .collect();
                rank_by_scores(&scores)
            })
            .collect();

        JointInstance {
            demand_values: self.order_values.clone(),
            supply_costs: self.courier_costs.clone(),
            demand_prefs,
            supply_prefs,
        }
    }

    /// Deliveries matched at delivery fee `fee`.
    pub fn deliveries_at(&self, fee: f64) -> usize {
        self.to_joint().matched_at(fee)
    }
}

/// A random delivery snapshot: uniform positions on the unit square, order
/// values and courier costs uniform in `[0, 1]`.
pub fn random_delivery(rng: &mut Rng, n_orders: usize, n_couriers: usize) -> Delivery {
    let point = |rng: &mut Rng| (rng.uniform(), rng.uniform());
    Delivery {
        order_values: (0..n_orders).map(|_| rng.uniform()).collect(),
        courier_costs: (0..n_couriers).map(|_| rng.uniform()).collect(),
        pickup: (0..n_orders).map(|_| point(rng)).collect(),
        dropoff: (0..n_orders).map(|_| point(rng)).collect(),
        courier_pos: (0..n_couriers).map(|_| point(rng)).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{PreferenceLearner, Ucb1};
    use crate::pricing::price_grid;

    #[test]
    fn riders_prefer_nearer_drivers() {
        let market = RideHailing {
            rider_values: vec![1.0],
            driver_costs: vec![0.0, 0.0],
            rider_pos: vec![(0.0, 0.0)],
            driver_pos: vec![(0.9, 0.9), (0.1, 0.1)], // driver 1 is much closer
        };
        let joint = market.to_joint();
        // The rider's top choice is the nearer driver (index 1).
        assert_eq!(joint.demand_prefs[0][0], 1);
    }

    #[test]
    fn surge_gates_participation() {
        let mut rng = Rng::new(1);
        let market = random_ride_hailing(&mut rng, 30, 30);
        // A high fare prices most riders out; a very low fare prices out drivers.
        assert!(market.rides_at(0.97) <= 3);
        assert!(market.rides_at(0.03) <= 3);
        // A mid fare clears far more rides.
        assert!(market.rides_at(0.5) > 8);
    }

    #[test]
    fn a_bandit_learns_the_surge_price() {
        // Over a stream of random ride-hailing snapshots, learn the fare that
        // maximizes completed rides, and compare to a Monte-Carlo oracle.
        let grid = price_grid(0.05, 0.95, 19);

        let mut oracle_rng = Rng::new(100);
        let mut totals = vec![0u64; grid.len()];
        for _ in 0..400 {
            let market = random_ride_hailing(&mut oracle_rng, 20, 20);
            for (k, &p) in grid.iter().enumerate() {
                totals[k] += market.rides_at(p) as u64;
            }
        }
        let oracle_price = grid[(0..grid.len()).max_by_key(|&k| totals[k]).unwrap()];

        let mut learner = Ucb1::new(grid.len(), 3.0);
        let mut rng = Rng::new(500);
        for _ in 0..15_000 {
            let market = random_ride_hailing(&mut rng, 20, 20);
            let arm = learner.ranking()[0];
            let rides = market.rides_at(grid[arm]);
            learner.update(arm, rides as f64);
        }
        let means = learner.means();
        let learned = grid[(0..grid.len())
            .max_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap())
            .unwrap()];

        let step = grid[1] - grid[0];
        assert!(
            (learned - oracle_price).abs() <= 2.0 * step,
            "learned surge {learned} not near oracle {oracle_price}"
        );
    }

    #[test]
    fn orders_prefer_couriers_near_pickup() {
        let market = Delivery {
            order_values: vec![1.0],
            courier_costs: vec![0.0, 0.0],
            pickup: vec![(0.0, 0.0)],
            dropoff: vec![(0.5, 0.5)],
            courier_pos: vec![(0.9, 0.9), (0.1, 0.1)], // courier 1 is near the pickup
        };
        let joint = market.to_joint();
        assert_eq!(joint.demand_prefs[0][0], 1);
    }

    #[test]
    fn couriers_prefer_lower_total_effort_orders() {
        // One courier, two orders: a short local job vs a long cross-town haul.
        let market = Delivery {
            order_values: vec![1.0, 1.0],
            courier_costs: vec![0.0],
            pickup: vec![(0.1, 0.1), (0.1, 0.1)],
            dropoff: vec![(0.2, 0.2), (0.9, 0.9)], // order 0 is the short delivery
            courier_pos: vec![(0.0, 0.0)],
        };
        let joint = market.to_joint();
        assert_eq!(joint.supply_prefs[0][0], 0);
    }

    #[test]
    fn delivery_fee_gates_and_a_bandit_learns_it() {
        let grid = price_grid(0.05, 0.95, 19);

        let mut oracle_rng = Rng::new(100);
        let mut totals = vec![0u64; grid.len()];
        for _ in 0..400 {
            let market = random_delivery(&mut oracle_rng, 20, 20);
            for (k, &p) in grid.iter().enumerate() {
                totals[k] += market.deliveries_at(p) as u64;
            }
        }
        let oracle = grid[(0..grid.len()).max_by_key(|&k| totals[k]).unwrap()];

        let mut learner = Ucb1::new(grid.len(), 3.0);
        let mut rng = Rng::new(500);
        for _ in 0..15_000 {
            let market = random_delivery(&mut rng, 20, 20);
            let arm = learner.ranking()[0];
            learner.update(arm, market.deliveries_at(grid[arm]) as f64);
        }
        let means = learner.means();
        let learned = grid[(0..grid.len())
            .max_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap())
            .unwrap()];

        let step = grid[1] - grid[0];
        assert!(
            (learned - oracle).abs() <= 2.0 * step,
            "learned fee {learned} not near oracle {oracle}"
        );
    }
}
