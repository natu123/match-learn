//! # match-learn
//!
//! **Stable matching that learns.** Online preference learning combined with
//! stable matching, built from scratch in Rust.
//!
//! Two-sided matching markets where each side's preferences are *unknown* and
//! learned online (Thompson Sampling / UCB), while a stable matching is kept at
//! every step. As preferences are learned, the matching converges toward the
//! stable optimum.
//!
//! See the project Roadmap for the path from this core toward dynamic pricing x
//! supply-demand matching.
//!
//! # Examples
//!
//! A stable matching:
//!
//! ```
//! use match_learn::gale_shapley;
//!
//! // Two proposers and two receivers, both sides preferring index 0.
//! let proposers = vec![vec![0, 1], vec![0, 1]];
//! let receivers = vec![vec![0, 1], vec![0, 1]];
//! let m = gale_shapley(&proposers, &receivers);
//! assert_eq!(m.proposer, vec![Some(0), Some(1)]);
//! ```
//!
//! Learning a market online — proposers do not know their own utilities and
//! learn them while a stable matching is kept every round:
//!
//! ```
//! use match_learn::{Market, simulate};
//!
//! let util = vec![vec![1.0, 0.2], vec![0.3, 1.0]]; // true utilities (unknown to agents)
//! let receiver_prefs = vec![vec![0, 1], vec![1, 0]];
//! let mut market = Market::with_thompson(util, receiver_prefs, 0.5, 1.0, 0.04, 0.2, 42);
//!
//! let report = simulate(&mut market, 2000);
//! assert!(report.tail_stable_fraction(400) > 0.9); // converged to the stable match
//! ```
//!
//! Learning the market-clearing price for a supply-demand queue:
//!
//! ```
//! use match_learn::marketplace::{Demand, Marketplace, Supply};
//! use match_learn::pricing::{LearnedPricer, Objective, price_grid};
//!
//! let demand = Demand { base: 12.0, max_price: 20.0 };
//! let supply = Supply { base: 12.0, ref_price: 10.0 };
//! let mut market = Marketplace::new(demand, supply, 0.02, 7);
//!
//! let grid = price_grid(1.0, 18.0, 18);
//! let mut pricer = LearnedPricer::with_ucb(grid, 0.7, Objective::Throughput);
//! for _ in 0..5000 {
//!     pricer.step(&mut market);
//! }
//! // The learned price sits near the analytic clearing price.
//! let clearing = market.clearing_price();
//! assert!((pricer.best_price() - clearing).abs() < 3.0);
//! ```

pub mod allocation;
pub mod applications;
pub mod assignment;
pub mod auction;
pub mod boston;
pub mod contextual;
pub mod coordinated;
pub mod data;
pub mod eval;
pub mod fairness;
pub mod joint;
pub mod learner;
pub mod linalg;
pub mod many_to_many;
pub mod many_to_one;
pub mod market;
pub mod marketplace;
pub mod matching;
pub mod online;
pub mod parallel;
pub mod prefs;
pub mod pricing;
#[cfg(feature = "python")]
mod python;
pub mod reserves;
pub mod rng;
pub mod strategyproof;
pub mod ties;
pub mod ttc;
pub mod two_sided;

pub use allocation::{
    is_pareto_efficient, probabilistic_serial, random_serial_dictatorship, sd_envy_free,
    serial_dictatorship,
};
pub use applications::{
    Crowdsourcing, Delivery, RideHailing, random_crowdsourcing, random_delivery,
    random_ride_hailing,
};
pub use assignment::{max_weight_assignment, min_cost_assignment};
pub use auction::{AuctionResult, double_auction, efficient_quantity, mcafee_auction};
pub use boston::boston_mechanism;
pub use contextual::LinearThompson;
pub use coordinated::{CoordinatedMarket, GatedCoordinatedMarket, near_tie_rankings};
pub use data::{correlated_market, from_text, to_text};
pub use eval::{LearningMarket, Report, simulate};
pub use fairness::{egalitarian_cost, egalitarian_stable, sex_equal_stable, sex_equality_cost};
pub use joint::{JointInstance, random_joint_instance};
pub use learner::{
    DiscountedThompson, ForcedExploreThompson, GaussianThompson, PreferenceLearner, Ucb1,
};
pub use many_to_many::{ManyToMany, is_pairwise_stable, many_to_many};
pub use many_to_one::{ManyToOne, hospital_residents};
pub use market::Market;
pub use marketplace::{Demand, Marketplace, RoundOutcome, Supply};
pub use matching::{Matching, all_stable_matchings, gale_shapley, is_stable};
pub use online::{OnlineMarket, OnlineStats, Policy};
pub use parallel::simulate_batch;
pub use prefs::{break_ties, rank_by_scores, rank_by_scores_random, restrict_to_acceptable};
pub use pricing::{LearnedPricer, Objective, price_grid};
pub use reserves::{deferred_acceptance_with_reserves, reserved_type_matched};
pub use rng::Rng;
pub use strategyproof::{proposer_manipulation, receiver_manipulation};
pub use ties::{
    is_strongly_stable, is_super_stable, is_weakly_stable, strongly_stable, super_stable,
    weakly_stable,
};
pub use ttc::top_trading_cycle;
pub use two_sided::TwoSidedMarket;
