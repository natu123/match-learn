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
//! This is the Phase 1 (mechanism-proof) core; see the project Roadmap for the
//! path toward dynamic pricing x supply-demand matching.

pub mod eval;
pub mod learner;
pub mod many_to_one;
pub mod market;
pub mod matching;
pub mod prefs;
pub mod rng;
pub mod ttc;
pub mod two_sided;

pub use eval::{LearningMarket, Report, simulate};
pub use learner::{GaussianThompson, PreferenceLearner, Ucb1};
pub use many_to_one::{ManyToOne, hospital_residents};
pub use market::Market;
pub use matching::{Matching, gale_shapley, is_stable};
pub use prefs::{break_ties, rank_by_scores, rank_by_scores_random, restrict_to_acceptable};
pub use rng::Rng;
pub use ttc::top_trading_cycle;
pub use two_sided::TwoSidedMarket;
