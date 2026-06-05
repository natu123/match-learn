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

pub mod learner;
pub mod market;
pub mod matching;
pub mod rng;

pub use learner::{GaussianThompson, PreferenceLearner, Ucb1};
pub use market::Market;
pub use matching::{gale_shapley, is_stable, Matching};
pub use rng::Rng;
