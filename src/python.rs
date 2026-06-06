//! Python bindings (PyO3), behind the optional `python` feature.
//!
//! Exposes the matching algorithms and the learning market loop to Python so
//! research code can drive the Rust core at Rust speed. Built as an extension
//! module with maturin; see `bench/` for the cross-language comparison.

use pyo3::prelude::*;

use crate::matching::Matching;

/// Convert a one-to-one matching to a Python-friendly list (`-1` = unmatched).
fn matching_to_vec(m: &Matching) -> Vec<i64> {
    m.proposer
        .iter()
        .map(|x| x.map_or(-1, |r| r as i64))
        .collect()
}

/// Proposer-optimal stable matching (Gale-Shapley). Returns the receiver each
/// proposer is matched to, or `-1` if unmatched.
#[pyfunction]
fn gale_shapley(proposer_prefs: Vec<Vec<usize>>, receiver_prefs: Vec<Vec<usize>>) -> Vec<i64> {
    matching_to_vec(&crate::gale_shapley(&proposer_prefs, &receiver_prefs))
}

/// Many-to-one (Hospital-Residents) stable matching. Returns the receiver each
/// proposer is assigned to, or `-1` if unmatched.
#[pyfunction]
fn hospital_residents(
    proposer_prefs: Vec<Vec<usize>>,
    receiver_prefs: Vec<Vec<usize>>,
    capacities: Vec<usize>,
) -> Vec<i64> {
    let m = crate::hospital_residents(&proposer_prefs, &receiver_prefs, &capacities);
    m.proposer
        .iter()
        .map(|x| x.map_or(-1, |r| r as i64))
        .collect()
}

/// Top Trading Cycles allocation for the housing market.
#[pyfunction]
fn top_trading_cycle(prefs: Vec<Vec<usize>>) -> Vec<usize> {
    crate::top_trading_cycle(&prefs)
}

/// A run report: the per-round regret/stability series plus summaries.
#[pyclass(name = "Report")]
struct PyReport {
    inner: crate::Report,
}

#[pymethods]
impl PyReport {
    /// Cumulative regret through each round.
    #[getter]
    fn cumulative_regret(&self) -> Vec<f64> {
        self.inner.cumulative_regret.clone()
    }

    /// Whether each round's matching was stable in the true market.
    #[getter]
    fn stable(&self) -> Vec<bool> {
        self.inner.stable.clone()
    }

    /// Number of rounds.
    #[getter]
    fn rounds(&self) -> usize {
        self.inner.rounds
    }

    /// Total regret over the run.
    fn total_regret(&self) -> f64 {
        self.inner.total_regret()
    }

    /// Fraction of the final `k` rounds whose matching was stable.
    fn tail_stable_fraction(&self, k: usize) -> f64 {
        self.inner.tail_stable_fraction(k)
    }

    fn __repr__(&self) -> String {
        format!(
            "Report(rounds={}, total_regret={:.3})",
            self.inner.rounds,
            self.inner.total_regret()
        )
    }
}

/// A one-sided-unknown learning market: learning proposers, known receiver
/// preferences.
#[pyclass(name = "Market")]
struct PyMarket {
    inner: crate::Market,
}

#[pymethods]
impl PyMarket {
    /// Build a market whose proposers use Gaussian Thompson Sampling.
    ///
    /// `util_p[p][r]` is proposer `p`'s true utility for receiver `r`;
    /// `receiver_prefs[r]` is receiver `r`'s known ranking over proposers.
    #[staticmethod]
    #[pyo3(signature = (util_p, receiver_prefs, prior_mean=0.5, prior_var=1.0, obs_var=0.04, noise=0.2, seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn thompson(
        util_p: Vec<Vec<f64>>,
        receiver_prefs: Vec<Vec<usize>>,
        prior_mean: f64,
        prior_var: f64,
        obs_var: f64,
        noise: f64,
        seed: u64,
    ) -> Self {
        Self {
            inner: crate::Market::with_thompson(
                util_p,
                receiver_prefs,
                prior_mean,
                prior_var,
                obs_var,
                noise,
                seed,
            ),
        }
    }

    /// Play one round; return the receiver each proposer matched (`-1` if none).
    fn step(&mut self) -> Vec<i64> {
        matching_to_vec(&self.inner.step())
    }

    /// Run `rounds` rounds and return a [`Report`].
    fn simulate(&mut self, rounds: usize) -> PyReport {
        PyReport {
            inner: crate::simulate(&mut self.inner, rounds),
        }
    }

    /// Number of proposers.
    #[getter]
    fn n_proposers(&self) -> usize {
        self.inner.n_proposers()
    }

    /// Number of receivers.
    #[getter]
    fn n_receivers(&self) -> usize {
        self.inner.n_receivers()
    }
}

/// The `match_learn` Python module.
#[pymodule]
fn match_learn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gale_shapley, m)?)?;
    m.add_function(wrap_pyfunction!(hospital_residents, m)?)?;
    m.add_function(wrap_pyfunction!(top_trading_cycle, m)?)?;
    m.add_class::<PyMarket>()?;
    m.add_class::<PyReport>()?;
    Ok(())
}
