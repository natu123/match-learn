//! Irreversible matching with interviews — the falsifiable experiment behind
//! match-learn paper (1): *interviews substitute for reversibility*.
//!
//! Learning a two-sided market needs information about preferences. There are two
//! ways to get it safely, and this module shows that a market is learnable iff at
//! least one is present:
//!
//! - **reversibility** — if a match can be undone, you can learn by matching and
//!   correcting your mistakes (the classic competing-bandits setting);
//! - **interviews** — safe pre-application observations that do not commit you.
//!
//! Take both away and learning is impossible. With *irreversible* matches (a
//! commit is absorbing: no deferral, no undo, a wrong match is a permanent loss)
//! and no interviews, the only way to sample a firm is to commit to it — so at the
//! moment of commitment you have observed nothing, and a near-tie cannot be
//! resolved. Any policy then suffers `Ω(T)` regret (the Heaven-or-Hell
//! obstruction, Plaut et al. 2025). The full picture is a clean 2x2 in the
//! cumulative-regret *shape*:
//!
//! ```text
//!                 recoverable            irreversible
//!  no interview   log T   [bandit]       Ω(T)     [Theorem A]
//!  interview      O(1)    [Mirfakhar]    log T    [Theorem B]
//! ```
//!
//! Each missing channel costs a factor: both present → `O(1)`, exactly one →
//! `log T` (reversibility *or* interviews suffice, and an interview substitutes
//! for an undo), neither → `Ω(T)`. The two `log T` cells even share the rate
//! while differing in mechanism: the bandit pays an *exploration* cost (it
//! samples by matching), the irreversible interviewer pays a *confidence* cost
//! (it must reach error `≈ 1/T` before an absorbing commit, so `log T` returns —
//! this is why interviews are only an *accelerator* when matches are recoverable
//! but a *necessary condition* when they are not).
//!
//! The theorems are proved elsewhere; [`simulate`] is their in-silico check. Run
//! the four regimes and read the regret-growth column: only irreversible +
//! no-interview grows linearly. See `examples/irreversible_interviews.rs`.

use crate::rng::Rng;

/// A market of agents facing firms, each pair `(a, f)` with a true mean reward
/// `means[a][f] ∈ [0, 1]`. Observations (match rewards and interviews) are
/// `Bernoulli(means[a][f])`.
#[derive(Debug, Clone)]
pub struct Market {
    means: Vec<Vec<f64>>,
}

impl Market {
    /// Build a market from explicit true means (`means[a][f]`).
    pub fn new(means: Vec<Vec<f64>>) -> Self {
        Market { means }
    }

    /// A **Heaven-or-Hell** market: every agent has one *heaven* firm (mean
    /// `heaven`), one *decoy* firm (mean `heaven - gap`, the near-tie that must be
    /// resolved to choose optimally), and the rest *hell* firms (mean `hell`),
    /// placed at uniformly random positions. The deciding gap is `gap`, which
    /// plays the role of the admissible gap `Δ_A`.
    pub fn heaven_or_hell(
        n_agents: usize,
        m_firms: usize,
        heaven: f64,
        gap: f64,
        hell: f64,
        rng: &mut Rng,
    ) -> Self {
        assert!(m_firms >= 2, "need at least a heaven and a decoy");
        let means = (0..n_agents)
            .map(|_| {
                let mut row = vec![hell; m_firms];
                let perm = rng.permutation(m_firms);
                row[perm[0]] = heaven;
                row[perm[1]] = heaven - gap;
                row
            })
            .collect();
        Market { means }
    }

    fn n_agents(&self) -> usize {
        self.means.len()
    }

    fn m_firms(&self) -> usize {
        self.means.first().map_or(0, Vec::len)
    }

    /// Each agent's best (highest-mean) firm.
    fn optima(&self) -> Vec<usize> {
        self.means
            .iter()
            .map(|row| {
                (0..row.len())
                    .max_by(|&i, &j| row[i].partial_cmp(&row[j]).unwrap())
                    .unwrap()
            })
            .collect()
    }
}

/// How an agent gathers information while unmatched.
#[derive(Debug, Clone, Copy)]
pub enum Policy {
    /// No interviews. With reversible matches this is a UCB1 bandit (learn by
    /// matching and switching); with irreversible matches it must commit blind.
    NoInterview,
    /// Take `per_round` safe interviews each round, then commit once the deciding
    /// gap is resolved. The required confidence depends on reversibility: a
    /// reversible agent can settle at a constant error (mistakes are fixable),
    /// while an irreversible agent must reach error `≈ 1/T` before its absorbing
    /// commit.
    Interview { per_round: usize },
}

/// The result of a run.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// Cumulative pseudo-regret after each round (length = horizon).
    pub regret: Vec<f64>,
    /// Agents that committed *irreversibly* to a catastrophic firm (mean below
    /// the threshold). Always `0` in the reversible regimes.
    pub catastrophes: usize,
    /// Agents settled on their best firm.
    pub optimal_commits: usize,
    /// Number of agents.
    pub n_agents: usize,
}

impl Outcome {
    /// The final cumulative regret.
    pub fn final_regret(&self) -> f64 {
        self.regret.last().copied().unwrap_or(0.0)
    }

    /// The fraction of agents that committed to a catastrophic firm.
    pub fn catastrophe_rate(&self) -> f64 {
        self.catastrophes as f64 / self.n_agents.max(1) as f64
    }
}

/// Run `market` under `policy` for `horizon` rounds. `reversible` selects the
/// recoverability regime: when `true`, an agent can change its match every round
/// and observes the matched firm's reward; when `false`, the first commit is
/// absorbing. A firm with mean below `catastrophe_threshold` is catastrophic.
/// Regret is pseudo-regret against each agent's best firm (an unmatched agent
/// pays the full opportunity cost each round).
pub fn simulate(
    market: &Market,
    reversible: bool,
    policy: Policy,
    horizon: usize,
    catastrophe_threshold: f64,
    seed: u64,
) -> Outcome {
    let mut rng = Rng::new(seed);
    let n = market.n_agents();
    let m = market.m_firms();
    let means = &market.means;
    let best = market.optima();

    // Interview confidence. A reversible agent can settle at a constant error
    // (it can undo a mistake); an irreversible one must be near-certain before an
    // absorbing commit — error ≈ 1/T — which is what reintroduces the log T.
    let delta = if reversible {
        0.05
    } else {
        1.0 / horizon as f64
    };
    let conf = (2.0 * m as f64 / delta).ln();

    let mut sum = vec![vec![0.0; m]; n];
    let mut cnt = vec![vec![0usize; m]; n];
    let mut settled: Vec<Option<usize>> = vec![None; n];

    let mut regret = Vec::with_capacity(horizon);
    let mut cumulative = 0.0;

    for t in 0..horizon {
        let mut round = 0.0;
        for a in 0..n {
            let shortfall = match policy {
                Policy::NoInterview => {
                    if reversible {
                        // Learn by matching: pull a firm, observe, keep the option
                        // to switch (UCB1). Recoverable, so never absorbing.
                        let f = ucb_pick(&sum[a], &cnt[a], t);
                        let obs = sample(&mut rng, means[a][f]);
                        sum[a][f] += obs;
                        cnt[a][f] += 1;
                        means[a][best[a]] - means[a][f]
                    } else if let Some(f) = settled[a] {
                        means[a][best[a]] - means[a][f]
                    } else {
                        // Nothing has been observed and a commit is irreversible:
                        // the agent can only commit blind. It locks to firm 0.
                        settled[a] = Some(0);
                        means[a][best[a]] - means[a][0]
                    }
                }
                Policy::Interview { per_round } => {
                    if let Some(f) = settled[a] {
                        means[a][best[a]] - means[a][f]
                    } else if let Some(b) = confident_best(&sum[a], &cnt[a], conf) {
                        settled[a] = Some(b);
                        means[a][best[a]] - means[a][b]
                    } else {
                        for _ in 0..per_round {
                            let f = least_sampled(&cnt[a]);
                            let obs = sample(&mut rng, means[a][f]);
                            sum[a][f] += obs;
                            cnt[a][f] += 1;
                        }
                        // Unmatched while interviewing: full opportunity cost.
                        means[a][best[a]]
                    }
                }
            };
            round += shortfall;
        }
        cumulative += round;
        regret.push(cumulative);
    }

    let mut catastrophes = 0;
    let mut optimal_commits = 0;
    for a in 0..n {
        if let Some(f) = settled[a] {
            if !reversible && means[a][f] < catastrophe_threshold {
                catastrophes += 1;
            }
            if f == best[a] {
                optimal_commits += 1;
            }
        }
    }

    Outcome {
        regret,
        catastrophes,
        optimal_commits,
        n_agents: n,
    }
}

/// A `Bernoulli(p)` observation.
fn sample(rng: &mut Rng, p: f64) -> f64 {
    if rng.uniform() < p { 1.0 } else { 0.0 }
}

/// The least-sampled firm (ties to the lowest index): round-robin interviews.
fn least_sampled(cnt: &[usize]) -> usize {
    (0..cnt.len()).min_by_key(|&f| cnt[f]).unwrap()
}

/// UCB1 arm choice: pull any unsampled firm, else the highest upper-confidence
/// bound `mean + sqrt(2 ln t / n_f)`.
fn ucb_pick(sum: &[f64], cnt: &[usize], t: usize) -> usize {
    if let Some(f) = (0..cnt.len()).find(|&f| cnt[f] == 0) {
        return f;
    }
    let ln_t = ((t + 1) as f64).ln();
    let ucb = |f: usize| sum[f] / cnt[f] as f64 + (2.0 * ln_t / cnt[f] as f64).sqrt();
    (0..cnt.len())
        .max_by(|&i, &j| ucb(i).partial_cmp(&ucb(j)).unwrap())
        .unwrap()
}

/// The empirically best firm, if its Hoeffding lower bound clears every other
/// firm's upper bound — i.e. the deciding gap is resolved at confidence `conf`
/// (where `conf = ln(2m/δ)` controls the radius `sqrt(conf / 2n_f)`). `None`
/// until every firm has been sampled and the separation holds.
fn confident_best(sum: &[f64], cnt: &[usize], conf: f64) -> Option<usize> {
    if cnt.contains(&0) {
        return None;
    }
    let mean = |f: usize| sum[f] / cnt[f] as f64;
    let radius = |f: usize| (conf / (2.0 * cnt[f] as f64)).sqrt();
    let b = (0..cnt.len())
        .max_by(|&i, &j| mean(i).partial_cmp(&mean(j)).unwrap())
        .unwrap();
    let lcb_best = mean(b) - radius(b);
    let ucb_rest = (0..cnt.len())
        .filter(|&f| f != b)
        .map(|f| mean(f) + radius(f))
        .fold(f64::NEG_INFINITY, f64::max);
    (lcb_best >= ucb_rest).then_some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn market(gap: f64, seed: u64) -> Market {
        Market::heaven_or_hell(400, 4, 0.9, gap, 0.1, &mut Rng::new(seed))
    }

    const INTERVIEW: Policy = Policy::Interview { per_round: 2 };

    /// Irreversible + no interview (Theorem A): commit blind, a constant fraction
    /// land in catastrophe, and regret is linear in the horizon.
    #[test]
    fn irreversible_no_interview_is_catastrophic_and_linear() {
        let out = simulate(&market(0.2, 0xA), false, Policy::NoInterview, 400, 0.3, 1);
        assert!(
            out.catastrophe_rate() > 0.3,
            "catastrophe rate {}",
            out.catastrophe_rate()
        );
        let ratio = out.regret[399] / out.regret[199];
        assert!(
            ratio > 1.9 && ratio < 2.1,
            "regret ratio {ratio} not linear"
        );
    }

    /// Irreversible + interview (Theorem B, corrected): catastrophes vanish, but
    /// regret is logarithmic in the horizon, *not* flat — a larger horizon demands
    /// more confidence (`δ ≈ 1/T`), hence more interviews.
    #[test]
    fn irreversible_interview_is_logarithmic_not_flat() {
        let short = simulate(&market(0.3, 0xB), false, INTERVIEW, 2_000, 0.3, 1);
        let long = simulate(&market(0.3, 0xB), false, INTERVIEW, 16_000, 0.3, 1);
        assert_eq!(short.catastrophes, 0);
        assert_eq!(long.catastrophes, 0);
        // Grows with the horizon (not flat) but far slower than linear (not Ω(T)).
        let ratio = long.final_regret() / short.final_regret();
        assert!(ratio > 1.08, "regret did not grow with horizon: {ratio}");
        assert!(ratio < 1.6, "regret grew too fast for log T: {ratio}");
    }

    /// Reversible + no interview: a UCB bandit learns by matching and correcting,
    /// so regret is logarithmic and nothing is stuck in catastrophe.
    #[test]
    fn reversible_no_interview_is_logarithmic() {
        let out = simulate(&market(0.2, 0xC), true, Policy::NoInterview, 2_000, 0.3, 1);
        assert_eq!(out.catastrophes, 0);
        let ratio = out.regret[1999] / out.regret[999];
        assert!(ratio > 1.0 && ratio < 1.5, "not logarithmic: ratio {ratio}");
    }

    /// Reversible + interview (Mirfakhar): a constant confidence suffices because
    /// mistakes are fixable, so regret is horizon-independent — flat.
    #[test]
    fn reversible_interview_is_constant() {
        let short = simulate(&market(0.3, 0xD), true, INTERVIEW, 2_000, 0.3, 1);
        let long = simulate(&market(0.3, 0xD), true, INTERVIEW, 16_000, 0.3, 1);
        let ratio = long.final_regret() / short.final_regret();
        assert!(
            ratio < 1.06,
            "reversible interview regret not flat: {ratio}"
        );
    }

    /// Theorem B quantified: at a fixed horizon the irreversible interview regret
    /// scales like `1/Δ²` — halving the deciding gap roughly quadruples it.
    #[test]
    fn irreversible_interview_regret_scales_inverse_square_in_gap() {
        let wide = simulate(&market(0.4, 0xE), false, INTERVIEW, 8_000, 0.3, 1);
        let narrow = simulate(&market(0.2, 0xE), false, INTERVIEW, 8_000, 0.3, 1);
        let ratio = narrow.final_regret() / wide.final_regret();
        assert!((2.5..6.0).contains(&ratio), "inverse-square off: {ratio}");
    }

    #[test]
    fn same_seed_same_run() {
        let m = market(0.3, 7);
        let a = simulate(&m, false, Policy::NoInterview, 100, 0.3, 42);
        let b = simulate(&m, false, Policy::NoInterview, 100, 0.3, 42);
        assert_eq!(a.regret, b.regret);
    }
}
