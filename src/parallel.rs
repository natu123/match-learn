//! Parallel batch evaluation across many markets.
//!
//! Sweeps — gates, benchmarks, parameter scans — run many independent markets,
//! which is embarrassingly parallel. [`simulate_batch`] spreads a batch across
//! worker threads using only `std::thread` (no external dependency; a future
//! version could swap in Rayon). Each market is built *inside* its worker from a
//! factory closure, so the markets themselves never cross threads — only the
//! resulting [`Report`]s come back.
//!
//! Results are returned in market-index order and are bit-for-bit identical to
//! running the same factory sequentially, so parallelism never changes outcomes.

use crate::eval::{LearningMarket, Report, simulate};
use std::thread;

/// Run `n_markets` simulations of `rounds` rounds each, across up to `threads`
/// worker threads.
///
/// `factory(i)` builds the `i`-th market (typically seeding it from `i`). The
/// returned `Vec<Report>` is in index order. `threads` is clamped to at least 1
/// and at most `n_markets`.
pub fn simulate_batch<M, F>(
    n_markets: usize,
    rounds: usize,
    threads: usize,
    factory: F,
) -> Vec<Report>
where
    M: LearningMarket,
    F: Fn(usize) -> M + Sync,
{
    if n_markets == 0 {
        return Vec::new();
    }
    let threads = threads.clamp(1, n_markets);
    let factory = &factory;

    // Each worker handles a strided subset of indices and returns (index, report)
    // pairs, which we scatter back into the right slots.
    let mut out: Vec<Option<Report>> = (0..n_markets).map(|_| None).collect();

    thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|w| {
                scope.spawn(move || {
                    let mut local = Vec::new();
                    let mut i = w;
                    while i < n_markets {
                        let mut market = factory(i);
                        local.push((i, simulate(&mut market, rounds)));
                        i += threads;
                    }
                    local
                })
            })
            .collect();

        for h in handles {
            for (i, report) in h.join().expect("worker thread panicked") {
                out[i] = Some(report);
            }
        }
    });

    out.into_iter()
        .map(|r| r.expect("every market was simulated"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{correlated_market, prefs_from_util};
    use crate::{Market, Rng};

    fn make_market(i: usize) -> Market {
        let seed = (i as u64 + 1) * 2654435761;
        let mut g = Rng::new(seed);
        let (util_p, util_r) = correlated_market(&mut g, 5, 5, 0.5);
        let receiver_prefs = prefs_from_util(&util_r);
        Market::with_thompson(util_p, receiver_prefs, 0.5, 1.0, 0.04, 0.2, seed ^ 0xABCD)
    }

    #[test]
    fn parallel_matches_sequential_bit_for_bit() {
        let n = 12;
        let rounds = 500;

        let sequential: Vec<Report> = (0..n)
            .map(|i| {
                let mut m = make_market(i);
                simulate(&mut m, rounds)
            })
            .collect();

        let parallel = simulate_batch(n, rounds, 4, make_market);

        assert_eq!(parallel.len(), n);
        for (s, p) in sequential.iter().zip(&parallel) {
            assert_eq!(s.cumulative_regret, p.cumulative_regret);
            assert_eq!(s.stable, p.stable);
        }
    }

    #[test]
    fn handles_edge_cases() {
        assert!(simulate_batch(0, 100, 4, make_market).is_empty());
        // More threads than markets is clamped, not a panic.
        let r = simulate_batch(2, 50, 16, make_market);
        assert_eq!(r.len(), 2);
        // Single thread still works.
        let r = simulate_batch(3, 50, 1, make_market);
        assert_eq!(r.len(), 3);
    }
}
