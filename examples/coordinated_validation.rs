//! Does coordination actually cure the cascade stall end-to-end?
//!
//! ```text
//! cargo run --release --example coordinated_validation
//! ```
//!
//! Builds markets with a deliberate near-tie (one proposer nearly indifferent
//! between two receivers) — the cascade trigger — and compares five policies on
//! tail stability and regret: plain Thompson, forced-explore Thompson, the
//! ungated `CoordinatedMarket`, the Prop-4 `GatedCoordinatedMarket`, and the
//! `StabilityCoordinatedMarket`.
//!
//! The story:
//! - the **ungated** `CoordinatedMarket` maximizes belief welfare every round and
//!   is *much* less stable than plain Thompson (~0.70 vs 0.92) — the negative
//!   finding;
//! - the **gated** coordinator only coordinates *certified* near-ties, so it never
//!   reorders an un-converged pair. It recovers nearly all the lost stability
//!   (~0.91 at a tight band) and bounds the rest — Prop 4 guarantees
//!   `2·eps`-stability, not strict stability, so a small eps-controlled gap to
//!   plain Thompson remains by design. `eps` tunes the welfare/stability tradeoff;
//! - the **stability** coordinator fixes the *objective* rather than gating it:
//!   it minimizes expected blocking pairs instead of belief welfare, so it has no
//!   `2·eps` ceiling and reaches the *highest* stability of all — above plain
//!   Thompson — at the cost of some proposer welfare (positive regret).

use match_learn::{
    CoordinatedMarket, GatedCoordinatedMarket, Market, Rng, StabilityCoordinatedMarket, simulate,
};

/// A market with `n` proposers where proposer 0 is nearly indifferent between
/// receivers 0 and 1 (means within `gap`), a near-tie that can cascade.
fn near_tie_market(rng: &mut Rng, n: usize, gap: f64) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let mut util: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    // Force a near-tie at the top of proposer 0's preferences.
    util[0][0] = 0.90;
    util[0][1] = 0.90 - gap;
    let receiver_prefs: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
    (util, receiver_prefs)
}

fn main() {
    let markets = 200;
    let n = 5;
    let rounds = 3000;
    let noise = 0.2;
    let gap = 0.01; // a tight near-tie
    let tail = rounds / 5;

    let mut seedgen = Rng::new(2026);
    let (mut ts_stable, mut ts_regret) = (0.0, 0.0);
    let (mut fe_stable, mut fe_regret) = (0.0, 0.0);
    let (mut co_stable, mut co_regret) = (0.0, 0.0);
    let (mut ga_stable, mut ga_regret) = (0.0, 0.0);
    let (mut gt_stable, mut gt_regret) = (0.0, 0.0);
    let (mut st_stable, mut st_regret) = (0.0, 0.0);

    for _ in 0..markets {
        let seed = (seedgen.below(1_000_000_000) as u64) + 1;
        let mut g = Rng::new(seed);
        let (util, recv) = near_tie_market(&mut g, n, gap);

        let mut ts = Market::with_thompson(
            util.clone(),
            recv.clone(),
            0.5,
            1.0,
            noise * noise,
            noise,
            seed ^ 0xABCD,
        );
        let r = simulate(&mut ts, rounds);
        ts_stable += r.tail_stable_fraction(tail);
        ts_regret += r.tail_mean_regret(tail);

        let mut fe = Market::with_forced_explore(
            util.clone(),
            recv.clone(),
            0.5,
            1.0,
            noise * noise,
            0.5,
            noise,
            seed ^ 0xABCD,
        );
        let r = simulate(&mut fe, rounds);
        fe_stable += r.tail_stable_fraction(tail);
        fe_regret += r.tail_mean_regret(tail);

        let mut co = CoordinatedMarket::new(
            util.clone(),
            recv.clone(),
            0.5,
            1.0,
            noise * noise,
            0.05,
            0.5,
            noise,
            seed ^ 0xABCD,
        );
        let r = simulate(&mut co, rounds);
        co_stable += r.tail_stable_fraction(tail);
        co_regret += r.tail_mean_regret(tail);

        let mut ga = GatedCoordinatedMarket::new(
            util.clone(),
            recv.clone(),
            0.5,
            1.0,
            noise * noise,
            0.05,
            1.96,
            0.5,
            noise,
            seed ^ 0xABCD,
        );
        let r = simulate(&mut ga, rounds);
        ga_stable += r.tail_stable_fraction(tail);
        ga_regret += r.tail_mean_regret(tail);

        // A tighter near-tie band certifies fewer pairs, trading welfare back for
        // strict stability (eps -> 0 recovers the forced-explore baseline).
        let mut gt = GatedCoordinatedMarket::new(
            util.clone(),
            recv.clone(),
            0.5,
            1.0,
            noise * noise,
            0.02,
            1.96,
            0.5,
            noise,
            seed ^ 0xABCD,
        );
        let r = simulate(&mut gt, rounds);
        gt_stable += r.tail_stable_fraction(tail);
        gt_regret += r.tail_mean_regret(tail);

        // The stability-targeting coordinator: minimizes expected blocking pairs
        // instead of belief welfare, so it has no 2*eps stability ceiling.
        let mut st = StabilityCoordinatedMarket::new(
            util,
            recv,
            0.5,
            1.0,
            noise * noise,
            0.05,
            0.5,
            noise,
            seed ^ 0xABCD,
        );
        let r = simulate(&mut st, rounds);
        st_stable += r.tail_stable_fraction(tail);
        st_regret += r.tail_mean_regret(tail);
    }

    let m = markets as f64;
    println!("Cascade cure validation ({markets} near-tie {n}x{n} markets, gap={gap})\n");
    println!(
        "  {:<24} {:>16} {:>16}",
        "policy", "tail stable frac", "tail regret/round"
    );
    let row = |name: &str, s: f64, r: f64| println!("  {name:<24} {:>16.3} {:>16.4}", s / m, r / m);
    row("plain Thompson", ts_stable, ts_regret);
    row("forced-explore Thompson", fe_stable, fe_regret);
    row("CoordinatedMarket", co_stable, co_regret);
    row("GatedCoordinatedMarket eps=.05", ga_stable, ga_regret);
    row("GatedCoordinatedMarket eps=.02", gt_stable, gt_regret);
    row("StabilityCoordinatedMarket", st_stable, st_regret);
    println!();
    println!("Honest reading (judge by stability, not regret-vs-M*): the ungated");
    println!("CoordinatedMarket maximizes *belief welfare*, so it raises proposer welfare");
    println!("(negative regret) but is *much* less stable than plain Thompson -- the negative");
    println!(
        "finding. It is not a learning artifact: belief-welfare maximization is unstable even"
    );
    println!("with accurate beliefs (welfare-max != stable-max). So the GATED coordinator, which");
    println!("only coordinates a *certified* near-tie, can *bound* the damage (Prop 4:");
    println!("2*eps-stability) but not erase it -- gated eps=.02 reaches ~0.91, still an");
    println!(
        "eps-controlled margin under plain Thompson. The StabilityCoordinatedMarket fixes the"
    );
    println!("*objective*: it minimizes expected blocking pairs (not welfare), so it has no 2*eps");
    println!("ceiling and reaches the *highest* stability of all, above plain Thompson. Its");
    println!("positive regret-vs-M* is the tell that regret is the wrong metric here -- it lands");
    println!("on stable matchings that need not be proposer-optimal, trading a little proposer");
    println!("welfare for strictly more stable matchings. (Cf. docs/theory-identifiability.md,");
    println!("Prop 4 and section 4a.)");
}
