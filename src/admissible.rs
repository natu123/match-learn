//! The **admissible gap** of a two-sided matching market (Basu, 2025).
//!
//! In competing bandits in matching markets, the instance-dependent stable
//! regret scales as `Θ(log T / Δ_A²)`, where `Δ_A` — Basu's *admissible gap*
//! (arXiv:2506.15926, Def. 7) — is the central difficulty parameter, matched by
//! both the upper bound and the instance-dependent lower bound. This module
//! computes it.
//!
//! `Δ_A` is the largest *minimum* preference gap that some *admissible*
//! coarsening of the true preferences can guarantee. A coarsening blurs together
//! arms (or users) whose utilities are close; it is **admissible** when the true
//! full ranking is compatible with it *and* a [super-stable](crate::ties)
//! matching still exists under it. Blurring a pair *raises* the smallest gap you
//! must still resolve — but only as far as super stability survives. `Δ_A` is the
//! best gap reachable this way: gaps below it never need to be resolved to reach
//! a stable matching, the gap at it does. This is the cardinal-utility
//! counterpart of the `σ²/Δ²` identification floor in
//! `docs/theory-identifiability.md`.
//!
//! # Why a partial order, not ties
//!
//! Keeping exactly the orderings with gap `≥ θ` is in general a *partial order*,
//! not a weak order (tie tiers): if `a≻b` has gap `≥ θ` and `b≻c` has gap `< θ`,
//! then `a≻c` has gap `≥ θ` (gaps add along the true order), so `a` outranks both
//! while `b, c` stay incomparable — which no tier structure expresses. So super
//! stability here is taken over partial orders; [`crate::ties::super_stable_irving`]
//! (weak orders) is the special case, not the general test.
//!
//! Preferences are cardinal utilities: `proposer_utils[p][a]` is proposer `p`'s
//! value for receiver `a`, and `receiver_utils[a][p]` is receiver `a`'s value for
//! proposer `p`. Markets are complete and equal-sized. `Δ_A = 0` signals that no
//! coarsening is admissible (e.g. exact ties already block super stability) — an
//! infinitely hard instance under the `1/Δ_A²` reading.

/// At resolution `theta`, does the agent with utilities `u` *strictly* prefer
/// `x` to `y` — rank `x` above `y` with a gap of at least `theta`? Pairs within
/// `theta` are blurred into incomparability.
fn prefers_at(u: &[f64], x: usize, y: usize, theta: f64) -> bool {
    u[x] - u[y] >= theta
}

/// Every permutation of `0..n` (the candidate perfect matchings).
fn all_permutations(n: usize) -> Vec<Vec<usize>> {
    let mut cur: Vec<usize> = (0..n).collect();
    let mut out = Vec::new();
    permute(&mut cur, 0, &mut out);
    out
}

fn permute(cur: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
    if k == cur.len() {
        out.push(cur.clone());
        return;
    }
    for i in k..cur.len() {
        cur.swap(k, i);
        permute(cur, k + 1, out);
        cur.swap(k, i);
    }
}

/// Is the perfect matching `mate` (proposer `p` ↦ receiver `mate[p]`)
/// super-stable at resolution `theta`? A pair super-blocks when *each* side
/// weakly prefers the other — i.e. neither strictly prefers its current partner.
fn is_super_stable_at(
    proposer_utils: &[Vec<f64>],
    receiver_utils: &[Vec<f64>],
    mate: &[usize],
    theta: f64,
) -> bool {
    let n = mate.len();
    let mut suitor = vec![0usize; n];
    for (p, &a) in mate.iter().enumerate() {
        suitor[a] = p;
    }
    for (p, &mate_p) in mate.iter().enumerate() {
        for a in 0..n {
            if mate_p == a {
                continue;
            }
            // p weakly prefers a iff it does not strictly prefer its own partner.
            let p_weak = !prefers_at(&proposer_utils[p], mate_p, a, theta);
            let a_weak = !prefers_at(&receiver_utils[a], suitor[a], p, theta);
            if p_weak && a_weak {
                return false;
            }
        }
    }
    true
}

/// Does some perfect matching survive as super-stable at resolution `theta`?
fn super_stable_exists_at(
    proposer_utils: &[Vec<f64>],
    receiver_utils: &[Vec<f64>],
    theta: f64,
) -> bool {
    let n = proposer_utils.len();
    all_permutations(n)
        .iter()
        .any(|mate| is_super_stable_at(proposer_utils, receiver_utils, mate, theta))
}

/// The distinct positive within-list preference gaps, ascending — the candidate
/// resolutions `θ` at which the threshold coarsening can change.
fn candidate_gaps(proposer_utils: &[Vec<f64>], receiver_utils: &[Vec<f64>]) -> Vec<f64> {
    let mut gaps: Vec<f64> = Vec::new();
    for side in [proposer_utils, receiver_utils] {
        for row in side {
            for x in 0..row.len() {
                for y in 0..row.len() {
                    let g = row[x] - row[y];
                    if g > 0.0 {
                        gaps.push(g);
                    }
                }
            }
        }
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    gaps.dedup();
    gaps
}

/// Basu's **admissible gap** `Δ_A` of a complete, equal-sized market given as
/// cardinal utilities.
///
/// `Δ_A` is the largest resolution `θ` at which the threshold coarsening "keep
/// orderings with gap `≥ θ`" still admits a super-stable matching. Super
/// stability is **monotone** in `θ`: a finer coarsening (smaller `θ`) only helps,
/// because refinement preserves super stability, so `super_stable_exists_at` is
/// true on a prefix of the ascending candidate gaps and false thereafter. We thus
/// **binary-search** the boundary — `O(log G)` super-stability checks over the `G`
/// distinct gaps instead of a linear scan. Returns `0.0` if no coarsening is
/// admissible (or `n < 2`).
///
/// Each super-stability test brute-forces over perfect matchings, so this is an
/// analysis tool for small markets, `O(n! · n²)` per check.
pub fn admissible_gap(proposer_utils: &[Vec<f64>], receiver_utils: &[Vec<f64>]) -> f64 {
    if proposer_utils.len() < 2 {
        return 0.0;
    }
    let gaps = candidate_gaps(proposer_utils, receiver_utils);
    // Binary-search the monotone predicate: `lo` settles at the first
    // non-admissible index, so the last admissible gap is `gaps[lo - 1]`.
    let mut lo = 0;
    let mut hi = gaps.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if super_stable_exists_at(proposer_utils, receiver_utils, gaps[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 { 0.0 } else { gaps[lo - 1] }
}

/// The original linear descending scan — the independent reference the binary
/// search is checked against.
#[cfg(test)]
fn admissible_gap_linear(proposer_utils: &[Vec<f64>], receiver_utils: &[Vec<f64>]) -> f64 {
    if proposer_utils.len() < 2 {
        return 0.0;
    }
    let mut gaps = candidate_gaps(proposer_utils, receiver_utils);
    gaps.reverse(); // coarsest first
    gaps.into_iter()
        .find(|&theta| super_stable_exists_at(proposer_utils, receiver_utils, theta))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// A market where each side's preferences are a strict permutation cycle:
    /// the unique stable matching is super-stable and cannot be coarsened, so
    /// `Δ_A` is the unit gap that distinguishes the deciding pair.
    #[test]
    fn unit_gap_market() {
        let prop = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let recv = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(admissible_gap(&prop, &recv), 1.0);
    }

    /// `Δ_A` is always at least the finest (minimum positive) gap, because the
    /// full strict order is admissible (it is just strict Gale-Shapley).
    #[test]
    fn at_least_the_minimum_gap() {
        let mut rng = Rng::new(0xD1A6);
        for _ in 0..300 {
            let n = 2 + rng.below(3); // 2..=4
            let prop = random_utils(n, &mut rng);
            let recv = random_utils(n, &mut rng);
            let da = admissible_gap(&prop, &recv);
            let min_gap = min_positive_gap(&prop, &recv);
            if let Some(mg) = min_gap {
                // Either no coarsening is admissible (0.0) or Δ_A ≥ the min gap.
                assert!(da == 0.0 || da >= mg - 1e-9, "da={da} min_gap={mg}");
            }
        }
    }

    /// The scan really lands on the boundary: the returned `Δ_A` is admissible,
    /// and any strictly larger gap (coarser coarsening) is not.
    #[test]
    fn returns_the_admissibility_boundary() {
        let mut rng = Rng::new(0xB0DA);
        for _ in 0..300 {
            let n = 2 + rng.below(3);
            let prop = random_utils(n, &mut rng);
            let recv = random_utils(n, &mut rng);
            let da = admissible_gap(&prop, &recv);
            if da > 0.0 {
                assert!(
                    super_stable_exists_at(&prop, &recv, da),
                    "Δ_A itself not admissible: da={da}"
                );
                // Any larger candidate gap must fail (coarser ⇒ less stable).
                for &g in &distinct_gaps(&prop, &recv) {
                    if g > da + 1e-9 {
                        assert!(
                            !super_stable_exists_at(&prop, &recv, g),
                            "a gap above Δ_A was admissible: g={g} da={da}"
                        );
                    }
                }
            }
        }
    }

    /// The headline check: the threshold scan equals the brute force over *all*
    /// admissible partial ranks (every transitively-closed sub-order of the true
    /// ranking on each side), confirming the threshold form is optimal.
    #[test]
    fn agrees_with_exhaustive_partial_rank_search() {
        let mut rng = Rng::new(0x9A17);
        let mut nontrivial = 0;
        for _ in 0..40 {
            let n = 2 + rng.below(2); // 2..=3 (exhaustive search is exponential)
            let prop = random_utils(n, &mut rng);
            let recv = random_utils(n, &mut rng);
            let fast = admissible_gap(&prop, &recv);
            let brute = admissible_gap_brute(&prop, &recv);
            assert!(
                (fast - brute).abs() < 1e-9,
                "Δ_A mismatch: fast={fast} brute={brute} prop={prop:?} recv={recv:?}"
            );
            if fast > min_positive_gap(&prop, &recv).unwrap_or(0.0) + 1e-9 {
                nontrivial += 1; // a case where coarsening strictly helped
            }
        }
        assert!(
            nontrivial > 0,
            "no case exercised a coarsening above the min gap"
        );
    }

    /// The monotonicity that licenses the binary search: super-stability holds on
    /// a prefix of the ascending candidate gaps and never reappears once lost.
    #[test]
    fn super_stability_is_monotone_in_theta() {
        let mut rng = Rng::new(0x3017);
        for _ in 0..300 {
            let n = 2 + rng.below(3);
            let prop = random_utils(n, &mut rng);
            let recv = random_utils(n, &mut rng);
            let flags: Vec<bool> = candidate_gaps(&prop, &recv)
                .iter()
                .map(|&t| super_stable_exists_at(&prop, &recv, t))
                .collect();
            if let Some(k) = flags.iter().position(|&f| !f) {
                assert!(
                    flags[k..].iter().all(|&f| !f),
                    "super stability reappeared after vanishing: {flags:?}"
                );
            }
        }
    }

    /// The binary search returns exactly what the linear scan does.
    #[test]
    fn binary_search_matches_linear_scan() {
        let mut rng = Rng::new(0x5EA4);
        for _ in 0..300 {
            let n = 2 + rng.below(3);
            let prop = random_utils(n, &mut rng);
            let recv = random_utils(n, &mut rng);
            assert_eq!(
                admissible_gap(&prop, &recv),
                admissible_gap_linear(&prop, &recv)
            );
        }
    }

    // --- test helpers ---

    /// Distinct per-agent utilities with *varied* gaps (cumulative random
    /// increments of `1..=4`), assigned to the other side in random order, so
    /// that coarsening small gaps but not large ones is genuinely possible.
    fn random_utils(n: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        (0..n)
            .map(|_| {
                let mut vals = Vec::with_capacity(n);
                let mut acc = 0i64;
                for _ in 0..n {
                    vals.push(acc as f64);
                    acc += 1 + rng.below(4) as i64; // next gap is 1..=4
                }
                let perm = rng.permutation(n);
                let mut row = vec![0.0; n];
                for (pos, &other) in perm.iter().enumerate() {
                    row[other] = vals[pos];
                }
                row
            })
            .collect()
    }

    fn distinct_gaps(prop: &[Vec<f64>], recv: &[Vec<f64>]) -> Vec<f64> {
        candidate_gaps(prop, recv)
    }

    fn min_positive_gap(prop: &[Vec<f64>], recv: &[Vec<f64>]) -> Option<f64> {
        distinct_gaps(prop, recv).first().copied()
    }

    /// All transitively-closed sub-orders of one agent's strict ranking, each as
    /// an `above[x][y]` matrix (`x` strictly preferred to `y`).
    #[allow(clippy::needless_range_loop)] // z is a column index into two rows
    fn agent_partial_ranks(u: &[f64]) -> Vec<Vec<Vec<bool>>> {
        let m = u.len();
        let fpairs: Vec<(usize, usize)> = (0..m)
            .flat_map(|x| (0..m).filter(move |&y| u[x] > u[y]).map(move |y| (x, y)))
            .collect();
        let k = fpairs.len();
        let mut out = Vec::new();
        for mask in 0u32..(1u32 << k) {
            let mut above = vec![vec![false; m]; m];
            for (bit, &(x, y)) in fpairs.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    above[x][y] = true;
                }
            }
            // Keep only transitively-closed relations.
            let mut closed = true;
            'check: for x in 0..m {
                for y in 0..m {
                    if above[x][y] {
                        for z in 0..m {
                            if above[y][z] && !above[x][z] {
                                closed = false;
                                break 'check;
                            }
                        }
                    }
                }
            }
            if closed {
                out.push(above);
            }
        }
        out
    }

    /// Is `mate` super-stable under explicit per-agent partial-order matrices?
    fn ss_under(
        prop_above: &[&Vec<Vec<bool>>],
        recv_above: &[&Vec<Vec<bool>>],
        mate: &[usize],
    ) -> bool {
        let n = mate.len();
        let mut suitor = vec![0usize; n];
        for (p, &a) in mate.iter().enumerate() {
            suitor[a] = p;
        }
        for (p, &mate_p) in mate.iter().enumerate() {
            for a in 0..n {
                if mate_p == a {
                    continue;
                }
                let p_weak = !prop_above[p][mate_p][a];
                let a_weak = !recv_above[a][suitor[a]][p];
                if p_weak && a_weak {
                    return false;
                }
            }
        }
        true
    }

    /// `Δ_A` by exhaustive search over every combination of admissible per-agent
    /// partial ranks — the definitional brute force, for `n ≤ 3`.
    fn admissible_gap_brute(prop: &[Vec<f64>], recv: &[Vec<f64>]) -> f64 {
        let n = prop.len();
        let prop_opts: Vec<Vec<Vec<Vec<bool>>>> =
            prop.iter().map(|u| agent_partial_ranks(u)).collect();
        let recv_opts: Vec<Vec<Vec<Vec<bool>>>> =
            recv.iter().map(|u| agent_partial_ranks(u)).collect();
        let perms = all_permutations(n);

        // Mixed-radix index over (prop agents..., recv agents...).
        let radices: Vec<usize> = prop_opts
            .iter()
            .chain(recv_opts.iter())
            .map(|o| o.len())
            .collect();
        let mut idx = vec![0usize; radices.len()];
        let mut best = 0.0f64;
        loop {
            // Assemble the chosen partial ranks.
            let prop_above: Vec<&Vec<Vec<bool>>> = (0..n).map(|p| &prop_opts[p][idx[p]]).collect();
            let recv_above: Vec<&Vec<Vec<bool>>> =
                (0..n).map(|a| &recv_opts[a][idx[n + a]]).collect();

            // Admissible iff some matching is super-stable under this combination.
            let admissible = perms
                .iter()
                .any(|mate| ss_under(&prop_above, &recv_above, mate));
            if admissible {
                // Δ_min: smallest retained (distinguished) gap across all agents.
                let mut dmin = f64::INFINITY;
                for (p, above) in prop_above.iter().enumerate() {
                    for x in 0..n {
                        for y in 0..n {
                            if above[x][y] {
                                dmin = dmin.min(prop[p][x] - prop[p][y]);
                            }
                        }
                    }
                }
                for (a, above) in recv_above.iter().enumerate() {
                    for x in 0..n {
                        for y in 0..n {
                            if above[x][y] {
                                dmin = dmin.min(recv[a][x] - recv[a][y]);
                            }
                        }
                    }
                }
                if dmin.is_finite() {
                    best = best.max(dmin);
                }
            }

            // Increment the mixed-radix counter.
            let mut i = 0;
            loop {
                if i == radices.len() {
                    return best;
                }
                idx[i] += 1;
                if idx[i] < radices[i] {
                    break;
                }
                idx[i] = 0;
                i += 1;
            }
        }
    }
}
