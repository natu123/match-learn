//! Top Trading Cycles (TTC) for the housing market.
//!
//! `n` agents each own one object (agent `a` is endowed with object `a`) and
//! have strict, complete preferences over all objects. TTC produces the unique
//! **core** allocation: it is Pareto efficient, individually rational, and
//! strategy-proof. It is a different mechanism from Gale-Shapley — one-sided
//! allocation by trading rather than two-sided matching — and seeds the
//! recommendation/exchange flavour of matching the project will build on.
//!
//! Algorithm: every remaining agent points at the owner of its most-preferred
//! remaining object. With out-degree one everywhere, the pointer graph always
//! contains a cycle; every agent in a cycle receives the object it points to,
//! and is then removed. Repeat until everyone is assigned.

/// Index of agent `a`'s most-preferred object that is still available.
fn top_available(prefs_a: &[usize], available: &[bool]) -> usize {
    *prefs_a
        .iter()
        .find(|&&o| available[o])
        .expect("an available object exists while agents remain")
}

/// Compute the TTC allocation for the housing market with identity endowment
/// (agent `a` owns object `a`).
///
/// `prefs[a]` is agent `a`'s strict ranking over all `n` objects (a permutation
/// of `0..n`). Returns `assignment` where `assignment[a]` is the object agent
/// `a` ends up with; the result is always a permutation.
pub fn top_trading_cycle(prefs: &[Vec<usize>]) -> Vec<usize> {
    let n = prefs.len();
    let mut assignment = vec![usize::MAX; n];
    let mut available = vec![true; n]; // object o still unallocated
    let mut remaining = n;

    while remaining > 0 {
        // Find a cycle by following pointers from any unassigned agent. Each
        // agent points to the owner of its top remaining object; with identity
        // endowment that owner is the object's own index.
        let start = (0..n)
            .find(|&a| assignment[a] == usize::MAX)
            .expect("an unassigned agent exists");

        let mut path = Vec::new();
        let mut on_path = vec![false; n];
        let mut cur = start;
        while !on_path[cur] {
            on_path[cur] = true;
            path.push(cur);
            cur = top_available(&prefs[cur], &available); // next agent (= object index)
        }
        // `cur` is where the path closed: the cycle is path[pos..].
        let pos = path
            .iter()
            .position(|&x| x == cur)
            .expect("cycle node on path");

        for &a in &path[pos..] {
            let obj = top_available(&prefs[a], &available);
            assignment[a] = obj;
            available[obj] = false;
            remaining -= 1;
        }
    }

    assignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn is_permutation(v: &[usize]) -> bool {
        let mut s = v.to_vec();
        s.sort_unstable();
        s == (0..v.len()).collect::<Vec<_>>()
    }

    /// Rank of object `o` in `prefs_a` (lower = more preferred).
    fn rank(prefs_a: &[usize], o: usize) -> usize {
        prefs_a.iter().position(|&x| x == o).unwrap()
    }

    #[test]
    fn everyone_keeps_their_own_when_it_is_their_favourite() {
        // Each agent prefers its own object first -> the identity allocation.
        let prefs = vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]];
        assert_eq!(top_trading_cycle(&prefs), vec![0, 1, 2]);
    }

    #[test]
    fn a_two_cycle_swaps() {
        // Agent 0 wants object 1, agent 1 wants object 0 -> they trade.
        let prefs = vec![vec![1, 0], vec![0, 1]];
        assert_eq!(top_trading_cycle(&prefs), vec![1, 0]);
    }

    #[test]
    fn result_is_a_permutation() {
        let mut rng = Rng::new(0x77C);
        for _ in 0..500 {
            let n = 1 + rng.below(6);
            let prefs: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let a = top_trading_cycle(&prefs);
            assert!(is_permutation(&a), "not a permutation: {a:?} for {prefs:?}");
        }
    }

    #[test]
    fn allocation_is_individually_rational() {
        // Every agent ends up with an object at least as good as its endowment.
        let mut rng = Rng::new(0x1234);
        for _ in 0..500 {
            let n = 1 + rng.below(6);
            let prefs: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let a = top_trading_cycle(&prefs);
            for (agent, p) in prefs.iter().enumerate() {
                assert!(
                    rank(p, a[agent]) <= rank(p, agent),
                    "agent {agent} did worse than its endowment"
                );
            }
        }
    }

    /// Whether `other` Pareto-dominates `base`: nobody worse, somebody better.
    fn pareto_dominates(prefs: &[Vec<usize>], other: &[usize], base: &[usize]) -> bool {
        let mut strictly_better = false;
        for (a, p) in prefs.iter().enumerate() {
            let ro = rank(p, other[a]);
            let rb = rank(p, base[a]);
            if ro > rb {
                return false; // someone is worse off
            }
            if ro < rb {
                strictly_better = true;
            }
        }
        strictly_better
    }

    #[test]
    fn allocation_is_pareto_efficient() {
        // Brute force: no permutation Pareto-dominates the TTC allocation.
        let mut rng = Rng::new(0xEFF);
        for _ in 0..200 {
            let n = 1 + rng.below(5); // up to 5! = 120 permutations
            let prefs: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let ttc = top_trading_cycle(&prefs);

            // Enumerate all permutations of 0..n via Heap-free lexicographic gen.
            let mut perm: Vec<usize> = (0..n).collect();
            loop {
                if pareto_dominates(&prefs, &perm, &ttc) {
                    panic!("TTC not Pareto-efficient: {perm:?} dominates {ttc:?} for {prefs:?}");
                }
                if !next_permutation(&mut perm) {
                    break;
                }
            }
        }
    }

    /// In-place next lexicographic permutation; false when the last is reached.
    fn next_permutation(a: &mut [usize]) -> bool {
        if a.len() < 2 {
            return false;
        }
        let mut i = a.len() - 1;
        while i > 0 && a[i - 1] >= a[i] {
            i -= 1;
        }
        if i == 0 {
            return false;
        }
        let mut j = a.len() - 1;
        while a[j] <= a[i - 1] {
            j -= 1;
        }
        a.swap(i - 1, j);
        a[i..].reverse();
        true
    }
}
