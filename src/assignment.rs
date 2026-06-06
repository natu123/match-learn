//! The assignment problem: welfare-optimal matching.
//!
//! Stable matching ([`gale_shapley`](crate::matching)) asks for a matching with
//! no blocking pair. The *assignment problem* asks a different question: the
//! matching that maximizes total weight (or minimizes total cost). It is the
//! welfare-optimal counterpart — what a central planner who can ignore stability
//! would choose — and a useful baseline for how much stability "costs".
//!
//! Solved from scratch with the O(n^3) Hungarian (Kuhn-Munkres) algorithm on a
//! square matrix.

/// Minimum-cost perfect assignment of an `n x n` cost matrix.
///
/// Returns `(assignment, total_cost)` where `assignment[r]` is the column
/// matched to row `r`. O(n^3). The matrix must be square.
pub fn min_cost_assignment(cost: &[Vec<f64>]) -> (Vec<usize>, f64) {
    let n = cost.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    assert!(
        cost.iter().all(|r| r.len() == n),
        "cost matrix must be square"
    );

    const INF: f64 = f64::INFINITY;
    // 1-indexed potentials and assignment (standard Hungarian formulation).
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = row assigned to column j
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![INF; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = INF;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // Augment along the path recorded in `way`.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![0usize; n];
    for j in 1..=n {
        assignment[p[j] - 1] = j - 1;
    }
    let total = (0..n).map(|r| cost[r][assignment[r]]).sum();
    (assignment, total)
}

/// Maximum-weight perfect assignment of an `n x n` weight matrix.
///
/// Returns `(assignment, total_weight)`. Solved by minimizing the negated
/// weights.
pub fn max_weight_assignment(weight: &[Vec<f64>]) -> (Vec<usize>, f64) {
    let neg: Vec<Vec<f64>> = weight
        .iter()
        .map(|r| r.iter().map(|&w| -w).collect())
        .collect();
    let (assignment, neg_total) = min_cost_assignment(&neg);
    (assignment, -neg_total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Brute-force optimum over all permutations (small `n`).
    fn brute_force_min(cost: &[Vec<f64>]) -> f64 {
        let n = cost.len();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut best = f64::INFINITY;
        loop {
            let total: f64 = (0..n).map(|r| cost[r][perm[r]]).sum();
            best = best.min(total);
            if !next_permutation(&mut perm) {
                break;
            }
        }
        best
    }

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

    #[test]
    fn small_known_instance() {
        // Optimal assignment: row0->col1 (1), row1->col0 (2), row2->col2 (3) = 6;
        // or the diagonal 2+4+3... the brute force decides. Just check correctness.
        let cost = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let (assignment, total) = min_cost_assignment(&cost);
        // It is a permutation.
        let mut seen = assignment.clone();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2]);
        assert!((total - brute_force_min(&cost)).abs() < 1e-9);
    }

    #[test]
    fn matches_brute_force_on_random_instances() {
        let mut rng = Rng::new(2026);
        for _ in 0..400 {
            let n = 1 + rng.below(6); // 1..=6
            let cost: Vec<Vec<f64>> = (0..n)
                .map(|_| (0..n).map(|_| rng.uniform()).collect())
                .collect();
            let (assignment, total) = min_cost_assignment(&cost);
            // Valid permutation.
            let mut seen = assignment.clone();
            seen.sort_unstable();
            assert_eq!(seen, (0..n).collect::<Vec<_>>());
            // Total matches the recomputed sum and the brute-force optimum.
            let recomputed: f64 = (0..n).map(|r| cost[r][assignment[r]]).sum();
            assert!((total - recomputed).abs() < 1e-9);
            assert!(
                (total - brute_force_min(&cost)).abs() < 1e-9,
                "Hungarian {total} != brute force {} for {cost:?}",
                brute_force_min(&cost)
            );
        }
    }

    #[test]
    fn max_weight_is_consistent() {
        let mut rng = Rng::new(7);
        for _ in 0..200 {
            let n = 1 + rng.below(5);
            let w: Vec<Vec<f64>> = (0..n)
                .map(|_| (0..n).map(|_| rng.uniform()).collect())
                .collect();
            let (assignment, total) = max_weight_assignment(&w);
            let recomputed: f64 = (0..n).map(|r| w[r][assignment[r]]).sum();
            assert!((total - recomputed).abs() < 1e-9);
            // Max weight = negated min cost of the negated matrix; spot-check it
            // is at least as good as the identity assignment.
            let identity: f64 = (0..n).map(|r| w[r][r]).sum();
            assert!(total >= identity - 1e-9);
        }
    }
}
