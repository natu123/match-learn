//! One-sided assignment: allocating indivisible objects to agents *without*
//! endowments (house allocation), the companion to [`ttc`](crate::ttc)'s
//! exchange-with-endowments setting.
//!
//! Three mechanisms over agents with strict preferences and unit-supply objects:
//!
//! - [`serial_dictatorship`] — agents pick their favourite remaining object in a
//!   fixed priority order. Pareto efficient and strategy-proof, but the priority
//!   order is decisive (the first agent always gets its top choice).
//! - [`random_serial_dictatorship`] — average serial dictatorship over random
//!   orders (the "random priority" mechanism), giving each agent a *fractional*
//!   assignment that is equal-treatment-of-equals fair.
//! - [`probabilistic_serial`] — the Bogomolnaia-Moulin simultaneous-eating
//!   algorithm: every agent "eats" its favourite available object at unit rate
//!   until supplies run out. Its fractional assignment is **ordinally efficient**
//!   and **envy-free** ([`sd_envy_free`]), which random priority is not.
//!
//! [`is_pareto_efficient`] checks Pareto efficiency of a discrete assignment.

/// Position of object `o` in `prefs_a` (0 = best), or `None` if unacceptable.
fn rank_of(prefs_a: &[usize], o: usize) -> Option<usize> {
    prefs_a.iter().position(|&x| x == o)
}

/// Serial dictatorship: agents take turns (in `order`) claiming their most
/// preferred still-available object.
///
/// `prefs[a]` ranks the objects acceptable to agent `a`; `order` is a priority
/// permutation of the agents; `n_objects` is the number of unit-supply objects.
/// Returns `assignment[a] = Some(o)` for the object agent `a` receives, or `None`
/// if every object it ranks was already taken. Always Pareto efficient.
///
/// ```
/// use match_learn::serial_dictatorship;
///
/// // Both agents want object 0 first; the priority order 0-then-1 decides.
/// let prefs = vec![vec![0, 1], vec![0, 1]];
/// let a = serial_dictatorship(&prefs, 2, &[0, 1]);
/// assert_eq!(a, vec![Some(0), Some(1)]);
/// ```
pub fn serial_dictatorship(
    prefs: &[Vec<usize>],
    n_objects: usize,
    order: &[usize],
) -> Vec<Option<usize>> {
    let mut taken = vec![false; n_objects];
    let mut assignment = vec![None; prefs.len()];
    for &a in order {
        if let Some(&o) = prefs[a].iter().find(|&&o| o < n_objects && !taken[o]) {
            taken[o] = true;
            assignment[a] = Some(o);
        }
    }
    assignment
}

/// Random serial dictatorship ("random priority"): the fractional assignment
/// obtained by averaging [`serial_dictatorship`] over `samples` random priority
/// orders.
///
/// `result[a][o]` is the probability agent `a` receives object `o`. With enough
/// samples this approximates the exact random-priority distribution; it treats
/// equals equally but is *not* ordinally efficient (see [`probabilistic_serial`]).
pub fn random_serial_dictatorship(
    prefs: &[Vec<usize>],
    n_objects: usize,
    samples: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let n = prefs.len();
    let mut rng = crate::rng::Rng::new(seed);
    let mut counts = vec![vec![0.0f64; n_objects]; n];
    for _ in 0..samples {
        let order = rng.permutation(n);
        let assignment = serial_dictatorship(prefs, n_objects, &order);
        for (a, slot) in assignment.iter().enumerate() {
            if let Some(o) = slot {
                counts[a][*o] += 1.0;
            }
        }
    }
    let denom = samples.max(1) as f64;
    for row in &mut counts {
        for c in row {
            *c /= denom;
        }
    }
    counts
}

/// Probabilistic serial (simultaneous eating) fractional assignment.
///
/// Time runs over `[0, 1]`. Every agent eats its most-preferred not-yet-exhausted
/// object at unit rate; when an object is fully consumed its eaters move to their
/// next choice. `result[a][o]` is the fraction of object `o` (= the probability)
/// agent `a` consumes. The outcome is ordinally efficient and envy-free in the
/// stochastic-dominance sense ([`sd_envy_free`]).
///
/// ```
/// use match_learn::probabilistic_serial;
///
/// // Two agents with the same ranking split each object 50/50 — the fair share
/// // that a fixed priority order cannot give.
/// let prefs = vec![vec![0, 1], vec![0, 1]];
/// let ps = probabilistic_serial(&prefs, 2);
/// assert!((ps[0][0] - 0.5).abs() < 1e-9 && (ps[1][0] - 0.5).abs() < 1e-9);
/// ```
pub fn probabilistic_serial(prefs: &[Vec<usize>], n_objects: usize) -> Vec<Vec<f64>> {
    const EPS: f64 = 1e-12;
    let n = prefs.len();
    let mut alloc = vec![vec![0.0f64; n_objects]; n];
    let mut remaining = vec![1.0f64; n_objects]; // supply of each object
    let mut ptr = vec![0usize; n]; // index into prefs[a] of current target
    let mut time_left = vec![1.0f64; n]; // eating budget per agent

    loop {
        // Each active agent's current target: the next acceptable object that
        // still has supply. Advance the pointer past exhausted/unacceptable ones.
        let mut target = vec![None; n];
        let mut count = vec![0.0f64; n_objects];
        for a in 0..n {
            if time_left[a] <= EPS {
                continue;
            }
            while ptr[a] < prefs[a].len() {
                let o = prefs[a][ptr[a]];
                if o < n_objects && remaining[o] > EPS {
                    target[a] = Some(o);
                    count[o] += 1.0;
                    break;
                }
                ptr[a] += 1;
            }
        }

        // Time until the next event: an object depletes, or an agent's budget ends.
        let mut dt = f64::INFINITY;
        for (o, &c) in count.iter().enumerate() {
            if c > 0.0 {
                dt = dt.min(remaining[o] / c);
            }
        }
        for a in 0..n {
            if target[a].is_some() {
                dt = dt.min(time_left[a]);
            }
        }
        if !dt.is_finite() {
            break; // no active agent with an available object
        }

        for a in 0..n {
            if let Some(o) = target[a] {
                alloc[a][o] += dt;
                time_left[a] -= dt;
            }
        }
        for (o, &c) in count.iter().enumerate() {
            if c > 0.0 {
                remaining[o] -= dt * c;
            }
        }
    }
    alloc
}

/// Whether a fractional assignment is envy-free in the stochastic-dominance
/// sense: no agent would prefer another agent's lottery to its own.
///
/// For every ordered pair of agents `(i, j)` and every prefix of `i`'s ranking,
/// `i` must assign at least as much total probability to that prefix under its
/// own row as under `j`'s. Probabilistic serial always satisfies this.
pub fn sd_envy_free(prefs: &[Vec<usize>], alloc: &[Vec<f64>]) -> bool {
    const EPS: f64 = 1e-9;
    let n = prefs.len();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let (mut mine, mut theirs) = (0.0, 0.0);
            for &o in &prefs[i] {
                mine += alloc[i][o];
                theirs += alloc[j][o];
                if mine + EPS < theirs {
                    return false; // i envies j over this prefix
                }
            }
        }
    }
    true
}

/// Whether a discrete `assignment` is Pareto efficient under strict preferences.
///
/// Inefficiency takes exactly two forms: an agent strictly prefers some *free*
/// (unassigned) object to what it holds, or a cycle of agents each strictly
/// prefers the next one's object — rotating it makes everyone in the cycle
/// better off. Absent both, no Pareto improvement exists.
pub fn is_pareto_efficient(
    prefs: &[Vec<usize>],
    n_objects: usize,
    assignment: &[Option<usize>],
) -> bool {
    let n = prefs.len();
    // `cur_rank[a]` = rank of a's object (or "worse than anything" if unmatched).
    let cur_rank = |a: usize| -> usize {
        match assignment[a] {
            Some(o) => rank_of(&prefs[a], o).unwrap_or(usize::MAX),
            None => usize::MAX,
        }
    };
    let prefers =
        |a: usize, o: usize| -> bool { rank_of(&prefs[a], o).is_some_and(|r| r < cur_rank(a)) };

    // 1. A free object someone strictly prefers is an immediate improvement.
    let mut assigned = vec![false; n_objects];
    for o in assignment.iter().flatten() {
        assigned[*o] = true;
    }
    for a in 0..n {
        for (o, &taken) in assigned.iter().enumerate() {
            if !taken && prefers(a, o) {
                return false;
            }
        }
    }

    // 2. A trading cycle a -> b (a wants b's object) means everyone can improve.
    let mut adj = vec![Vec::new(); n];
    for (a, neighbours) in adj.iter_mut().enumerate() {
        for (b, slot) in assignment.iter().enumerate() {
            if a == b {
                continue;
            }
            if let Some(ob) = slot
                && prefers(a, *ob)
            {
                neighbours.push(b);
            }
        }
    }
    !has_cycle(&adj)
}

/// Directed-cycle detection by three-colour DFS.
fn has_cycle(adj: &[Vec<usize>]) -> bool {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        White,
        Gray,
        Black,
    }
    let n = adj.len();
    let mut mark = vec![Mark::White; n];
    // Iterative DFS to avoid recursion limits on large graphs.
    for start in 0..n {
        if mark[start] != Mark::White {
            continue;
        }
        let mut stack = vec![(start, 0usize)];
        mark[start] = Mark::Gray;
        while let Some(&mut (node, ref mut idx)) = stack.last_mut() {
            if *idx < adj[node].len() {
                let next = adj[node][*idx];
                *idx += 1;
                match mark[next] {
                    Mark::Gray => return true, // back edge -> cycle
                    Mark::White => {
                        mark[next] = Mark::Gray;
                        stack.push((next, 0));
                    }
                    Mark::Black => {}
                }
            } else {
                mark[node] = Mark::Black;
                stack.pop();
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn first_in_order_gets_top_choice() {
        let prefs = vec![vec![1, 0, 2], vec![1, 2, 0], vec![0, 1, 2]];
        let a = serial_dictatorship(&prefs, 3, &[2, 0, 1]);
        assert_eq!(a[2], Some(0)); // agent 2 picks first, takes its favourite 0
        assert_eq!(a[0], Some(1)); // agent 0 next, takes 1
        assert_eq!(a[1], Some(2)); // agent 1 last, 1 gone -> takes 2
    }

    #[test]
    fn serial_dictatorship_is_always_pareto_efficient() {
        let mut rng = Rng::new(0x5D);
        for _ in 0..2000 {
            let n = 1 + rng.below(5);
            let prefs: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let order = rng.permutation(n);
            let a = serial_dictatorship(&prefs, n, &order);
            assert!(
                is_pareto_efficient(&prefs, n, &a),
                "SD inefficient: prefs={prefs:?} order={order:?} a={a:?}"
            );
        }
    }

    /// Brute-force oracle: is any other full assignment a Pareto improvement?
    fn brute_pareto_efficient(
        prefs: &[Vec<usize>],
        n_objects: usize,
        assignment: &[Option<usize>],
    ) -> bool {
        let n = prefs.len();
        let cur_rank = |a: usize| match assignment[a] {
            Some(o) => rank_of(&prefs[a], o).unwrap_or(usize::MAX),
            None => usize::MAX,
        };
        // Enumerate every injective assignment of objects (or none) to agents.
        let mut choice = vec![0usize; n]; // 0..n_objects = object, n_objects = unassigned
        loop {
            let mut used = vec![false; n_objects];
            let mut valid = true;
            for &c in &choice {
                if c < n_objects {
                    if used[c] {
                        valid = false;
                        break;
                    }
                    used[c] = true;
                }
            }
            if valid {
                let mut nobody_worse = true;
                let mut someone_better = false;
                for a in 0..n {
                    let r = if choice[a] < n_objects {
                        rank_of(&prefs[a], choice[a]).unwrap_or(usize::MAX)
                    } else {
                        usize::MAX
                    };
                    if r > cur_rank(a) {
                        nobody_worse = false;
                        break;
                    }
                    if r < cur_rank(a) {
                        someone_better = true;
                    }
                }
                if nobody_worse && someone_better {
                    return false; // a Pareto improvement exists
                }
            }
            // mixed-radix increment, base (n_objects + 1)
            let mut i = 0;
            loop {
                if i == n {
                    return true;
                }
                choice[i] += 1;
                if choice[i] <= n_objects {
                    break;
                }
                choice[i] = 0;
                i += 1;
            }
        }
    }

    #[test]
    fn pareto_check_matches_brute_force() {
        let mut rng = Rng::new(0x9A12);
        for _ in 0..1500 {
            let n = 1 + rng.below(4);
            let prefs: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            // Test on an arbitrary (often inefficient) assignment, not just SD.
            let assignment: Vec<Option<usize>> =
                rng.permutation(n).iter().map(|&o| Some(o)).collect();
            assert_eq!(
                is_pareto_efficient(&prefs, n, &assignment),
                brute_pareto_efficient(&prefs, n, &assignment),
                "mismatch: prefs={prefs:?} a={assignment:?}"
            );
        }
    }

    #[test]
    fn identical_preferences_split_evenly() {
        // n agents all ranking 0,1,2,..: probabilistic serial gives each a 1/n
        // share of every object.
        let prefs = vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]];
        let ps = probabilistic_serial(&prefs, 3);
        for row in &ps {
            for &x in row {
                assert!((x - 1.0 / 3.0).abs() < 1e-9, "expected even split, got {x}");
            }
        }
    }

    #[test]
    fn probabilistic_serial_is_a_bistochastic_assignment() {
        let mut rng = Rng::new(0xB5);
        for _ in 0..500 {
            let n = 1 + rng.below(5);
            let prefs: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let ps = probabilistic_serial(&prefs, n);
            // Each agent eats a total of 1; each object is fully consumed.
            for row in &ps {
                let s: f64 = row.iter().sum();
                assert!((s - 1.0).abs() < 1e-9, "row sum {s} != 1");
            }
            for o in 0..n {
                let s: f64 = ps.iter().map(|row| row[o]).sum();
                assert!((s - 1.0).abs() < 1e-9, "column sum {s} != 1");
            }
        }
    }

    #[test]
    fn probabilistic_serial_is_envy_free() {
        let mut rng = Rng::new(0xEF1);
        for _ in 0..500 {
            let n = 1 + rng.below(5);
            let prefs: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let ps = probabilistic_serial(&prefs, n);
            assert!(
                sd_envy_free(&prefs, &ps),
                "PS not envy-free: prefs={prefs:?}"
            );
        }
    }

    #[test]
    fn random_serial_dictatorship_rows_are_distributions() {
        let prefs = vec![vec![0, 1, 2], vec![1, 0, 2], vec![2, 1, 0]];
        let rsd = random_serial_dictatorship(&prefs, 3, 600, 42);
        for row in &rsd {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "row sum {s} != 1");
        }
    }
}
