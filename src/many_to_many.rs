//! Many-to-many stable matching: both sides hold multiple partners.
//!
//! A generalization of [one-to-one Gale-Shapley](crate::matching) and the
//! [many-to-one](crate::many_to_one) Hospital-Residents problem in which *both*
//! sides have quotas: each worker may hold up to `worker_quota[w]` firms and each
//! firm up to `firm_quota[f]` workers. This is the shape of labor markets where a
//! worker takes several part-time jobs and a firm hires several workers, of
//! supplier-buyer networks, and of course-allocation.
//!
//! Worker-proposing deferred acceptance with *responsive* (quota-based)
//! preferences yields a **pairwise-stable**, worker-optimal matching. With both
//! quotas equal to 1 it reduces exactly to one-to-one Gale-Shapley; with the firm
//! quota free and the worker quota 1 it reduces to Hospital-Residents.

use crate::matching::rank_table;

/// A many-to-many matching: two consistent views of the same pair set.
///
/// `worker[w]` lists the firms holding worker `w` (at most `worker_quota[w]`);
/// `firm[f]` lists the workers at firm `f` (at most `firm_quota[f]`). `f` appears
/// in `worker[w]` iff `w` appears in `firm[f]`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ManyToMany {
    /// Firms held by each worker.
    pub worker: Vec<Vec<usize>>,
    /// Workers held by each firm.
    pub firm: Vec<Vec<usize>>,
}

impl ManyToMany {
    /// Total number of matched (worker, firm) pairs.
    pub fn pairs(&self) -> usize {
        self.worker.iter().map(|fs| fs.len()).sum()
    }

    /// Sort every inner list ascending, for order-independent comparison.
    pub fn normalized(mut self) -> Self {
        for fs in &mut self.worker {
            fs.sort_unstable();
        }
        for ws in &mut self.firm {
            ws.sort_unstable();
        }
        self
    }
}

/// Worker-optimal, pairwise-stable many-to-many matching.
///
/// `worker_prefs[w]` ranks firms; `firm_prefs[f]` ranks workers; `worker_quota`
/// and `firm_quota` give each side's capacities. A pair is acceptable only if
/// each appears on the other's list. Runs in `O(W * F)` proposals.
///
/// ```
/// use match_learn::many_to_many;
///
/// // Two workers, two firms, each wanting up to two partners and all mutually
/// // acceptable: everyone matches everyone.
/// let workers = vec![vec![0, 1], vec![0, 1]];
/// let firms = vec![vec![0, 1], vec![0, 1]];
/// let m = many_to_many(&workers, &firms, &[2, 2], &[2, 2]);
/// assert_eq!(m.pairs(), 4);
/// ```
pub fn many_to_many(
    worker_prefs: &[Vec<usize>],
    firm_prefs: &[Vec<usize>],
    worker_quota: &[usize],
    firm_quota: &[usize],
) -> ManyToMany {
    let n_w = worker_prefs.len();
    let n_f = firm_prefs.len();
    assert_eq!(worker_quota.len(), n_w, "one quota per worker");
    assert_eq!(firm_quota.len(), n_f, "one quota per firm");
    let firm_rank = rank_table(firm_prefs, n_w);

    let mut worker: Vec<Vec<usize>> = vec![Vec::new(); n_w];
    let mut firm: Vec<Vec<usize>> = vec![Vec::new(); n_f];
    let mut next = vec![0usize; n_w]; // next firm on `w`'s list to propose to
    let mut free: Vec<usize> = (0..n_w).collect();

    while let Some(w) = free.pop() {
        // Propose while the worker has spare quota and unproposed firms remain.
        while worker[w].len() < worker_quota[w] && next[w] < worker_prefs[w].len() {
            let f = worker_prefs[w][next[w]];
            next[w] += 1;
            if f >= n_f || firm_rank[f][w].is_none() {
                continue; // out of range or `f` finds `w` unacceptable
            }
            // Tentatively accept the offer.
            firm[f].push(w);
            worker[w].push(f);
            if firm[f].len() <= firm_quota[f] {
                continue; // within quota; the pair holds
            }
            // Over quota: `f` rejects its worst-ranked held worker.
            let (worst_pos, _) = firm[f]
                .iter()
                .enumerate()
                .max_by_key(|(_, q)| firm_rank[f][**q].expect("held pair is acceptable"))
                .expect("firm is non-empty when over quota");
            let worst = firm[f].remove(worst_pos);
            worker[worst].retain(|&g| g != f);
            if worst != w && !free.contains(&worst) {
                free.push(worst); // a bumped worker re-enters to refill its quota
            }
        }
    }

    ManyToMany { worker, firm }
}

/// Whether `m` is pairwise-stable for the many-to-many instance.
///
/// A blocking pair `(w, f)` is mutually acceptable and *not* currently matched,
/// where each side would take the other: `w` has a free slot or prefers `f` to
/// its worst current firm, and symmetrically for `f`. Under responsive
/// preferences pairwise stability characterizes the core.
pub fn is_pairwise_stable(
    worker_prefs: &[Vec<usize>],
    firm_prefs: &[Vec<usize>],
    worker_quota: &[usize],
    firm_quota: &[usize],
    m: &ManyToMany,
) -> bool {
    let n_w = worker_prefs.len();
    let n_f = firm_prefs.len();
    let worker_rank = rank_table(worker_prefs, n_f);
    let firm_rank = rank_table(firm_prefs, n_w);

    for w in 0..n_w {
        for &f in &worker_prefs[w] {
            if f >= n_f || firm_rank[f][w].is_none() {
                continue; // not mutually acceptable
            }
            if m.worker[w].contains(&f) {
                continue; // already matched, cannot block
            }
            // Does `w` want `f`?
            let w_wants = if m.worker[w].len() < worker_quota[w] {
                true
            } else {
                let f_rank = worker_rank[w][f].expect("acceptable");
                let worst = m.worker[w]
                    .iter()
                    .map(|&g| worker_rank[w][g].expect("held pair is acceptable"))
                    .max()
                    .expect("held is non-empty when at quota");
                f_rank < worst
            };
            if !w_wants {
                continue;
            }
            // Does `f` want `w`?
            let f_wants = if m.firm[f].len() < firm_quota[f] {
                true
            } else {
                let w_rank = firm_rank[f][w].expect("acceptable");
                let worst = m.firm[f]
                    .iter()
                    .map(|&x| firm_rank[f][x].expect("held pair is acceptable"))
                    .max()
                    .expect("held is non-empty when at quota");
                w_rank < worst
            };
            if f_wants {
                return false; // (w, f) blocks
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn both_quotas_one_recovers_gale_shapley() {
        let workers = vec![vec![0, 1], vec![0, 1]];
        let firms = vec![vec![0, 1], vec![0, 1]];
        let m = many_to_many(&workers, &firms, &[1, 1], &[1, 1]);
        let gs = crate::matching::gale_shapley(&workers, &firms);
        for w in 0..2 {
            let got = m.worker[w].first().copied();
            assert_eq!(got, gs.proposer[w]);
        }
        assert!(is_pairwise_stable(&workers, &firms, &[1, 1], &[1, 1], &m));
    }

    #[test]
    fn worker_quota_one_recovers_hospital_residents() {
        // Worker quota 1, firm quota free: the many-to-many DA must reproduce the
        // Hospital-Residents matching (workers = residents, firms = hospitals).
        let mut rng = Rng::new(0x4D324D);
        for _ in 0..200 {
            let n_w = 1 + rng.below(4);
            let n_f = 1 + rng.below(3);
            let workers: Vec<Vec<usize>> = (0..n_w).map(|_| rng.permutation(n_f)).collect();
            let firms: Vec<Vec<usize>> = (0..n_f).map(|_| rng.permutation(n_w)).collect();
            let fcap: Vec<usize> = (0..n_f).map(|_| 1 + rng.below(2)).collect();
            let wq = vec![1usize; n_w];
            let m = many_to_many(&workers, &firms, &wq, &fcap);
            let hr = crate::many_to_one::hospital_residents(&workers, &firms, &fcap);
            for w in 0..n_w {
                assert_eq!(m.worker[w].first().copied(), hr.proposer[w]);
            }
        }
    }

    #[test]
    fn saturates_when_everyone_acceptable() {
        // 3 workers, 2 firms, big quotas: every acceptable pair matches.
        let workers = vec![vec![0, 1], vec![0, 1], vec![0, 1]];
        let firms = vec![vec![0, 1, 2], vec![0, 1, 2]];
        let m = many_to_many(&workers, &firms, &[2, 2, 2], &[3, 3]);
        assert_eq!(m.pairs(), 6);
        assert!(is_pairwise_stable(
            &workers,
            &firms,
            &[2, 2, 2],
            &[3, 3],
            &m
        ));
    }

    /// Brute force: enumerate every quota-respecting matching over the worker x
    /// firm grid and keep the pairwise-stable ones.
    fn all_pairwise_stable(
        workers: &[Vec<usize>],
        firms: &[Vec<usize>],
        wq: &[usize],
        fq: &[usize],
    ) -> Vec<ManyToMany> {
        let n_w = workers.len();
        let n_f = firms.len();
        let bits = n_w * n_f;
        let mut out = Vec::new();
        for mask in 0u32..(1u32 << bits) {
            let mut worker = vec![Vec::new(); n_w];
            let mut firm = vec![Vec::new(); n_f];
            for bit in 0..bits {
                if mask & (1 << bit) != 0 {
                    let (w, f) = (bit / n_f, bit % n_f);
                    worker[w].push(f);
                    firm[f].push(w);
                }
            }
            let ok = worker.iter().enumerate().all(|(w, fs)| fs.len() <= wq[w])
                && firm.iter().enumerate().all(|(f, ws)| ws.len() <= fq[f]);
            if ok {
                let m = ManyToMany { worker, firm };
                if is_pairwise_stable(workers, firms, wq, fq, &m) {
                    out.push(m);
                }
            }
        }
        out
    }

    #[test]
    fn matches_brute_force_oracle_on_random_instances() {
        let mut rng = Rng::new(0x4D324D32);
        for _ in 0..300 {
            let n_w = 1 + rng.below(3); // 1..=3
            let n_f = 1 + rng.below(3);
            let workers: Vec<Vec<usize>> = (0..n_w).map(|_| rng.permutation(n_f)).collect();
            let firms: Vec<Vec<usize>> = (0..n_f).map(|_| rng.permutation(n_w)).collect();
            let wq: Vec<usize> = (0..n_w).map(|_| 1 + rng.below(2)).collect();
            let fq: Vec<usize> = (0..n_f).map(|_| 1 + rng.below(2)).collect();
            let m = many_to_many(&workers, &firms, &wq, &fq);
            assert!(
                is_pairwise_stable(&workers, &firms, &wq, &fq, &m),
                "produced an unstable matching: w={workers:?} f={firms:?} wq={wq:?} fq={fq:?}"
            );
            let want = m.normalized();
            let oracle = all_pairwise_stable(&workers, &firms, &wq, &fq);
            assert!(
                oracle.into_iter().any(|o| o.normalized() == want),
                "matching not in oracle set: w={workers:?} f={firms:?} wq={wq:?} fq={fq:?}"
            );
        }
    }
}
