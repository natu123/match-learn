//! Independent, exponential reference for tiny valid preference profiles.
//! Uses the original tiers, not production rank tables, validators or enumerators.
//! Reuse via `#[path = "support/tiny_matching_oracle.rs"] mod oracle;`.

use match_learn::matching::Matching;

pub type Profile = Vec<Vec<Vec<usize>>>;

#[derive(Clone, Copy, Debug)]
pub enum Stability {
    Weak,
    Strong,
    Super,
}

fn rank(tiers: &[Vec<usize>], partner: usize) -> Option<usize> {
    tiers.iter().position(|tier| tier.contains(&partner))
}

/// Recursively enumerate only injective, mutually acceptable assignments.
pub fn feasible_matchings(p: &Profile, r: &Profile) -> Vec<Matching> {
    fn visit(p: &Profile, r: &Profile, m: &mut Matching, out: &mut Vec<Matching>, i: usize) {
        if i == p.len() {
            out.push(m.clone());
            return;
        }
        visit(p, r, m, out, i + 1);
        for j in 0..r.len() {
            if m.receiver[j].is_none() && rank(&p[i], j).is_some() && rank(&r[j], i).is_some() {
                m.proposer[i] = Some(j);
                m.receiver[j] = Some(i);
                visit(p, r, m, out, i + 1);
                m.proposer[i] = None;
                m.receiver[j] = None;
            }
        }
    }
    let mut out = Vec::new();
    let mut m = Matching {
        proposer: vec![None; p.len()],
        receiver: vec![None; r.len()],
    };
    visit(p, r, &mut m, &mut out, 0);
    out
}

/// Direct blocker definition; caller supplies a feasible matching.
pub fn has_no_blocker(p: &Profile, r: &Profile, m: &Matching, kind: Stability) -> bool {
    for (i, pi) in p.iter().enumerate() {
        for (j, rj) in r.iter().enumerate() {
            if m.proposer[i] == Some(j) {
                continue;
            }
            let (Some(a), Some(b)) = (rank(pi, j), rank(rj, i)) else {
                continue;
            };
            let current_a = m.proposer[i]
                .and_then(|x| rank(pi, x))
                .unwrap_or(usize::MAX);
            let current_b = m.receiver[j]
                .and_then(|x| rank(rj, x))
                .unwrap_or(usize::MAX);
            let blocks = match kind {
                Stability::Weak => a < current_a && b < current_b,
                Stability::Strong => {
                    (a < current_a && b <= current_b) || (a <= current_a && b < current_b)
                }
                Stability::Super => a <= current_a && b <= current_b,
            };
            if blocks {
                return false;
            }
        }
    }
    true
}

pub fn stable_matchings(p: &Profile, r: &Profile, kind: Stability) -> Vec<Matching> {
    feasible_matchings(p, r)
        .into_iter()
        .filter(|m| has_no_blocker(p, r, m, kind))
        .collect()
}

/// Maximum weak-stable cardinality and, among those matchings, maximum number
/// of retained pairs from `previous`. Unmatched slots never count as retained.
pub fn cardinality_retention(
    p: &Profile,
    r: &Profile,
    previous: &Matching,
) -> Option<(usize, usize)> {
    stable_matchings(p, r, Stability::Weak)
        .iter()
        .map(|m| {
            let size = m.proposer.iter().filter(|x| x.is_some()).count();
            let retained = m
                .proposer
                .iter()
                .enumerate()
                .filter(|&(i, x)| x.is_some() && previous.proposer.get(i) == Some(x))
                .count();
            (size, retained)
        })
        .max()
}
