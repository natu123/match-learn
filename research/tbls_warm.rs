//! Original conservative external-update adapter. Not author TBLS, no recourse score.
use super::*;

pub struct Previous<'a> {
    pub proposer_profile: &'a Profile,
    pub receiver_profile: &'a Profile,
    pub matching: &'a Matching,
    pub strategy: &'a Strategy,
}

pub struct WarmOutput {
    pub result: Output,
    /// Normalized real profiles for the next call's Previous state.
    pub proposer_profile: Profile,
    pub receiver_profile: Profile,
    pub projected: Matching,
    pub initial_repair: Stats,
    pub discarded_previous_pairs: usize,
    pub padded_size: usize,
    pub calibration_gs_cardinality: usize,
}

fn validate_rows(p: &Profile, r: &Profile) -> Result<(), &'static str> {
    if p.len().max(r.len()) > 64 {
        return Err("adapter side size exceeds 64");
    }
    for (profile, opposite_size) in [(p, r.len()), (r, p.len())] {
        for row in profile {
            let mut seen = vec![false; opposite_size];
            for tier in row {
                if tier.is_empty() {
                    return Err("empty tiers are invalid before normalization");
                }
                for &j in tier {
                    if j >= opposite_size || seen[j] {
                        return Err("invalid preference before normalization");
                    }
                    seen[j] = true;
                }
            }
        }
    }
    Ok(())
}

fn compatible(profile: &Profile, strict: &[Vec<usize>]) -> bool {
    if profile.len() != strict.len() {
        return false;
    }
    profile.iter().zip(strict).all(|(row, order)| {
        if row.iter().map(Vec::len).sum::<usize>() != order.len() {
            return false;
        }
        let mut index = 0;
        for tier in row {
            let mut expected = tier.clone();
            expected.sort_unstable();
            let mut actual = order[index..index + tier.len()].to_vec();
            actual.sort_unstable();
            if actual != expected {
                return false;
            }
            index += tier.len();
        }
        true
    })
}

fn normalize(p: &Profile, r: &Profile) -> (Profile, Profile) {
    let acceptable =
        |profile: &Profile, i: usize, j: usize| profile[i].iter().any(|tier| tier.contains(&j));
    let prune = |profile: &Profile, opposite: &Profile| -> Profile {
        profile
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .filter_map(|tier| {
                        let kept: Vec<_> = tier
                            .iter()
                            .copied()
                            .filter(|&j| acceptable(opposite, j, i))
                            .collect();
                        if kept.is_empty() { None } else { Some(kept) }
                    })
                    .collect()
            })
            .collect()
    };
    (prune(p, r), prune(r, p))
}

fn carried_strategy(profile: &Profile, previous: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
    profile
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let old: &[usize] = previous.get(i).map(Vec::as_slice).unwrap_or(&[]);
            row.iter()
                .flat_map(|tier| {
                    let mut retained: Vec<_> =
                        old.iter().copied().filter(|j| tier.contains(j)).collect();
                    let mut inserted: Vec<_> =
                        tier.iter().copied().filter(|j| !old.contains(j)).collect();
                    rng.shuffle(&mut inserted);
                    retained.extend(inserted);
                    retained
                })
                .collect()
        })
        .collect()
}

fn project(p: &Profile, r: &Profile, old: &Matching) -> Matching {
    let mut m = Matching {
        proposer: vec![None; p.len()],
        receiver: vec![None; r.len()],
    };
    for (i, partner) in old.proposer.iter().enumerate() {
        if let Some(j) = *partner
            && p[i].iter().any(|tier| tier.contains(&j))
            && r[j].iter().any(|tier| tier.contains(&i))
        {
            m.proposer[i] = Some(j);
            m.receiver[j] = Some(i);
        }
    }
    m
}

/// Keep original IDs; arrival appends rows, departure empties a row and incident
/// edges. Shrinking dimensions is rejected to prevent accidental renumbering.
pub fn solve(
    p: &Profile,
    r: &Profile,
    previous: Previous<'_>,
    seed: u64,
    cfg: &Config,
) -> Result<WarmOutput, &'static str> {
    validate_rows(previous.proposer_profile, previous.receiver_profile)?;
    validate_rows(p, r)?;
    if p.len() < previous.proposer_profile.len() || r.len() < previous.receiver_profile.len() {
        return Err("departures must retain inactive ID slots, not truncate dimensions");
    }
    if !compatible(previous.proposer_profile, &previous.strategy.proposer)
        || !compatible(previous.receiver_profile, &previous.strategy.receiver)
        || !is_weakly_stable(
            previous.proposer_profile,
            previous.receiver_profile,
            previous.matching,
        )
        || !is_stable(
            &previous.strategy.proposer,
            &previous.strategy.receiver,
            previous.matching,
        )
    {
        return Err("previous matching/strategy must be coherent, compatible and stable");
    }
    let (mut np, mut nr) = normalize(p, r);
    let projected = project(&np, &nr, previous.matching);
    let discarded_previous_pairs = previous.matching.pairs() - projected.pairs();
    let real_proposers = np.len();
    let real_receivers = nr.len();
    let n = real_proposers.max(real_receivers);
    np.resize(n, vec![]);
    nr.resize(n, vec![]);
    validate(&np, &nr, cfg)?;
    let mut rng = Rng::new(seed);
    let s = Strategy {
        proposer: carried_strategy(&np, &previous.strategy.proposer, &mut rng),
        receiver: carried_strategy(&nr, &previous.strategy.receiver, &mut rng),
    };
    let started = Instant::now();
    let calibration = gale_shapley(&s.proposer, &s.receiver);
    let calibrated_repair_time = started.elapsed();
    let calibration_gs_cardinality = calibration.pairs();
    let budget = cfg
        .repair_budget
        .unwrap_or(RepairBudget::Time(calibrated_repair_time));
    let mut m = projected.clone();
    m.proposer.resize(n, None);
    m.receiver.resize(n, None);
    let mut initial_repair = Stats::default();
    // Conservative adaptation: arbitrary external edits invalidate the subset
    // precondition. Scan every padded agent before using the static search.
    repair(
        &s,
        &mut m,
        (0..2 * n).collect(),
        budget,
        &mut rng,
        &mut initial_repair,
    );
    if !is_stable(&s.proposer, &s.receiver, &m) || !is_weakly_stable(&np, &nr, &m) {
        return Err("warm initial repair failed stability guard");
    }
    let mut result = search(&np, &nr, cfg, m, s, rng, calibrated_repair_time)?;
    result.matching.proposer.truncate(real_proposers);
    result.matching.receiver.truncate(real_receivers);
    result.initial.proposer.truncate(real_proposers);
    result.initial.receiver.truncate(real_receivers);
    result.strategy.proposer.truncate(real_proposers);
    result.strategy.receiver.truncate(real_receivers);
    np.truncate(real_proposers);
    nr.truncate(real_receivers);
    if !is_weakly_stable(p, r, &result.matching)
        || !is_stable(
            &result.strategy.proposer,
            &result.strategy.receiver,
            &result.matching,
        )
    {
        return Err("trimmed result failed original-profile stability guard");
    }
    Ok(WarmOutput {
        result,
        proposer_profile: np,
        receiver_profile: nr,
        projected,
        initial_repair,
        discarded_previous_pairs,
        padded_size: n,
        calibration_gs_cardinality,
    })
}
