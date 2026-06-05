//! Market data: realistic generators and a text format for preferences.
//!
//! Uniform-random utilities (used in the gate) are a worst case for *agreement*:
//! every agent ranks the other side independently. Real two-sided markets are
//! more *correlated* — there are popular schools, sought-after hospitals,
//! desirable partners — so preferences mix a shared "common value" with private
//! taste. [`correlated_market`] generates exactly that, tunable from fully
//! private (uniform) to fully common (everyone agrees).
//!
//! [`to_text`] / [`from_text`] (de)serialize strict preference lists in a small
//! line-based format, so instances can be saved, shared, or read from external
//! datasets.

use crate::rng::Rng;

/// Generate a two-sided market mixing common and private values.
///
/// Each receiver `r` has a common quality `q_r ~ U(0,1)` that every proposer
/// values, blended with private taste: `util_p[p][r] = w*q_r + (1-w)*e_pr`,
/// where `e_pr ~ U(0,1)` and `w = common_weight in [0,1]`. Symmetrically for the
/// receiver side with its own common qualities. `w = 0` is fully private
/// (uniform, independent rankings); `w = 1` is fully common (everyone agrees).
///
/// Returns `(util_p, util_r)` with `util_p` shaped `[n_p][n_r]` and `util_r`
/// shaped `[n_r][n_p]`.
pub fn correlated_market(
    rng: &mut Rng,
    n_p: usize,
    n_r: usize,
    common_weight: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    assert!(
        (0.0..=1.0).contains(&common_weight),
        "common_weight must be in [0, 1]"
    );
    let w = common_weight;

    // Common qualities for each side.
    let q_receivers: Vec<f64> = (0..n_r).map(|_| rng.uniform()).collect();
    let q_proposers: Vec<f64> = (0..n_p).map(|_| rng.uniform()).collect();

    let util_p: Vec<Vec<f64>> = (0..n_p)
        .map(|_| {
            (0..n_r)
                .map(|r| w * q_receivers[r] + (1.0 - w) * rng.uniform())
                .collect()
        })
        .collect();
    let util_r: Vec<Vec<f64>> = (0..n_r)
        .map(|_| {
            (0..n_p)
                .map(|p| w * q_proposers[p] + (1.0 - w) * rng.uniform())
                .collect()
        })
        .collect();

    (util_p, util_r)
}

/// Strict preference rankings (descending) implied by a utility matrix.
pub fn prefs_from_util(util: &[Vec<f64>]) -> Vec<Vec<usize>> {
    util.iter()
        .map(|row| crate::prefs::rank_by_scores(row))
        .collect()
}

/// Serialize a two-sided instance to text.
///
/// Format: a header line `P R`, then `P` proposer preference lines, then `R`
/// receiver preference lines; each line is the space-separated partner indices,
/// most preferred first. Lines may be incomplete (unacceptable partners
/// omitted).
pub fn to_text(proposer_prefs: &[Vec<usize>], receiver_prefs: &[Vec<usize>]) -> String {
    let mut out = format!("{} {}\n", proposer_prefs.len(), receiver_prefs.len());
    for list in proposer_prefs {
        let line: Vec<String> = list.iter().map(|x| x.to_string()).collect();
        out.push_str(&line.join(" "));
        out.push('\n');
    }
    for list in receiver_prefs {
        let line: Vec<String> = list.iter().map(|x| x.to_string()).collect();
        out.push_str(&line.join(" "));
        out.push('\n');
    }
    out
}

/// Parse the text format written by [`to_text`].
///
/// Blank lines are treated as empty preference lists. Returns
/// `(proposer_prefs, receiver_prefs)`. Panics on a malformed header or wrong
/// line count.
pub fn from_text(text: &str) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut lines = text.lines();
    let header = lines.next().expect("missing header line");
    let mut it = header.split_whitespace();
    let n_p: usize = it
        .next()
        .expect("missing P")
        .parse()
        .expect("P not a number");
    let n_r: usize = it
        .next()
        .expect("missing R")
        .parse()
        .expect("R not a number");

    let parse_line = |line: &str| -> Vec<usize> {
        line.split_whitespace()
            .map(|t| t.parse().expect("preference entry not a number"))
            .collect()
    };

    let proposer_prefs: Vec<Vec<usize>> = (0..n_p)
        .map(|_| parse_line(lines.next().expect("missing a proposer line")))
        .collect();
    let receiver_prefs: Vec<Vec<usize>> = (0..n_r)
        .map(|_| parse_line(lines.next().expect("missing a receiver line")))
        .collect();

    (proposer_prefs, receiver_prefs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How often two proposers share their single most-preferred receiver, over
    /// all pairs — a simple agreement measure.
    fn top_agreement(prefs: &[Vec<usize>]) -> f64 {
        let mut same = 0;
        let mut total = 0;
        for i in 0..prefs.len() {
            for j in (i + 1)..prefs.len() {
                total += 1;
                if prefs[i][0] == prefs[j][0] {
                    same += 1;
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            same as f64 / total as f64
        }
    }

    #[test]
    fn higher_common_weight_means_more_agreement() {
        let mut rng = Rng::new(2026);
        let private = {
            let (up, _) = correlated_market(&mut rng, 12, 8, 0.0);
            top_agreement(&prefs_from_util(&up))
        };
        let common = {
            let (up, _) = correlated_market(&mut rng, 12, 8, 0.95);
            top_agreement(&prefs_from_util(&up))
        };
        assert!(
            common > private,
            "common-value agreement {common} should exceed private {private}"
        );
        assert!(
            common > 0.5,
            "near-common market should mostly agree: {common}"
        );
    }

    #[test]
    fn generated_dimensions_are_correct() {
        let mut rng = Rng::new(1);
        let (up, ur) = correlated_market(&mut rng, 5, 3, 0.5);
        assert_eq!(up.len(), 5);
        assert!(up.iter().all(|r| r.len() == 3));
        assert_eq!(ur.len(), 3);
        assert!(ur.iter().all(|r| r.len() == 5));
    }

    #[test]
    fn text_round_trips() {
        let prop = vec![vec![2, 0, 1], vec![1, 2, 0], vec![0, 1, 2]];
        let recv = vec![vec![0, 1, 2], vec![2, 1, 0], vec![1, 0, 2]];
        let text = to_text(&prop, &recv);
        let (p2, r2) = from_text(&text);
        assert_eq!(p2, prop);
        assert_eq!(r2, recv);
    }

    #[test]
    fn text_handles_incomplete_lists() {
        let prop = vec![vec![1], vec![0, 1]]; // proposer 0 accepts only receiver 1
        let recv = vec![vec![0, 1], vec![1]];
        let text = to_text(&prop, &recv);
        let (p2, r2) = from_text(&text);
        assert_eq!(p2, prop);
        assert_eq!(r2, recv);
    }
}
