//! Additivity of the cascade floor over **vertex-disjoint** near-tie swings
//! (`docs/theory-identifiability.md`, Prop. 2″). Tiles `k` independent copies of
//! the 4×4 net-floor block (`net_floor_4x4.rs`) into one market whose blocks
//! share no proposer or receiver. Cross-block receivers are unacceptable, so
//! Gale-Shapley decomposes over the connected components and the per-block floors
//! simply add: net floor `= k · 1.20 = Θ(k)`.
//!
//! It also shows independence directly: mis-ordering only block `j` adds exactly
//! one block's floor (`1.20`), leaving the others at zero.
//!
//! ```text
//! cargo run --release --example multipair_floor
//! ```

use match_learn::gale_shapley;
use match_learn::matching::Matching;

const DP: f64 = 0.01;

/// (full utility matrix, proposer rankings, receiver rankings) for a tiled market.
type TiledMarket = (Vec<Vec<f64>>, Vec<Vec<usize>>, Vec<Vec<usize>>);

/// The 4×4 net-floor block: proposer utilities p,q,r,s over receivers A,B,C,D.
fn block_util() -> Vec<Vec<f64>> {
    vec![
        vec![1.00, 1.00 - DP, 0.10, 0.00], // p
        vec![0.00, 0.90, 0.40, 0.05],      // q
        vec![0.00, 0.10, 0.80, 0.50],      // r
        vec![0.30, 0.20, 0.10, 0.70],      // s
    ]
}

/// Receiver preferences within a block (over local proposers p,q,r,s).
fn block_recv() -> Vec<Vec<usize>> {
    vec![
        vec![0, 1, 2, 3], // A
        vec![0, 1, 2, 3], // B
        vec![1, 2, 0, 3], // C
        vec![2, 3, 0, 1], // D
    ]
}

/// Local true ranking per proposer (descending by block utility).
fn block_true_order() -> Vec<Vec<usize>> {
    vec![
        vec![0, 1, 2, 3], // p: A B C D
        vec![1, 2, 3, 0], // q: B C D A
        vec![2, 3, 1, 0], // r: C D B A
        vec![3, 0, 1, 2], // s: D A B C
    ]
}

/// Build a `k`-block tiled market: global index of (block b, local i) is `4b+i`.
/// `mis_blocks[b] == true` means block b's proposer p reports B before A.
/// Returns (full utility matrix, proposer rankings, receiver rankings).
fn tiled(k: usize, mis_blocks: &[bool]) -> TiledMarket {
    let n = 4 * k;
    let bu = block_util();
    let br = block_recv();
    let bt = block_true_order();

    let mut util = vec![vec![0.0; n]; n]; // cross-block utility 0
    let mut prop_rank = vec![Vec::new(); n];
    let mut recv_rank = vec![Vec::new(); n];

    for (b, &mis) in mis_blocks.iter().enumerate() {
        let off = 4 * b;
        for p in 0..4 {
            for r in 0..4 {
                util[off + p][off + r] = bu[p][r];
            }
            // proposer ranking lists only this block's receivers (others unacceptable)
            let order = if p == 0 && mis {
                vec![1, 0, 2, 3] // p mis-orders A,B
            } else {
                bt[p].clone()
            };
            prop_rank[off + p] = order.iter().map(|&x| off + x).collect();
        }
        for r in 0..4 {
            recv_rank[off + r] = br[r].iter().map(|&x| off + x).collect();
        }
    }
    (util, prop_rank, recv_rank)
}

fn net_regret(util: &[Vec<f64>], star: &Matching, m: &Matching) -> f64 {
    (0..util.len())
        .map(|p| {
            let b = star.proposer[p].map_or(0.0, |r| util[p][r]);
            let g = m.proposer[p].map_or(0.0, |r| util[p][r]);
            b - g
        })
        .sum()
}

fn main() {
    println!("Multi-pair floor additivity (vertex-disjoint 4×4 blocks, Δ_p={DP})\n");
    println!("  k   all-mis net   (expected k·1.21)   only-block-0 net   (expected 1.21)");
    for k in 1..=4 {
        // M* of the whole tiled market: every block in its true order.
        let (util, true_rank, recv) = tiled(k, &vec![false; k]);
        let star = gale_shapley(&true_rank, &recv);

        // All blocks mis-order simultaneously.
        let (_, all_mis_rank, _) = tiled(k, &vec![true; k]);
        let all_mis = gale_shapley(&all_mis_rank, &recv);
        let net_all = net_regret(&util, &star, &all_mis);

        // Only block 0 mis-orders.
        let mut one = vec![false; k];
        one[0] = true;
        let (_, one_rank, _) = tiled(k, &one);
        let one_mis = gale_shapley(&one_rank, &recv);
        let net_one = net_regret(&util, &star, &one_mis);

        println!(
            "  {k}   {net_all:>8.3}        {:>8.3}          {net_one:>8.3}            {:>8.3}",
            k as f64 * 1.21,
            1.21,
        );
        assert!(
            (net_all - k as f64 * 1.21).abs() < 1e-9,
            "all-mis net floor should be k·1.21"
        );
        assert!(
            (net_one - 1.21).abs() < 1e-9,
            "single-block mis should add exactly one block's floor"
        );
    }
    println!(
        "\nGale-Shapley decomposes over the blocks (cross-block receivers unacceptable),\n\
         so each independent near-tie swing contributes its own Θ(1) floor: the net\n\
         floor is additive, k·1.20 = Θ(k). Overlapping (vertex-sharing) swings are\n\
         instance-dependent and not covered — see Prop. 2″ remark."
    );
}
