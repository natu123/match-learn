//! Double auctions: clearing a market from reported values and costs (Phase 8).
//!
//! Posted-price markets (the [`marketplace`](crate::marketplace) and
//! [`pricing`](crate::pricing) modules) set one price and let participation
//! follow. A *double auction* instead collects buyers' bids and sellers' asks
//! and computes both the quantity traded and the price — the market-design view
//! of "price as preference".
//!
//! Two mechanisms:
//! - [`double_auction`] is **efficient**: it trades every pair whose buyer value
//!   exceeds the seller cost, maximizing gains from trade, at a uniform price.
//!   (Efficient double auctions are not strategy-proof.)
//! - [`mcafee_auction`] is **truthful** (dominant-strategy incentive
//!   compatible), individually rational, and weakly budget-balanced, at the cost
//!   of at most one foregone trade — the classic McAfee mechanism.

/// Outcome of a double auction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuctionResult {
    /// Number of buyer-seller pairs that trade.
    pub quantity: usize,
    /// Price each trading buyer pays.
    pub buyer_price: f64,
    /// Price each trading seller receives.
    pub seller_price: f64,
    /// Gains from trade actually realized (sum of `value - cost` over trades).
    pub welfare: f64,
}

/// Buyer values sorted descending and seller costs sorted ascending.
fn sorted(values: &[f64], costs: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut b = values.to_vec();
    b.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    let mut s = costs.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    (b, s)
}

/// The efficient trading quantity: the largest `k` with the `k`-th highest bid
/// at least the `k`-th lowest ask.
pub fn efficient_quantity(values: &[f64], costs: &[f64]) -> usize {
    let (b, s) = sorted(values, costs);
    let mut k = 0;
    while k < b.len().min(s.len()) && b[k] >= s[k] {
        k += 1;
    }
    k
}

/// Realized welfare if the top `q` buyers trade with the bottom `q` sellers.
fn welfare_of(b: &[f64], s: &[f64], q: usize) -> f64 {
    (0..q).map(|i| b[i] - s[i]).sum()
}

/// Efficient uniform-price double auction.
///
/// Trades the `efficient_quantity` pairs at a single price in the marginal
/// pair's `[cost, value]` interval (its midpoint), so every trade is individually
/// rational and total welfare is maximized.
pub fn double_auction(values: &[f64], costs: &[f64]) -> AuctionResult {
    let (b, s) = sorted(values, costs);
    let k = {
        let mut k = 0;
        while k < b.len().min(s.len()) && b[k] >= s[k] {
            k += 1;
        }
        k
    };
    if k == 0 {
        return AuctionResult {
            quantity: 0,
            buyer_price: 0.0,
            seller_price: 0.0,
            welfare: 0.0,
        };
    }
    let price = 0.5 * (s[k - 1] + b[k - 1]);
    AuctionResult {
        quantity: k,
        buyer_price: price,
        seller_price: price,
        welfare: welfare_of(&b, &s, k),
    }
}

/// McAfee's truthful double auction.
///
/// Dominant-strategy incentive compatible, individually rational, and weakly
/// budget-balanced. When the midpoint price of the first *excluded* pair lies in
/// the marginal trading pair's interval, all `k` efficient trades clear at that
/// price (budget balanced); otherwise one trade is sacrificed and buyers pay the
/// marginal bid while sellers receive the marginal ask.
pub fn mcafee_auction(values: &[f64], costs: &[f64]) -> AuctionResult {
    let (b, s) = sorted(values, costs);
    let k = {
        let mut k = 0;
        while k < b.len().min(s.len()) && b[k] >= s[k] {
            k += 1;
        }
        k
    };
    if k == 0 {
        return AuctionResult {
            quantity: 0,
            buyer_price: 0.0,
            seller_price: 0.0,
            welfare: 0.0,
        };
    }

    // Midpoint of the first excluded pair, when one exists.
    if k < b.len() && k < s.len() {
        let p0 = 0.5 * (b[k] + s[k]);
        if s[k - 1] <= p0 && p0 <= b[k - 1] {
            // All k efficient trades clear at p0 (budget balanced).
            return AuctionResult {
                quantity: k,
                buyer_price: p0,
                seller_price: p0,
                welfare: welfare_of(&b, &s, k),
            };
        }
    }

    // Sacrifice the marginal trade: k-1 trades, buyers pay b[k-1], sellers get
    // s[k-1]. (Truthful; weakly budget-balanced since b[k-1] >= s[k-1].)
    let q = k - 1;
    AuctionResult {
        quantity: q,
        buyer_price: b[k - 1],
        seller_price: s[k - 1],
        welfare: welfare_of(&b, &s, q),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn efficient_quantity_is_correct() {
        // bids 9,7,5,3 ; asks 2,4,6,8. Pairs: 9>=2, 7>=4, 5<6 -> stop. q = 2.
        let values = vec![5.0, 9.0, 3.0, 7.0];
        let costs = vec![6.0, 2.0, 8.0, 4.0];
        assert_eq!(efficient_quantity(&values, &costs), 2);
    }

    #[test]
    fn double_auction_is_individually_rational_and_efficient() {
        let mut rng = Rng::new(1);
        for _ in 0..500 {
            let n = 1 + rng.below(8);
            let m = 1 + rng.below(8);
            let values: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
            let costs: Vec<f64> = (0..m).map(|_| rng.uniform()).collect();

            let r = double_auction(&values, &costs);

            // Every trade is individually rational: the marginal buyer's value
            // is at least the price, the marginal seller's cost at most the price.
            let (b, s) = sorted(&values, &costs);
            if r.quantity > 0 {
                assert!(b[r.quantity - 1] >= r.buyer_price - 1e-12);
                assert!(s[r.quantity - 1] <= r.seller_price + 1e-12);
            }
            // No quantity yields more welfare than the efficient one.
            for q in 0..=n.min(m) {
                assert!(r.welfare >= welfare_of(&b, &s, q) - 1e-9);
            }
        }
    }

    #[test]
    fn mcafee_is_individually_rational_and_budget_balanced() {
        let mut rng = Rng::new(7);
        for _ in 0..500 {
            let n = 1 + rng.below(8);
            let m = 1 + rng.below(8);
            let values: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
            let costs: Vec<f64> = (0..m).map(|_| rng.uniform()).collect();

            let r = mcafee_auction(&values, &costs);
            let (b, s) = sorted(&values, &costs);

            // Weakly budget balanced: buyers pay at least what sellers receive.
            assert!(r.buyer_price >= r.seller_price - 1e-12);
            // Individually rational and at most the efficient quantity.
            assert!(r.quantity <= efficient_quantity(&values, &costs));
            if r.quantity > 0 {
                assert!(b[r.quantity - 1] >= r.buyer_price - 1e-12);
                assert!(s[r.quantity - 1] <= r.seller_price + 1e-12);
            }
        }
    }

    #[test]
    fn mcafee_is_truthful_for_buyers() {
        // A trading buyer cannot improve its utility by misreporting its value.
        let mut rng = Rng::new(20260606);
        for _ in 0..300 {
            let n = 2 + rng.below(6);
            let m = 2 + rng.below(6);
            let values: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
            let costs: Vec<f64> = (0..m).map(|_| rng.uniform()).collect();

            // Utility of buyer 0 (value values[0]) under truthful reporting.
            let truthful_util = buyer_utility(&values, &costs, 0, values[0]);

            // Try a range of misreports.
            for step in 0..10 {
                let lie = step as f64 / 9.0;
                let dev = buyer_utility(&values, &costs, 0, lie);
                assert!(
                    dev <= truthful_util + 1e-9,
                    "buyer gained by lying: truthful={truthful_util}, lie@{lie}={dev}"
                );
            }
        }
    }

    /// Utility of buyer `i` (true value `true_value`) when it reports `report`:
    /// `true_value - buyer_price` if it ends up trading, else 0. Whether it trades
    /// is decided by re-running the auction with the reported value, and checking
    /// if `i` is among the top-`quantity` bids.
    fn buyer_utility(values: &[f64], costs: &[f64], i: usize, report: f64) -> f64 {
        let mut reported = values.to_vec();
        reported[i] = report;
        let r = mcafee_auction(&reported, costs);
        if r.quantity == 0 {
            return 0.0;
        }
        // Buyer i trades iff its reported value is among the top `quantity` bids
        // (ties broken so that exactly `quantity` buyers trade).
        let mut idx: Vec<usize> = (0..reported.len()).collect();
        idx.sort_by(|&a, &b| {
            reported[b]
                .partial_cmp(&reported[a])
                .unwrap()
                .then(a.cmp(&b))
        });
        let trades = idx[..r.quantity].contains(&i);
        if trades {
            values[i] - r.buyer_price
        } else {
            0.0
        }
    }
}
