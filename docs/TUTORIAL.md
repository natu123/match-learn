# match-learn tutorial

A guided tour from a textbook stable matching to a learned ride-hailing surge
price. Every snippet uses only the public API; run `cargo test --doc` to see the
shorter ones checked.

```toml
[dependencies]
match-learn = "0.1"
```

## 1. Stable matching

The foundation is Gale-Shapley deferred acceptance. Preferences are ranked
lists; the result is the proposer-optimal stable matching.

```rust
use match_learn::gale_shapley;

let proposers = vec![vec![0, 1, 2], vec![1, 0, 2], vec![0, 1, 2]];
let receivers = vec![vec![1, 0, 2], vec![0, 1, 2], vec![0, 1, 2]];
let m = gale_shapley(&proposers, &receivers);
println!("{:?}", m.proposer); // each proposer's partner, or None
```

Lists may be partial (anyone omitted is unacceptable), and the two sides may
differ in size. Two more mechanisms come built in:

- `hospital_residents(proposers, receivers, capacities)` — many-to-one with
  quotas.
- `top_trading_cycle(prefs)` — one-sided allocation by trading (the housing
  market), yielding the unique core allocation.

Gale-Shapley is *proposer-optimal* — best for one side, worst for the other. The
`fairness` module measures that imbalance (rank cost per side) and finds the
`egalitarian_stable` (min total cost) and `sex_equal_stable` (min imbalance)
matchings instead.

## 2. Learning preferences online

Real markets don't hand you preferences — agents discover them by interacting.
A `Market` has learning proposers (whose true utilities are unknown to them) and
known receiver preferences. Each round it ranks from current beliefs, matches,
observes noisy rewards, and updates.

```rust
use match_learn::{Market, simulate};

let util = vec![
    vec![1.0, 0.4, 0.1],
    vec![0.2, 1.0, 0.5],
    vec![0.1, 0.3, 1.0],
];
let receiver_prefs = vec![vec![0, 1, 2], vec![1, 0, 2], vec![2, 1, 0]];

// (prior_mean, prior_var, obs_var, noise, seed). Setting obs_var = noise^2 is
// well-specified inference and converges fastest.
let mut market = Market::with_thompson(util, receiver_prefs, 0.5, 1.0, 0.04, 0.2, 42);
let report = simulate(&mut market, 3000);

println!("total regret   = {:.2}", report.total_regret());
println!("tail stability = {:.2}", report.tail_stable_fraction(600));
```

`Report` exposes the per-round `cumulative_regret` and `stable` series plus
`total_regret`, `tail_stable_fraction`, `tail_mean_regret`, and `settled_round`.

### Choosing a learner

`Market::with_ucb(...)` uses UCB1 instead of Thompson Sampling. The learners
also stand alone (`GaussianThompson`, `Ucb1`, `DiscountedThompson` for
non-stationary preferences, `LinearThompson` for contextual features); any type
implementing `PreferenceLearner` plugs into `Market::new`.

### Both sides unknown

`TwoSidedMarket` makes receivers learn too — both sides are bandits, matched on
their belief-rankings each round. It reaches the same stable matching, just
noisier early on.

## 3. Evaluating

`simulate` works on anything implementing `LearningMarket`, returning a
`Report`. For sweeps, `simulate_batch(n_markets, rounds, threads, factory)` runs
many markets across threads (and falls back to sequential on `wasm32`), with
results bit-for-bit identical to running them one by one.

The repository's `tests/gate.rs` shows the acceptance bar: across random
markets, regret is sublinear (`R(2T)/R(T)` well under 2) and the matching
stabilizes.

## 4. From matching to a market: dynamic pricing

Matching becomes *market clearing* once a price decides who participates. A
`Marketplace` has price-responsive demand and supply arriving into queues:

```rust
use match_learn::marketplace::{Demand, Marketplace, Supply};

let demand = Demand { base: 12.0, max_price: 20.0 };
let supply = Supply { base: 12.0, ref_price: 10.0 };
let mut market = Marketplace::new(demand, supply, 0.02 /* abandonment */, 7);

let p = market.clearing_price();        // closed form
let outcome = market.step(p);           // arrivals, matches, queues, revenue
```

Too cheap floods the demand queue; too dear idles supply; the clearing price
balances them and maximizes matched volume.

### Learning the price

When the response curves are unknown, learn the price with the same bandits:

```rust
use match_learn::pricing::{LearnedPricer, Objective, price_grid};

let grid = price_grid(1.0, 18.0, 18);
let mut pricer = LearnedPricer::with_ucb(grid, 0.7, Objective::Throughput);
for _ in 0..5000 {
    pricer.step(&mut market);
}
println!("learned price ~ {:.2}", pricer.best_price());
```

The `regret_queue` example sweeps the exploration constant to show the
regret-vs-queue-imbalance tradeoff.

## 5. Joint pricing × matching

With heterogeneous agents, a price gates *participation* and Gale-Shapley
matches the entrants by preference — the two halves of the library meeting:

```rust
use match_learn::{random_joint_instance, Rng};

let mut rng = Rng::new(1);
let market = random_joint_instance(&mut rng, 20, 20);
let matched = market.matched_at(0.5);   // entrants matched at this price
let welfare = market.welfare_at(0.5);   // gains from trade
```

Raising the price thins demand but thickens supply, so matched volume peaks at
an interior price — which a bandit learns over a stream of markets.

## 5b. Online (dynamic) matching

Real platforms are dynamic: agents arrive and leave over time. `OnlineMarket`
models this on the plane, and a `Policy` decides *when* to match — greedily every
tick, or batched to accumulate a richer pool.

```rust
use match_learn::online::{OnlineMarket, Policy};

let mut market = OnlineMarket::new(3.0 /* arrivals/tick */, 0.04 /* abandon */, 7);
let stats = market.run(10_000, Policy::Batched(8));
println!("matched {}, abandoned {}, mean distance {:.3}",
    stats.matched, stats.abandoned, stats.mean_distance());
```

Longer batching pairs closer partners but abandons more waiting agents — and a
bandit can learn the net-value-maximizing interval online (see the
`online_matching` example and the `online` module tests).

## 6. Applications

`RideHailing` and `Delivery` map concrete platforms onto `JointInstance`: agents
on a plane, proximity preferences, and a surge fee / delivery fee gating
participation. The `ride_hailing` example learns the surge fare that maximizes
completed rides.

```rust
use match_learn::{random_ride_hailing, Rng};

let mut rng = Rng::new(1);
let market = random_ride_hailing(&mut rng, 20, 20);
let rides = market.rides_at(0.5);
```

## 7. Beyond Rust

- **Python**: build with maturin (`maturin develop --release`, optional `python`
  feature) and `import match_learn` — see `python/example.py`.
- **WASM**: the core compiles to `wasm32-unknown-unknown`.
- **Benchmarks**: `bench/` compares against the Python `matching` library and
  MABWiser.

## Where next

See the Roadmap in the README. The frontier is the learning × matching interface
— better exploration that provably avoids the greedy-Thompson stall, and the
dynamic-pricing market design that the Phase 7–8 modules begin.
