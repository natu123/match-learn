"""match-learn from Python.

Build and install the bindings first (inside a virtualenv):

    pip install maturin
    maturin develop --release

Then run:

    python python/example.py
"""

import match_learn as ml


def main():
    # --- Stable matching algorithms ---
    proposer_prefs = [[0, 1, 2], [1, 0, 2], [0, 1, 2]]
    receiver_prefs = [[1, 0, 2], [0, 1, 2], [0, 1, 2]]
    print("Gale-Shapley   :", ml.gale_shapley(proposer_prefs, receiver_prefs))

    # Many-to-one: one hospital with 2 slots, three residents.
    print("Hospital-Resid.:", ml.hospital_residents([[0], [0], [0]], [[0, 1, 2]], [2]))

    # Top Trading Cycles: agents 0 and 1 swap.
    print("Top Trading Cyc:", ml.top_trading_cycle([[1, 0], [0, 1]]))

    # --- Learning market (online preference learning x stable matching) ---
    util_p = [
        [1.0, 0.4, 0.1],
        [0.2, 1.0, 0.5],
        [0.1, 0.3, 1.0],
    ]
    receiver_prefs = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
    market = ml.Market.thompson(util_p, receiver_prefs, noise=0.2, obs_var=0.04, seed=42)

    report = market.simulate(3000)
    print()
    print(report)
    print("total regret          :", round(report.total_regret(), 3))
    print("tail stable fraction  :", round(report.tail_stable_fraction(600), 3))
    print("converged matching    :", market.step())


if __name__ == "__main__":
    main()
