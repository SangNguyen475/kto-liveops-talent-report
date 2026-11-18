"""Players needed per group for the A/B tests (two proportions, alpha 0.05, power 0.8).

The real baselines of KTO are confidential, so the baselines below are assumptions.
Replace them with the real values before running a test.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
CASES = [  # (metric, assumed baseline, smallest change to detect)
    ("RR7 (Test 1)", 0.20, 0.02),
    ("RR7 (Test 1)", 0.20, 0.03),
    ("RR30 of High and Medium tiers (Test 2)", 0.08, 0.02),
]


def n_per_group(p1, delta, alpha=0.05, power=0.8):
    p2 = p1 + delta
    pbar = (p1 + p2) / 2
    a = norm.ppf(1 - alpha / 2) * np.sqrt(2 * pbar * (1 - pbar))
    b = norm.ppf(power) * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return int(np.ceil((a + b) ** 2 / delta ** 2))


def main():
    rows = [{"metric": m, "baseline_assumed": p, "min_change": d, "players_per_group": n_per_group(p, d)} for m, p, d in CASES]
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "sample_size.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
