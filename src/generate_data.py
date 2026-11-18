"""Generate a sample of 20,000 new players for the Early Churn Alert.

The real player data of Kiem The Origin (KTO) is confidential, so this script
creates a synthetic sample that follows the patterns found during the internship:
- many players drop at a break in the main quest chain on day one,
- players who join a guild early stay much longer,
- players from paid UA have a slightly higher churn risk.
The random seed is fixed, so running the script again gives the same file.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 2026
N = 20000
OUT = Path(__file__).resolve().parents[1] / "data" / "new_players_sample.csv"


def churn_probability(d1_min, passed, acts, days, guild, lvl, checkin, paid3, paid_ua):
    """Probability of churn by D7 used to create the label (also reused by the dashboard scenario)."""
    z = (2.2 - 0.018 * d1_min - 1.1 * passed - 0.08 * acts - 0.55 * (days - 1) - 1.35 * guild
         - 0.025 * (lvl - 30) - 0.15 * checkin - 0.8 * paid3 + 0.35 * paid_ua)
    return 1 / (1 + np.exp(-z))


def main():
    rng = np.random.default_rng(SEED)
    paid_ua = rng.binomial(1, 0.55, N)
    d1_min = np.clip(rng.gamma(2.2, 28, N) - paid_ua * 8, 3, 300)
    passed = rng.binomial(1, 1 / (1 + np.exp(-(d1_min - 45) / 15)))
    acts = np.clip(np.round(rng.poisson(3 + d1_min / 25) + passed * 3), 0, 22)
    days = np.clip(1 + rng.binomial(2, np.clip(0.2 + d1_min / 250 + passed * 0.2, 0, 0.95)), 1, 3)
    guild = rng.binomial(1, np.clip(0.04 + 0.012 * acts + 0.1 * (days == 3), 0, 0.9))
    lvl = np.clip(np.round(18 + passed * 14 + days * 4 + acts * 0.6 + rng.normal(0, 4, N)), 8, 59)
    checkin = np.minimum(days, rng.binomial(3, 0.8, N))
    paid3 = rng.binomial(1, np.clip(0.02 + 0.03 * guild + 0.01 * days, 0, 1))
    # churn by D7 = no login from D4 to D7, drawn from a logistic function of the behaviour above
    churn = rng.binomial(1, churn_probability(d1_min, passed, acts, days, guild, lvl, checkin, paid3, paid_ua))
    df = pd.DataFrame({
        "player_id": [f"P{i:05d}" for i in range(1, N + 1)],
        "d1_online_minutes": d1_min.round(1),
        "activities_touched_d1_d3": acts.astype(int),
        "passed_quest_break_d1": passed,
        "active_days_d1_d3": days.astype(int),
        "level_d3": lvl.astype(int),
        "joined_guild_d3": guild,
        "checkin_claims_d1_d3": checkin.astype(int),
        "paid_d3": paid3,
        "from_paid_ua": paid_ua,
        "churn_by_d7": churn,
    })
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} players to {OUT.name}")
    print(f"Churn by D7: {df.churn_by_d7.mean():.1%}")
    print(f"Passed the day-one quest break: {df.passed_quest_break_d1.mean():.1%}")
    print(f"Joined a guild by D3: {df.joined_guild_d3.mean():.1%}")


if __name__ == "__main__":
    main()
