# KTO Live Ops talent report: code and results

This repository holds the code behind my talent report "Extending the Live Ops Plan for New Players of an MMORPG at VNGGames: Early Churn Alert, A/B Testing and KPI Dashboard" (University of Economics and Law, Faculty of Information Systems, 2026).

The report extends my internship at VNGGames, where our team designed a Live Ops plan to improve the 30-day retention of new players in Kiếm Thế Origin (KTO). Since I cannot use the real player data of KTO, every number here comes from a sample of 20,000 new players created in Python. The sample follows the patterns our team found during the internship, so the results only show how each module works and should not be read as real results of the game.

## What is inside

| Path | Content |
|---|---|
| `src/generate_data.py` | Creates the sample of 20,000 new players (fixed seed 2026) |
| `src/train_model.py` | Trains logistic regression and a decision tree for the Early Churn Alert, saves scores, risk tiers, tree rules and the coefficient chart |
| `src/sample_size.py` | Players needed per group for the A/B tests (alpha 0.05, power 0.8, assumed baselines) |
| `src/dashboard_mockup.py` | Builds the KPI dashboard mock-up from the sample and checks the alert rules |
| `results/` | Model results, tree rules, sample sizes, dashboard values and the full run log |
| `dashboard/dashboard.html` | The dashboard mock-up as a web page |

## How to run

```bash
pip install -r requirements.txt
bash run_all.sh
```

The seeds are fixed, so the output in `results/run_log.txt` should be the same on every run. Running the scripts also creates `data/new_players_sample.csv` (the 20,000 players), `results/test_scores.csv` (the churn score and risk tier of each test player) and the two charts in `figures/`. These generated files are not stored here because the scripts recreate exactly the same files.

## Main results (sample data)

| Measure | Value |
|---|---|
| Churn by D7 in the sample | 36.5% |
| AUC, logistic regression | 0.891 |
| AUC, decision tree (depth 4) | 0.874 |
| Top 20% risk group: precision / recall | 83.9% / 45.9% |
| Churn by D7: Low / Medium / High tier | 7.4% / 53.6% / 83.9% |
| Players per group, RR7 20% (assumed) +2 pp | 6,510 |

## Assumptions to keep in mind

- The churn label is drawn from a logistic formula written in `generate_data.py`. Because the sample follows this formula, the AUC is higher than what real data would give.
- The A/B scenario in the dashboard assumes dynamic guidance helps 10% of the test players who missed the day-one quest break. With such a small effect, the difference is not significant, which is exactly why the sample size has to be planned first.
- The incident on server S184 (guild join chance halved on day 14) is added on purpose so that the alert rule has something to catch.
