"""Train the Early Churn Alert models on the sample and save the results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    "d1_online_minutes": "D1 online minutes",
    "activities_touched_d1_d3": "Activities touched by D3",
    "passed_quest_break_d1": "Passed day-one quest break",
    "active_days_d1_d3": "Active days D1-D3",
    "level_d3": "Level on D3",
    "joined_guild_d3": "Joined a guild by D3",
    "checkin_claims_d1_d3": "Check-in claims D1-D3",
    "paid_d3": "Paid by D3",
    "from_paid_ua": "Came from paid UA",
}


def top_share(y, p, q=0.2):
    th = np.quantile(p, 1 - q)
    pred = (p >= th).astype(int)
    return precision_score(y, pred), recall_score(y, pred)


def main():
    df = pd.read_csv(ROOT / "data" / "new_players_sample.csv")
    X, y = df[list(FEATURES)].values, df["churn_by_d7"].values
    ids = df["player_id"].values
    Xtr, Xte, ytr, yte, idtr, idte = train_test_split(X, y, ids, test_size=0.3, random_state=1, stratify=y)

    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=1).fit(Xtr, ytr)
    p_dt = dt.predict_proba(Xte)[:, 1]

    prec, rec = top_share(yte, p_lr)
    q50, q80 = np.quantile(p_lr, [0.5, 0.8])
    tier = np.where(p_lr >= q80, "High", np.where(p_lr >= q50, "Medium", "Low"))
    tiers = {t: {"share": float((tier == t).mean()), "churn_by_d7": float(yte[tier == t].mean())}
             for t in ["Low", "Medium", "High"]}
    res = {
        "players_total": int(len(df)), "players_train": int(len(ytr)), "players_test": int(len(yte)),
        "churn_rate": float(y.mean()),
        "logistic_regression": {"auc": float(roc_auc_score(yte, p_lr)),
                                "precision_top20": float(prec), "recall_top20": float(rec),
                                "standardised_coefficients": dict(zip(FEATURES.values(), map(float, lr.coef_[0])))},
        "decision_tree_depth4": {"auc": float(roc_auc_score(yte, p_dt))},
        "risk_tiers_test_set": tiers,
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    json.dump(res, open(out / "model_results.json", "w"), indent=2)
    pd.DataFrame({"player_id": idte, "churn_score": p_lr.round(4), "risk_tier": tier, "churn_by_d7": yte}) \
        .to_csv(out / "test_scores.csv", index=False)

    # two-level tree used for the rule table in the report
    dt2 = DecisionTreeClassifier(max_depth=2, min_samples_leaf=300, random_state=1).fit(Xtr, ytr)
    rules = export_text(dt2, feature_names=list(FEATURES.values()), show_weights=True)
    (out / "decision_tree_rules.txt").write_text(rules)

    # coefficient chart (Figure 1.1 in the report)
    items = sorted(res["logistic_regression"]["standardised_coefficients"].items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=200)
    ax.barh([k for k, v in items], [v for k, v in items], height=0.55,
            color=["#2a78d6" if v < 0 else "#eb6834" for k, v in items])
    for i, (k, v) in enumerate(items):
        ax.text(v + (-0.02 if v < 0 else 0.02), i, f"{v:.2f}", va="center", ha="right" if v < 0 else "left", fontsize=8.5)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("Standardised coefficient (negative = lower churn risk)")
    ax.set_xlim(-0.85, 0.3)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(color="#2a78d6", label="lowers churn risk"), Patch(color="#eb6834", label="raises churn risk")],
              frameon=False, loc="lower right", fontsize=8.5)
    (ROOT / "figures").mkdir(exist_ok=True)
    plt.savefig(ROOT / "figures" / "coefficients.png", bbox_inches="tight")

    print(f"Train / test: {len(ytr):,} / {len(yte):,} players, churn rate {y.mean():.1%}")
    print(f"Logistic regression AUC: {res['logistic_regression']['auc']:.3f}")
    print(f"Decision tree (depth 4) AUC: {res['decision_tree_depth4']['auc']:.3f}")
    print(f"Top 20% risk group: precision {prec:.1%}, recall {rec:.1%}")
    for t, v in tiers.items():
        print(f"  {t:<6} tier: {v['share']:.0%} of players, churn by D7 {v['churn_by_d7']:.1%}")
    print("Two-level decision tree rules:\n" + rules)


if __name__ == "__main__":
    main()
