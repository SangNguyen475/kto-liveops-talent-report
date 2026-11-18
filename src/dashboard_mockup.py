"""Build the KPI dashboard mock-up from the sample and evaluate the alert rules.

Everything shown is computed from data/new_players_sample.csv and results/test_scores.csv,
plus two scenarios that are stated clearly below (they are assumptions, not real results):
  1. A/B scenario: dynamic guidance helps 10% of the test-group players who did not pass
     the day-one quest break to pass it. Churn is then drawn again with the same formula
     as in generate_data.py, using the same random numbers for both groups.
  2. Incident scenario: on the last day, a bug in the guild invite flow on server S184
     halves the guild join chance of new players, so the alert rules have something to catch.
"""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from generate_data import churn_probability

ROOT = Path(__file__).resolve().parents[1]
SERVERS = [f"S{n}" for n in range(180, 187)]
BOOST = 0.10           # scenario 1
INCIDENT = ("S184", 14, 0.5)  # scenario 2: server, day, multiplier on guild join


def group_of(pid):
    return "test" if int(hashlib.md5(pid.encode()).hexdigest(), 16) % 2 == 0 else "control"


def main():
    rng = np.random.default_rng(7)
    df = pd.read_csv(ROOT / "data" / "new_players_sample.csv")
    scores = pd.read_csv(ROOT / "results" / "test_scores.csv")
    model = json.load(open(ROOT / "results" / "model_results.json"))
    ss = pd.read_csv(ROOT / "results" / "sample_size.csv")

    # ---- leading metrics (tier 1)
    tiles = {
        "Activities touched by D3 per new player": f"{df.activities_touched_d1_d3.mean():.1f}",
        "Passed day-one quest break": f"{df.passed_quest_break_d1.mean():.0%}",
        "Guild join rate by D3": f"{df.joined_guild_d3.mean():.0%}",
        "High-risk players (churn alert)": f"{(scores.risk_tier == 'High').mean():.0%}",
    }

    # ---- A/B scenario (tier 2)
    df["group"] = df.player_id.map(group_of)
    u = rng.random(len(df))
    passed_new = df.passed_quest_break_d1.values.copy()
    helped = (df.group.values == "test") & (passed_new == 0) & (rng.random(len(df)) < BOOST)
    passed_new[helped] = 1
    p = churn_probability(df.d1_online_minutes, passed_new, df.activities_touched_d1_d3, df.active_days_d1_d3,
                          df.joined_guild_d3, df.level_d3, df.checkin_claims_d1_d3, df.paid_d3, df.from_paid_ua)
    df["churn_ab"] = (u < p).astype(int)
    ab = df.groupby("group").agg(players=("player_id", "size"), rr7=("churn_ab", lambda s: 1 - s.mean()))  # share not churned by D7
    from scipy.stats import norm
    n1, n2 = ab.loc["test", "players"], ab.loc["control", "players"]
    p1, p2 = ab.loc["test", "rr7"], ab.loc["control", "rr7"]
    pp = (p1 * n1 + p2 * n2) / (n1 + n2)
    zval = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    pval = float(2 * (1 - norm.cdf(abs(zval))))
    srm = abs(ab.loc["test", "players"] - ab.loc["control", "players"]) / ab.players.sum()
    tiers = model["risk_tiers_test_set"]

    # ---- daily guild join by server (tier 3 + alerts)
    df["server"] = rng.choice(SERVERS, len(df))
    df["join_day"] = rng.integers(1, 15, len(df))
    joined = df.joined_guild_d3.values.copy()
    s, d, m = INCIDENT
    hit = (df.server == s).values & (df.join_day == d).values & (joined == 1)
    joined[hit & (rng.random(len(df)) > m)] = 0
    df["guild_daily"] = joined
    daily = df.groupby(["server", "join_day"]).guild_daily.mean().unstack()
    alerts = []
    for srv in SERVERS:
        last, avg7 = daily.loc[srv, 14], daily.loc[srv, 7:13].mean()
        if last < 0.7 * avg7:
            alerts.append(("Critical", f"Guild join rate on {srv} dropped {1 - last / avg7:.0%} vs 7-day average", "Live Ops + Community"))
    if srm > 0.02:
        alerts.append(("Critical", f"Sample ratio of Test 1 is off by {srm:.1%}", "Data"))
    need = int(ss.players_per_group.iloc[0])
    if ab.players.min() >= need:
        alerts.append(("OK", f"Sample size of Test 1 reached ({ab.players.min():,} / {need:,} per group)", "Data"))

    summary = {"tiles": tiles, "ab": {g: {"players": int(r.players), "rr7": float(r.rr7)} for g, r in ab.iterrows()},
               "difference_pp": float((p1 - p2) * 100), "p_value": pval, "sample_ratio_gap": float(srm), "alerts": alerts,
               "guild_join_last_day": {k: float(v) for k, v in daily[14].items()}}
    json.dump(summary, open(ROOT / "results" / "dashboard_values.json", "w"), indent=2)
    render(tiles, ab, tiers, daily, alerts, (p1 - p2) * 100, pval)
    print(json.dumps(summary, indent=2))


def render(tiles, ab, tiers, daily, alerts, diff, pval):
    tile_html = "".join(f'<div class="t"><div class="l">{k}</div><div class="v">{v}</div></div>' for k, v in tiles.items())
    abbars = "".join(f'<div class="br"><span>{g.title()} group ({int(r.players):,} players)</span><div class="bt"><div class="bf" style="width:{r.rr7*100:.1f}%;background:{c}"></div></div><b>{r.rr7:.1%}</b></div>'
                     for (g, r), c in zip(ab.loc[["test", "control"]].iterrows(), ["#2a78d6", "#9b9a95"]))
    tierbars = "".join(f'<div class="br"><span>{t} risk ({v["share"]:.0%})</span><div class="bt"><div class="bf" style="width:{v["churn_by_d7"]*100:.1f}%;background:#eb6834"></div></div><b>{v["churn_by_d7"]:.1%}</b></div>'
                       for t, v in tiers.items())
    W, H = 520, 170
    lines, colors = "", {s: "#c9c8c2" for s in daily.index}
    colors["S184"] = "#d03b3b"
    for srv in daily.index:
        vals = daily.loc[srv].values * 100
        pts = " ".join(f"{40 + i * (W - 50) / 13:.1f},{H - 20 - (v - 0) / 30 * (H - 40):.1f}" for i, v in enumerate(vals))
        lines += f'<polyline points="{pts}" fill="none" stroke="{colors[srv]}" stroke-width="{2.4 if srv == "S184" else 1.4}"/>'
    grid = "".join(f'<line x1="40" x2="510" y1="{H - 20 - v / 30 * (H - 40):.1f}" y2="{H - 20 - v / 30 * (H - 40):.1f}" stroke="#e6e5e0"/><text x="32" y="{H - 16 - v / 30 * (H - 40):.1f}" font-size="10" fill="#52514e" text-anchor="end">{v}%</text>' for v in (0, 10, 20, 30))
    xl = "".join(f'<text x="{40 + (d - 1) * (W - 50) / 13:.1f}" y="{H - 4}" font-size="10" fill="#52514e" text-anchor="middle">Day {d}</text>' for d in (1, 4, 7, 10, 14))
    svg = f'<svg width="{W + 10}" height="{H}">{grid}{lines}{xl}</svg>'
    col = {"Critical": "#d03b3b", "Warning": "#fab219", "OK": "#0ca30c"}
    al = "".join(f'<div class="al" style="border-color:{col[a]}">{a}: {m}<small>Owner: {o}</small></div>' for a, m, o in alerts)
    html = f'''<html><head><meta charset="utf-8"><style>
body{{margin:0;background:#f3f2ee;font-family:"DejaVu Sans",Arial;color:#0b0b0b;width:1180px;padding-bottom:6px}}
.top{{background:#1d2b4f;color:#fff;padding:14px 22px;display:flex;justify-content:space-between;align-items:center}}
.top h1{{font-size:19px;margin:0}}.top span{{font-size:12px;opacity:.85}}
.sec{{font-size:12px;font-weight:bold;color:#52514e;letter-spacing:.5px;padding:14px 22px 6px;text-transform:uppercase}}
.tiles{{display:flex;gap:12px;padding:0 22px}}.t{{flex:1;background:#fcfcfb;border:1px solid #dddbd4;border-radius:8px;padding:12px 14px}}
.t .l{{font-size:11.5px;color:#52514e}}.t .v{{font-size:26px;font-weight:bold;margin:4px 0}}
.row{{display:flex;gap:12px;padding:0 22px 6px}}.card{{flex:1;background:#fcfcfb;border:1px solid #dddbd4;border-radius:8px;padding:12px 14px}}
.card h3{{font-size:13px;margin:0 0 6px}}.note{{font-size:10.5px;color:#52514e}}
.br{{display:flex;align-items:center;gap:8px;font-size:12px;margin:9px 0}}.br span{{width:200px}}.bt{{flex:1;height:10px;background:#ecebe6;border-radius:5px}}.bf{{height:10px;border-radius:5px}}.br b{{width:48px;text-align:right}}
.al{{font-size:12px;border-left:4px solid;padding:6px 10px;margin:8px 0;background:#fff}}.al small{{display:block;color:#52514e;margin-top:2px}}
</style></head><body>
<div class="top"><h1>KTO New-Player Live Ops Monitor</h1><span>Sample data for illustration | New servers S180-S186 | Day 14</span></div>
<div class="sec">Tier 1 - Leading (D1 - D3)</div><div class="tiles">{tile_html}</div>
<div class="sec">Tier 2 - Intermediate (D3 - D14)</div>
<div class="row"><div class="card"><h3>Not churned by D7, by group: dynamic guidance A/B test</h3><div class="note">Scenario: guidance helps 10% of test players who missed the quest break</div>{abbars}<div class="note">Difference {diff:+.1f} pp, p = {pval:.2f}: not significant, keep the test running or fix the design</div></div>
<div class="card"><h3>Churn by D7 in each risk tier (Early Churn Alert)</h3><div class="note">Test set of the sample</div>{tierbars}</div></div>
<div class="sec">Tier 3 - Daily checks and alerts</div>
<div class="row"><div class="card"><h3>Guild join rate by server and join day</h3><div class="note">S184 in red (incident scenario on day 14)</div>{svg}</div>
<div class="card"><h3>Alerts today</h3>{al}</div></div>
</body></html>'''
    (ROOT / "dashboard").mkdir(exist_ok=True)
    (ROOT / "dashboard" / "dashboard.html").write_text(html, encoding="utf-8")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1180, "height": 600}, device_scale_factor=2)
            pg.goto((ROOT / "dashboard" / "dashboard.html").as_uri())
            pg.locator("body").screenshot(path=str(ROOT / "figures" / "dashboard.png"))
            b.close()
    except Exception as e:
        print("Screenshot skipped (playwright not available):", e)


if __name__ == "__main__":
    main()
