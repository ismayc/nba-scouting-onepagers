"""Auto-generated opponent scouting one-pager, one per team, from public data.

The daily deliverable of a coaching-analytics seat: tomorrow's opponent on
one page, every number traceable to a pipeline. This generator composes the
repo's existing machinery rather than inventing new stats:

  - 2025-26 shot profile priced by the shot-quality model's committed
    2023-24 coefficients, legitimate on unseen seasons because the
    season-out backtest (shot-quality-study/06) validated exactly that
    transfer;
  - shot-making shrunk by the empirical-Bayes rule from 08_projection.py
    (imported, not reimplemented), so small-sample hot streaks don't drive
    the scout;
  - lineup net ratings from the per-team Base+Advanced files and player
    RAPM from lineup-valuation-study's 2025-26 model.

Writes scouting/reports/<ABBR>.md and .html (self-contained, print-ready).

Run: python build_onepager.py PHX          one team
     python build_onepager.py --all        all 30
     python build_onepager.py --check      regenerate PHX in memory,
                                                    fail if the committed
                                                    report is stale
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parent
SIBLINGS = REPO.parent          # study repos checked out next door
REPORTS = REPO / "docs"         # served by GitHub Pages
SHOTS_CSV = SIBLINGS / "shot-quality-study/data/shotdetail_2025.csv"
COEFS_CSV = SIBLINGS / "shot-quality-study/output/coefficients.csv"
VALUES_CSV = SIBLINGS / "lineup-valuation-study/output/2025-26/player_values.csv"
LINEUP_RAW = SIBLINGS / "lineup-valuation-study/data/raw/2025-26"
SEASON = "2025-26"
MIN_FGA_EB = 300          # qualifying pool for the EB hyperparameters
CORNERS = ["Left Corner 3", "Right Corner 3"]


def load_module(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SIBLINGS / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


shotq = load_module("shot-quality-study/python/02_model.py", "shotq_model")
proj = load_module("shot-quality-study/python/08_projection.py",
                   "shotq_projection")
glossary = load_module("basketball-analysis-tools/glossary.py", "portfolio_glossary")


def load_scored_shots() -> pl.DataFrame:
    """All 2025-26 shots, priced by the committed 2023-24 model."""
    df = proj.prepare(SHOTS_CSV)
    X, _ = shotq.design_matrix(df)
    beta = pl.read_csv(COEFS_CSV)["coef"].to_numpy()
    secs = pl.col("MINUTES_REMAINING") * 60 + pl.col("SECONDS_REMAINING")
    return df.with_columns(
        xpts=pl.Series(shotq.predict_make(X, beta)) * pl.col("value"),
        secs_left=secs)


def team_abbrs(shots: pl.DataFrame) -> dict[str, str]:
    """TEAM_NAME -> abbreviation. Half of a team's shots are at home, where
    HTM is its own abbreviation, so the per-team mode of HTM is the answer;
    sanity-gated against the lineup files' abbreviation set."""
    mode = (shots.group_by("TEAM_NAME", "HTM").len()
            .sort("len", descending=True)
            .unique("TEAM_NAME", keep="first"))
    mapping = dict(zip(mode["TEAM_NAME"], mode["HTM"]))
    known = {p.stem.rsplit("_", 1)[1] for p in LINEUP_RAW.glob("lineups_base_*.parquet")}
    assert set(mapping.values()) == known, "abbreviation mapping drifted"
    return mapping


def profile(df: pl.DataFrame) -> dict[str, float]:
    """Shot-profile metrics for any slice of scored shots."""
    three = df.filter(pl.col("value") == 3)
    return {
        "fga": df.height,
        "pps": float(df["points"].mean()),
        "xpps": float(df["xpts"].mean()),
        "making_100": float((df["points"] - df["xpts"]).mean() * 100),
        "pct_rim": float((df["SHOT_ZONE_BASIC"] == "Restricted Area").mean()),
        "pct_midrange": float((df["SHOT_ZONE_BASIC"] == "Mid-Range").mean()),
        "pct_three": float((df["value"] == 3).mean()),
        "corner_share_of_3s": float(three["SHOT_ZONE_BASIC"]
                                    .is_in(CORNERS).mean()),
        "pct_rush": float((df["secs_left"] <= 4).mean()),
        "rush_execution_100": float(
            (df.filter(pl.col("secs_left") <= 4)["points"]
             - df.filter(pl.col("secs_left") <= 4)["xpts"]).mean() * 100),
    }


def eb_hyperparams(shots: pl.DataFrame) -> tuple[float, float]:
    """League mu and tau2 for shot-making, from the season's own 300+ FGA
    pool, the same normal-normal EB rule the projection study validated."""
    pool = shotq.player_table(shots).filter(pl.col("fga") >= MIN_FGA_EB)
    _, mu, tau2, _ = proj.eb_posterior(pool["making_100"].to_numpy(),
                                       pool["making_100_se"].to_numpy())
    return mu, tau2


def personnel(shots: pl.DataFrame, team: str, mu: float, tau2: float,
              values: pl.DataFrame) -> pl.DataFrame:
    tbl = (shotq.player_table(shots.filter(pl.col("TEAM_NAME") == team))
           .sort("fga", descending=True).head(8))
    se = tbl["making_100_se"].to_numpy()
    w = tau2 / (tau2 + se ** 2)
    return (tbl.with_columns(
        making_eb=pl.Series(mu + w * (tbl["making_100"].to_numpy() - mu)))
        .join(values.select("player_id", "rapm_100"),
              left_on="PLAYER_ID", right_on="player_id", how="left"))


def top_lineups(abbr: str, n: int = 3) -> pl.DataFrame:
    base = pl.read_parquet(LINEUP_RAW / f"lineups_base_{abbr}.parquet")
    adv = pl.read_parquet(LINEUP_RAW / f"lineups_advanced_{abbr}.parquet")
    return (base.select("GROUP_ID", "GROUP_NAME", "MIN", "PLUS_MINUS")
            .join(adv.select("GROUP_ID", "POSS"), on="GROUP_ID")
            .filter(pl.col("POSS") > 0)
            .with_columns(net_100=100.0 * pl.col("PLUS_MINUS")
                          / pl.col("POSS"))
            .sort("POSS", descending=True).head(n))


def fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def fmt_delta(x: float, pct: bool = True) -> str:
    return (f"{x:+.1f} pp" if pct else f"{x:+.3f}").replace("+-", "-")


def keys_bullets(p: dict, lg: dict, pers: pl.DataFrame,
                 lineups: pl.DataFrame) -> list[str]:
    keys = []
    d_corner = p["corner_share_of_3s"] - lg["corner_share_of_3s"]
    if d_corner > 0.02:
        keys.append(f"**They hunt corner threes**: {fmt_pct(p['corner_share_of_3s'])} "
                    f"of their threes come from the corners (league "
                    f"{fmt_pct(lg['corner_share_of_3s'])}). Run shooters off "
                    "the corner line first.")
    elif d_corner < -0.02:
        keys.append(f"**Above-the-break three team**: corners are only "
                    f"{fmt_pct(p['corner_share_of_3s'])} of their threes "
                    f"(league {fmt_pct(lg['corner_share_of_3s'])}); their "
                    "spacing lives higher on the arc.")
    if p["pct_rim"] - lg["pct_rim"] > 0.02:
        keys.append(f"**Rim-pressure offense**: {fmt_pct(p['pct_rim'])} of "
                    f"attempts at the rim vs league {fmt_pct(lg['pct_rim'])}. "
                    "Wall building beats switching small.")
    if p["pct_three"] - lg["pct_three"] > 0.03:
        keys.append(f"**Volume-three profile**: {fmt_pct(p['pct_three'])} of "
                    f"attempts from deep (league {fmt_pct(lg['pct_three'])}).")
    if p["pct_midrange"] - lg["pct_midrange"] > 0.03:
        keys.append(f"**Mid-range reliant**: {fmt_pct(p['pct_midrange'])} "
                    f"from mid-range vs league {fmt_pct(lg['pct_midrange'])}; "
                    "live with contested twos over open threes.")
    qual = pers.filter(pl.col("fga") >= 200)
    if qual.height:
        mk = qual.sort("making_eb", descending=True).row(0, named=True)
        if mk["making_eb"] > 2:
            keys.append(f"**{mk['PLAYER_NAME']} is a genuine shot-maker**: "
                        f"{mk['making_eb']:+.1f} pts/100 above expectation "
                        "after empirical-Bayes shrinkage; contest even the "
                        "bad shots.")
        sel = qual.sort("xpps", descending=True).row(0, named=True)
        keys.append(f"**Best shot diet: {sel['PLAYER_NAME']}** "
                    f"({sel['xpps']:.2f} xPPS on {sel['fga']} FGA). The "
                    "value is where the shots come from; deny the spot, not "
                    "the release.")
    if lineups.height:
        lu = (lineups.sort("net_100", descending=True).row(0, named=True))
        keys.append(f"**Their most-used five that wins its minutes**: "
                    f"{lu['GROUP_NAME']} ({lu['net_100']:+.1f}/100 over "
                    f"{lu['POSS']:,} possessions).")
    return keys


def render_md(team: str, abbr: str, p: dict, lg: dict, pers: pl.DataFrame,
              lineups: pl.DataFrame, n_games: int) -> str:
    rows = [
        ("PPS (points per shot)", f"{p['pps']:.3f}", f"{lg['pps']:.3f}",
         fmt_delta(p["pps"] - lg["pps"], pct=False)),
        ("xPPS, the shot-diet price*", f"{p['xpps']:.3f}", f"{lg['xpps']:.3f}",
         fmt_delta(p["xpps"] - lg["xpps"], pct=False)),
        ("Shot-making, pts/100 vs model", f"{p['making_100']:+.1f}",
         f"{lg['making_100']:+.1f}",
         f"{p['making_100'] - lg['making_100']:+.1f}"),
        ("Rim share", fmt_pct(p["pct_rim"]), fmt_pct(lg["pct_rim"]),
         fmt_delta((p["pct_rim"] - lg["pct_rim"]) * 100)),
        ("Mid-range share", fmt_pct(p["pct_midrange"]),
         fmt_pct(lg["pct_midrange"]),
         fmt_delta((p["pct_midrange"] - lg["pct_midrange"]) * 100)),
        ("Three share", fmt_pct(p["pct_three"]), fmt_pct(lg["pct_three"]),
         fmt_delta((p["pct_three"] - lg["pct_three"]) * 100)),
        ("Corner share of threes", fmt_pct(p["corner_share_of_3s"]),
         fmt_pct(lg["corner_share_of_3s"]),
         fmt_delta((p["corner_share_of_3s"] - lg["corner_share_of_3s"]) * 100)),
        ("Final-4s (rush) shot share", fmt_pct(p["pct_rush"]),
         fmt_pct(lg["pct_rush"]),
         fmt_delta((p["pct_rush"] - lg["pct_rush"]) * 100)),
    ]
    profile_tbl = "\n".join(f"| {a} | {b} | {c} | {d} |"
                            for a, b, c, d in rows)
    pers_tbl = "\n".join(
        f"| {r['PLAYER_NAME']} | {r['fga']:,} | {r['pps']:.3f} "
        f"| {r['xpps']:.3f} | {r['making_100']:+.1f} | {r['making_eb']:+.1f} "
        f"| {'' if r['rapm_100'] is None else format(r['rapm_100'], '+.1f')} |"
        for r in pers.iter_rows(named=True))
    lineup_tbl = "\n".join(
        f"| {r['GROUP_NAME']} | {r['MIN']:,.0f} | {r['POSS']:,} "
        f"| {r['net_100']:+.1f} |"
        for r in lineups.iter_rows(named=True))
    keys = "\n".join(f"- {k}" for k in keys_bullets(p, lg, pers, lineups))

    return f"""# Scouting one-pager: {team} ({abbr})

**{SEASON} regular season · {n_games} games · {p['fga']:,} FGA.**
Auto-generated by `scouting/build_onepager.py` from public data; every
number traceable to a gated pipeline in this repo. Shot pricing uses the
2023-24 shot-quality model, whose season-out transfer was validated in the
backtest; shot-making is empirical-Bayes shrunk before it is allowed to
drive a conclusion.

## Keys

{keys}

## Shot profile vs league

| | {abbr} | League | Δ |
|---|---|---|---|
{profile_tbl}

\\* xPPS prices where they shoot from, independent of whether shots fell:
the stable, scoutable part of the profile (year-over-year r ≈ .9).

## Personnel (top 8 by volume)

| Player | FGA | PPS | xPPS (diet) | Making/100 | EB making | RAPM/100 |
|---|---|---|---|---|---|---|
{pers_tbl}

EB making = shot-making regressed to league mean by reliability: the
number to trust. RAPM from the opponent-adjusted 2025-26 lineup model
(±≈3/100 CI at a season of minutes; treat as descriptive).

## Most-used lineups

| Lineup | Min | Poss | Net/100 |
|---|---|---|---|
{lineup_tbl}
"""


HTML_STYLE = """
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
       width: min(1150px, 94vw); margin: 2rem auto; padding: 0; color: #1a1a2e;
       line-height: 1.45; }
h1 { border-bottom: 3px solid #e56020; padding-bottom: .3rem; }
h2 { color: #1d1160; margin-top: 1.6rem; }
table { border-collapse: collapse; width: 100%; font-size: .92rem;
        font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid #ddd; padding: .35rem .5rem;
         text-align: left; }
th { background: #1d1160; color: #fff; }
tr:nth-child(even) td { background: #f6f5fa; }
li { margin: .35rem 0; }
em.note { color: #555; font-size: .85rem; }
abbr[title] { text-decoration: underline dotted #e56020; cursor: help;
              text-underline-offset: 2px; }
@media print { body { margin: 0 auto; } }
"""


def md_to_html(md: str, title: str) -> str:
    """Minimal, dependency-free markdown -> HTML for the report's own
    subset: headings, tables, bullet lists, bold, paragraphs."""
    import re

    def inline(s: str) -> str:
        s = (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;"))
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)

    out, table, bullets, para = [], [], [], []

    def flush():
        if table:
            head, rows = table[0], table[2:]
            cells = lambda r: [c.strip() for c in r.strip("|").split("|")]
            out.append("<table><tr>" + "".join(
                f"<th>{inline(c)}</th>" for c in cells(head)) + "</tr>")
            out.extend("<tr>" + "".join(f"<td>{inline(c)}</td>"
                                        for c in cells(r)) + "</tr>"
                       for r in rows)
            out.append("</table>")
            table.clear()
        if bullets:
            out.append("<ul>" + "".join(f"<li>{inline(b)}</li>"
                                        for b in bullets) + "</ul>")
            bullets.clear()
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            if bullets or para:
                flush()
            table.append(line)
        elif line.startswith("- "):
            if table or para:
                flush()
            bullets.append(line[2:])
        elif line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{inline(line.lstrip('# '))}</h{level}>")
        elif not line.strip():
            flush()
        else:
            if table or bullets:
                flush()
            para.append(line.replace("\\*", "*"))
    flush()
    return (f"<title>{title}</title>\n<style>{HTML_STYLE}</style>\n"
            + tipify_html("\n".join(out)) + "\n")


def tipify_html(body: str) -> str:
    """Hover tooltips at point of use: wrap glossary terms in <abbr>
    everywhere except inside tags and inside the Terms section itself
    (a term needs no tooltip in its own definition list)."""
    import re

    out, skip = [], False
    for part in re.split(r"(<h2>.*?</h2>)", body):
        if part.startswith("<h2>"):
            skip = part == "<h2>Terms</h2>"
            out.append(part)
        elif skip:
            out.append(part)
        else:
            out.append("".join(
                seg if seg.startswith("<") else glossary.wrap_terms(seg)
                for seg in re.split(r"(<[^>]+>)", part)))
    return "".join(out)


def build_report(team: str, abbr: str, shots: pl.DataFrame, lg: dict,
                 mu: float, tau2: float, values: pl.DataFrame) -> str:
    tshots = shots.filter(pl.col("TEAM_NAME") == team)
    p = profile(tshots)
    pers = personnel(shots, team, mu, tau2, values)
    lineups = top_lineups(abbr)
    n_games = tshots["GAME_ID"].n_unique()
    md = render_md(team, abbr, p, lg, pers, lineups, n_games)
    # Definitions up top for every glossary term the page actually uses;
    # the HTML renderer adds hover tooltips at each point of use.
    keys = glossary.terms_for(md)
    block = ("## Terms\n\n"
             + "\n".join(f"- **{k}**: {glossary.TERMS[k][0]}"
                         for k in keys)
             + "\n\n")
    return md.replace("## Keys\n", block + "## Keys\n", 1)


def main(argv: list[str]) -> int:
    shots = load_scored_shots()
    lg = profile(shots)
    mu, tau2 = eb_hyperparams(shots)
    values = pl.read_csv(VALUES_CSV)
    abbrs = team_abbrs(shots)

    if argv and argv[0] == "--check":
        team = next(t for t, a in abbrs.items() if a == "PHX")
        md = build_report(team, "PHX", shots, lg, mu, tau2, values)
        committed = (REPORTS / "PHX.md").read_text()
        if md != committed:
            print("SCOUTING CHECK FAILED: committed PHX.md is stale; "
                  "re-run scouting/build_onepager.py --all")
            return 1
        print("SCOUTING CHECK PASSED (committed PHX report reproduces "
              "byte-for-byte)")
        return 0

    if argv and argv[0] == "--all":
        targets = sorted(abbrs.items(), key=lambda kv: kv[1])
    else:
        want = set(argv) or {"PHX"}
        targets = [(t, a) for t, a in abbrs.items() if a in want]
        missing = want - {a for _, a in targets}
        if missing:
            print(f"unknown team(s): {sorted(missing)}; "
                  f"known: {sorted(abbrs.values())}")
            return 1

    REPORTS.mkdir(parents=True, exist_ok=True)
    for team, abbr in targets:
        md = build_report(team, abbr, shots, lg, mu, tau2, values)
        (REPORTS / f"{abbr}.md").write_text(md)
        (REPORTS / f"{abbr}.html").write_text(
            md_to_html(md, f"{team} scouting one-pager, {SEASON}"))
        print(f"wrote {abbr}.md / {abbr}.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
