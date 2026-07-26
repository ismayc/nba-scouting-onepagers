"""Unit tests for scouting/build_onepager.py pure functions.

The end-to-end gate (committed PHX report reproduces byte-for-byte) runs in
run_all_checks.sh; these pin the profile arithmetic, the keys rules, and
the report's tiny markdown renderer on synthetic data.
"""
from __future__ import annotations

import polars as pl


def synthetic_shots() -> pl.DataFrame:
    # Four shots: made corner 3, missed above-break 3, made rim 2, missed
    # mid-range 2 taken in the final-4s rush.
    return pl.DataFrame({
        "SHOT_ZONE_BASIC": ["Left Corner 3", "Above the Break 3",
                            "Restricted Area", "Mid-Range"],
        "value": [3, 3, 2, 2],
        "points": [3, 0, 2, 0],
        "xpts": [1.1, 1.0, 1.3, 0.8],
        "secs_left": [300, 120, 45, 2],
    })


def test_profile_arithmetic(scout):
    p = scout.profile(synthetic_shots())
    assert p["fga"] == 4
    assert p["pps"] == 5 / 4
    assert p["xpps"] == (1.1 + 1.0 + 1.3 + 0.8) / 4
    assert p["pct_rim"] == p["pct_midrange"] == 1 / 4
    assert p["pct_three"] == 1 / 2
    assert p["corner_share_of_3s"] == 1 / 2
    assert p["pct_rush"] == 1 / 4
    # The one rush shot: 0 points on 0.8 xpts -> -80 per 100.
    assert abs(p["rush_execution_100"] - (-80.0)) < 1e-9


def test_keys_rules_fire_on_clear_profiles(scout):
    lg = {"corner_share_of_3s": 0.25, "pct_rim": 0.28, "pct_three": 0.40,
          "pct_midrange": 0.10}
    team = {"corner_share_of_3s": 0.31, "pct_rim": 0.33, "pct_three": 0.46,
            "pct_midrange": 0.10}
    keys = scout.keys_bullets(team, lg,
                              pl.DataFrame(schema={"fga": pl.Int64}),
                              pl.DataFrame(schema={"GROUP_NAME": pl.String}))
    text = " ".join(keys)
    assert "corner" in text.lower()
    assert "Rim-pressure" in text
    assert "Volume-three" in text


def test_md_to_html_renders_report_subset(scout):
    md = ("# Title\n\nSome **bold** prose.\n\n"
          "| a | b |\n|---|---|\n| 1 | 2 |\n\n- first key\n- second key\n")
    html = scout.md_to_html(md, "t")
    assert "<h1>Title</h1>" in html
    assert "<strong>bold</strong>" in html
    assert "<th>a</th>" in html and "<td>2</td>" in html
    assert html.count("<li>") == 2
    assert "<script" not in html


def test_html_tooltips_skip_the_terms_section(scout):
    md = ("# T\n\n## Terms\n\n- **xPPS** — the definition.\n\n"
          "## Body\n\nxPPS drives the read.\n")
    html = scout.md_to_html(md, "t")
    terms_part = html.split("<h2>Body</h2>")[0]
    body_part = html.split("<h2>Body</h2>")[1]
    assert "<abbr" not in terms_part
    assert '<abbr title="' in body_part and ">xPPS</abbr>" in body_part
