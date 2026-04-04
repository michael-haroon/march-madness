"""
WarrenNolan.com NET Nitty Gritty Report Scraper
------------------------------------------------
Extracts result-based metrics (KPI, SOR), predictive metrics (BPI, POM, SAG),
and strength of schedule (NET SOS, NET Non-Conf SOS, RPI SOS, RPI Non-Conf SOS)
for every team in the saved HTML file.

Usage:
    python scrape_nolan.py

Output:
    nolan_metrics.csv
"""

import csv
import os
import re
from bs4 import BeautifulSoup
from pathlib import Path

search_dir = Path("/Users/michaelharoon/Projects/tasty/march-madness/data/raw/html")

HTML_FILES = [f for f in search_dir.iterdir() if f.is_file()]
OUTPUT_FILE = '/Users/michaelharoon/Projects/tasty/march-madness/data/team_sheets/{date}_Team_Sheet_Selection.csv'


def clean(text):
    """Strip whitespace and non-breaking spaces."""
    return text.replace("\xa0", " ").strip()


def parse_record(text):
    """Return (wins, losses) from a 'W-L' string, or (None, None)."""
    text = clean(text)
    m = re.match(r"^(\d+)-(\d+)$", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def get_br_values(tag):
    """
    Given a div that contains values separated by <br> tags,
    return a list of clean non-empty strings.
    """
    html = str(tag)
    # Replace <br> variants with newlines, then parse again
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    inner = BeautifulSoup(html, "html.parser").get_text("\n")
    lines = [clean(l) for l in inner.split("\n")]
    return [l for l in lines if l]


def parse_team_block(block):
    """Parse one full team block and return a dict of fields."""
    data = {}

    # ── NET rank & team name ─────────────────────────────────────────────
    rank_div = block.find("div", class_="ts-rank")
    data["NET_Rank"] = clean(rank_div.get_text()) if rank_div else None

    name_div = block.find("div", class_="ts-teamname")
    if name_div:
        # Team name is the direct text before the <span>
        name_text = name_div.contents[0] if name_div.contents else ""
        data["Team"] = clean(str(name_text).replace("<br>", "").replace("</br>", ""))
        # Conference (record) is inside the nested <span>
        conf_span = name_div.find("span")
        data["Conference_Record"] = clean(conf_span.get_text()) if conf_span else None
    else:
        data["Team"] = None
        data["Conference_Record"] = None

    # ── Overall record & road record ────────────────────────────────────
    # They sit in ts-flex-size-1 divs; the second such div holds records.
    flex1_divs = block.find_all("div", class_="ts-flex-size-1")
    if len(flex1_divs) >= 2:
        records_div = flex1_divs[1]
        center_divs = records_div.find_all("div", class_="ts-data-center")
        if len(center_divs) >= 1:
            vals = get_br_values(center_divs[0])
            # vals[0] = label ("RECORD"), vals[1] = overall, vals[2] = non-conf
            data["Overall_Record"] = vals[1] if len(vals) > 1 else None
            data["NonConf_Record"] = vals[2] if len(vals) > 2 else None
        if len(center_divs) >= 2:
            vals = get_br_values(center_divs[1])
            data["Road_Record"] = vals[1] if len(vals) > 1 else None

    # ── Strength of Schedule ─────────────────────────────────────────────
    flex1_sos = flex1_divs[2] if len(flex1_divs) >= 3 else None
    if flex1_sos:
        right_divs = flex1_sos.find_all("div", class_="ts-title-right")
        center_divs = flex1_sos.find_all("div", class_="ts-data-center")
        # Layout: [NET SOS label block] [NET SOS values] [RPI SOS label block] [RPI SOS values]
        if len(center_divs) >= 1:
            net_vals = get_br_values(center_divs[0])
            data["NET_SOS"] = net_vals[0] if len(net_vals) > 0 else None
            data["NET_NonConf_SOS"] = net_vals[1] if len(net_vals) > 1 else None
        if len(center_divs) >= 2:
            rpi_vals = get_br_values(center_divs[1])
            data["RPI_SOS"] = rpi_vals[0] if len(rpi_vals) > 0 else None
            data["RPI_NonConf_SOS"] = rpi_vals[1] if len(rpi_vals) > 1 else None

    # ── Average NET wins/losses ──────────────────────────────────────────
    flex0_divs = block.find_all("div", class_="ts-flex-size-0")
    if flex0_divs:
        avg_div = flex0_divs[0]
        avg_vals = get_br_values(avg_div)
        # Expected: ["Average NET", "Wins:  113", "Losses:  2"]
        for v in avg_vals:
            if v.lower().startswith("wins"):
                data["Avg_NET_Wins"] = clean(v.split(":", 1)[-1])
            elif v.lower().startswith("losses"):
                data["Avg_NET_Losses"] = clean(v.split(":", 1)[-1])

    # ── Result-Based Metrics (KPI, SOR) ─────────────────────────────────
    # ── Predictive Metrics (BPI, POM, SAG) ──────────────────────────────
    half_width_divs = block.find_all("div", class_="ts-half-width")
    for hw in half_width_divs:
        title = hw.find("div", class_="ts-title-full-width")
        if not title:
            continue
        title_text = clean(title.get_text()).lower()

        labels_div = hw.find("div", class_="ts-data-right")
        values_div = hw.find("div", class_="ts-data-left")
        if not labels_div or not values_div:
            continue

        labels = get_br_values(labels_div)
        values = get_br_values(values_div)

        for label, value in zip(labels, values):
            key = label.replace(":", "").strip()
            if "result" in title_text:
                data[f"RB_{key}"] = value
            elif "predictive" in title_text:
                data[f"PM_{key}"] = value

    return data


def main():
    for html_file in HTML_FILES:
        print(f"Reading {html_file} ...")
        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Every team has a "ts-wrapper show" or "ts-wrapper hide" div marked with "-full"
        team_blocks = [
            d for d in soup.find_all("div", class_="ts-wrapper")
            if d.get("id", "").endswith("-full")
        ]
        print(f"Found {len(team_blocks)} team blocks.")

        rows = []
        for block in team_blocks:
            try:
                row = parse_team_block(block)
                rows.append(row)
            except Exception as e:
                print(f"  ⚠ Error parsing block id={block.get('id')}: {e}")

        if not rows:
            print("No data extracted — check HTML structure.")
            return

        # Build unified column order
        preferred_cols = [
            "NET_Rank", "Team", "Conference_Record",
            "Overall_Record", "NonConf_Record", "Road_Record",
            "NET_SOS", "NET_NonConf_SOS", "RPI_SOS", "RPI_NonConf_SOS",
            "Avg_NET_Wins", "Avg_NET_Losses",
            "RB_KPI", "RB_SOR",
            "PM_BPI", "PM_POM", "PM_SAG",
        ]
        # Add any extra keys found
        all_keys = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        extra = [k for k in all_keys if k not in preferred_cols]
        fieldnames = preferred_cols + extra

        filename = os.path.basename(html_file)
        date = filename.split(' ')[0]

        with open(OUTPUT_FILE.format(date=date), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print(f"✅  Saved {len(rows)} rows → {OUTPUT_FILE.format(date=date)}")
        # Preview first 3 rows
        print("\nPreview (first 3 teams):")
        preview_cols = ["NET_Rank", "Team", "NET_SOS", "RPI_SOS", "RB_KPI", "RB_SOR", "PM_BPI", "PM_POM", "PM_SAG"]
        header = " | ".join(f"{c:<18}" for c in preview_cols)
        print(header)
        print("-" * len(header))
        for row in rows[:3]:
            print(" | ".join(f"{str(row.get(c, '')):<18}" for c in preview_cols))
        print('\n')


if __name__ == "__main__":
    main()
