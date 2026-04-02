import re
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

OUT_CSV = Path("/Users/michaelharoon/Projects/tasty/march-madness/data/raw/bpi/espn_bpi_team_rankings_2008_to_2012.csv")
SEASONS = list(range(2008, 2013))

TEAM_BLOCK_RE = re.compile(
    r'"team":\{.*?"displayName":"(?P<team>.*?)".*?\},"stats":\[(?P<stats>.*?)\]',
    re.S
)
BPI_RANK_RE = re.compile(r'"name":"bpirank","value":"(?P<rank>\d+)"')


def unescape_espn_text(s: str) -> str:
    return (
        s.replace(r"\/", "/")
         .replace(r"\\u002F", "/")
         .replace(r"\\u0026", "&")
         .replace(r'\\"', '"')
         .strip()
    )


def season_url(season: int) -> str:
    return f"https://www.espn.com/mens-college-basketball/bpi/_/season/{season}/sort/bpi.bpi/dir/asc"


def click_show_more_until_done(page, max_clicks=100):
    for i in range(max_clicks):
        link = page.locator("a.loadMore__link")

        if link.count() == 0:
            print("No more 'Show More' button.")
            break

        try:
            # scroll into view
            link.first.scroll_into_view_if_needed()

            # try normal click first
            try:
                link.first.click(timeout=5000)
            except:
                # fallback: JS click (more reliable on ESPN)
                page.evaluate("""
                () => {
                    const btn = document.querySelector('a.loadMore__link');
                    if (btn) btn.click();
                }
                """)

        except Exception as e:
            print("Click failed:", e)
            break

        # wait for data load (network OR render)
        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except:
            pass

        # fallback wait (ESPN often needs this)
        page.wait_for_timeout(1500)

        # debug row growth
        rows = page.locator("tbody tr").count()
        print(f"Click {i+1}: rows now = {rows}")


def parse_rows_from_dom(page, season: int) -> pd.DataFrame:
    # ALL rows (stats side)
    rows = page.locator("div.Table__Scroller tbody tr:visible")

    # ALL team names (left side)
    teams = page.locator("span.TeamLink__Name a")

    row_count = rows.count()
    team_count = teams.count()

    print(f"teams: {team_count}, rows: {row_count}")

    # sanity check
    if team_count == 0:
        raise RuntimeError("Team selector is broken")

    data = []
    n = min(row_count, team_count)

    for i in range(n):
        try:
            team = teams.nth(i).inner_text().strip()
            tds = rows.nth(i).locator("td")

            rank_text = tds.nth(2).inner_text().strip()

            if not rank_text.isdigit():
                continue

            rank = int(rank_text)

            # DEBUG LOGGING
            if i < 15:
                print(f"[{i}] {team} -> rank {rank}")

            data.append({
                "season": season,
                "team": team,
                "bpi_rank": rank
            })

        except Exception as e:
            print(f"error at {i}: {e}")
            continue

    if not data:
        raise RuntimeError(f"No rows parsed for season {season}")

    return (
        pd.DataFrame(data)
        .drop_duplicates(subset=["season", "team"])
        .sort_values("bpi_rank")
        .reset_index(drop=True)
    )

def main():
    all_dfs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()

        for season in SEASONS:
            url = season_url(season)
            print(f"\nScraping {season}: {url}")

            page = context.new_page()

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=120000)

                # initial wait for table render
                page.wait_for_selector("tbody tr", timeout=10000)

                click_show_more_until_done(page)

                df = parse_rows_from_dom(page, season)
                
                print(f"Final rows parsed: {len(df)}")
                all_dfs.append(df)

            except Exception as e:
                print(f"Failed season {season}: {e}")

            finally:
                page.close()
                time.sleep(0.5)

        browser.close()

    if not all_dfs:
        raise RuntimeError("No seasons scraped successfully.")

    final = pd.concat(all_dfs, ignore_index=True).sort_values(["season", "bpi_rank"])
    final.to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()