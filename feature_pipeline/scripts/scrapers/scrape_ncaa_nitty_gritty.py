import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

BASE_URL = "https://stats.ncaa.org/selection_rankings/nitty_gritties/"
BASE_DOWNLOAD_DIR = '/Users/michaelharoon/Projects/tasty/march-madness/data/raw/nitty_gritty/ncaa_nitty_gritty_downloads/'

def clean_filename(s: str) -> str:
    s = re.sub(r"[\\\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s

def wait_for_page_ready(page):
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_load_state("networkidle")

def get_thru_games_select(page):
    candidate_selectors = [
        "select:near(:text('Thru Games'))",
        "label:has-text('Thru Games') + select",
        "select[name*='thru']",
        "select[id*='thru']",
        "select",
    ]
    for sel in candidate_selectors:
        try:
            locator = page.locator(sel).first
            if locator.count() > 0:
                return locator
        except Exception:
            pass
    raise RuntimeError("Could not find the 'Thru Games' dropdown. Inspect the page and update the selector.")

def get_excel_button(page):
    candidate_selectors = [
        "text=Excel",
        "input[value='Excel']",
        "button:has-text('Excel')",
        "a:has-text('Excel')",
    ]
    for sel in candidate_selectors:
        try:
            locator = page.locator(sel).first
            if locator.count() > 0:
                return locator
        except Exception:
            pass
    raise RuntimeError("Could not find the Excel button/link. Inspect the page and update the selector.")

def export_current_view(page, tag):
    excel = get_excel_button(page)
    with page.expect_download(timeout=60000) as download_info:
        excel.click()
    download = download_info.value
    suggested = download.suggested_filename
    ext = Path(suggested).suffix or ".xls"
    out_path = DOWNLOAD_DIR + f"{clean_filename(tag)}{ext}"
    download.save_as(str(out_path))
    return out_path

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        page.goto(START_URL, wait_until="domcontentloaded", timeout=120000)
        wait_for_page_ready(page)

        page.wait_for_timeout(5000)

        wait_for_page_ready(page)

        initial_file = export_current_view(page, "thru_games_initial")
        print(f"Downloaded initial export: {initial_file}")

        thru_games = get_thru_games_select(page)

        options = thru_games.locator("option").evaluate_all(
            """opts => opts.map(o => ({
                value: o.value,
                label: (o.textContent || '').trim()
            }))"""
        )

        seen = set()
        for opt in options:
            value = opt["value"]
            label = opt["label"] or value

            if not value or value in seen:
                continue
            seen.add(value)

            print(f"Selecting: {label} ({value})")

            thru_games.select_option(value=value)
            wait_for_page_ready(page)
            time.sleep(1.5)

            tag = f"thru_games_{label}"
            try:
                out_file = export_current_view(page, tag)
                print(f"Downloaded: {out_file}")
            except PlaywrightTimeoutError:
                print(f"Timed out downloading for option: {label}")
            except Exception as e:
                print(f"Failed for option {label}: {e}")

        context.close()
        browser.close()

if __name__ == "__main__":
    years = {
        '2023/' : '30968',
        '2022/' : '25023',
        '2021/' : '18704',
    }
    for key,val in years.items():
        START_URL = BASE_URL + val
        DOWNLOAD_DIR = BASE_DOWNLOAD_DIR + key
        main()