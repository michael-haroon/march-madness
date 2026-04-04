import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from playwright.sync_api import sync_playwright

START_URL = "https://s3.amazonaws.com/ncaa/files/rpiarchive/list.html"
BASE_URL = "https://s3.amazonaws.com/ncaa/files/rpiarchive/"
DOWNLOAD_DIR = Path("/Users/michaelharoon/Projects/tasty/march-madness/data/raw/rpi_archive")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
}

def clean_name(name: str) -> str:
    name = re.sub(r'[\\\\/:*?\"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name

def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 2
    while True:
        candidate = path.with_name(f"{stem} ({i}){suffix}")
        if not candidate.exists():
            return candidate
        i += 1

def normalize_date(date_str: str) -> str:
    parts = date_str.strip().split("/")
    if len(parts) != 3:
        return clean_name(date_str)
    m, d, y = parts
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

def build_filename(through_date: str, ext: str, report_type: str = "Team_Sheets") -> str:
    date_part = normalize_date(through_date)
    ext = ext if ext.startswith(".") else f".{ext}"
    return clean_name(f"{date_part}_{report_type}") + ext

def looks_like_html(data: bytes) -> bool:
    sample = data[:400].lstrip().lower()
    return (
        sample.startswith(b"<!doctype html")
        or sample.startswith(b"<html")
        or b"<body" in sample
    )

def looks_like_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF")

def looks_like_xlsx(data: bytes) -> bool:
    return data.startswith(b"PK\x03\x04")

def looks_like_xls_ole(data: bytes) -> bool:
    return data.startswith(bytes.fromhex("D0CF11E0A1B11AE1"))

def download_file(item: dict):
    url = item["url"]
    through_date = item["through_date"]
    document_type = item["document_type"]

    ext = Path(url).suffix.lower() or ".pdf"
    if "team sheets" in document_type.lower():
        report_stub = "Team_Sheets"
    else:
        report_stub = clean_name(document_type)

    filename = build_filename(through_date, ext, report_stub)
    out_path = unique_path(DOWNLOAD_DIR / filename)

    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
            r.raise_for_status()
            chunks = []
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    chunks.append(chunk)
            data = b"".join(chunks)

        if not data:
            print(f"Empty response: {url}")
            return

        if looks_like_html(data):
            print(f"Skipped HTML response: {url}")
            return

        suffix = out_path.suffix.lower()
        if suffix == ".pdf" and not looks_like_pdf(data):
            print(f"Skipped invalid PDF payload: {url}")
            return
        if suffix == ".xlsx" and not looks_like_xlsx(data):
            print(f"Skipped invalid XLSX payload: {url}")
            return
        if suffix == ".xls" and not (looks_like_xls_ole(data) or looks_like_xlsx(data)):
            print(f"Skipped invalid XLS payload: {url}")
            return

        out_path.write_bytes(data)
        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Failed: {url} -> {e}")

def wait_for_filters(page):
    page.wait_for_selector("#rpi tfoot select", timeout=120000)

def get_footer_select(page, index_zero_based: int):
    return page.locator("#rpi tfoot select").nth(index_zero_based)

def select_by_text(locator, text_value: str):
    locator.select_option(label=text_value)

def apply_filters(page):
    wait_for_filters(page)

    # Footer dropdown order:
    # 0 Division, 1 Sport, 2 Gender, 3 Year, 4 Report Type, 5 Document Type
    select_by_text(get_footer_select(page, 0), "I")
    page.wait_for_timeout(500)

    select_by_text(get_footer_select(page, 1), "Basketball")
    page.wait_for_timeout(500)

    select_by_text(get_footer_select(page, 2), "Men")
    page.wait_for_timeout(500)

    # Leave Year blank
    # Leave Report Type blank

    select_by_text(get_footer_select(page, 5), "Team Sheets")
    page.wait_for_timeout(1500)

def current_page_number(page):
    candidates = [
        ".dataTables_paginate .paginate_button.current",
        "span.current",
        ".pagination .active a",
        ".pagination .active span",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                return loc.inner_text().strip()
        except Exception:
            pass
    return "unknown"

def get_download_items_on_current_page(page):
    rows = page.locator("#rpi tbody tr")
    count = rows.count()
    items = []

    for i in range(count):
        row = rows.nth(i)
        cells = row.locator("td")
        if cells.count() < 8:
            continue

        division = cells.nth(0).inner_text().strip()
        sport = cells.nth(1).inner_text().strip()
        gender = cells.nth(2).inner_text().strip()
        year = cells.nth(3).inner_text().strip()
        report_type = cells.nth(4).inner_text().strip()
        document_type = cells.nth(5).inner_text().strip()
        through_date = cells.nth(6).inner_text().strip()

        link = cells.nth(7).locator("a").first
        href = link.get_attribute("href")
        if not href:
            continue

        items.append({
            "url": urljoin(BASE_URL, href),
            "division": division,
            "sport": sport,
            "gender": gender,
            "year": year,
            "report_type": report_type,
            "document_type": document_type,
            "through_date": through_date,
        })

    return items

def click_next(page):
    next_selectors = [
        "a#rpi_next",
        "a.paginate_button.next",
        "li.paginate_button.next a",
        "text=Next",
    ]
    for sel in next_selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() == 0:
                continue

            classes = (loc.get_attribute("class") or "").lower()
            aria_disabled = (loc.get_attribute("aria-disabled") or "").lower()

            parent_class = ""
            try:
                parent_class = (loc.locator("xpath=..").get_attribute("class") or "").lower()
            except Exception:
                pass

            disabled = (
                "disabled" in classes
                or "disabled" in parent_class
                or aria_disabled == "true"
            )
            if disabled:
                return False

            before = current_page_number(page)
            loc.click()
            page.wait_for_timeout(1200)
            after = current_page_number(page)

            if before == after:
                page.wait_for_timeout(1200)

            return True
        except Exception:
            pass
    return False

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        page.goto(START_URL, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_selector("#rpi", timeout=120000)

        apply_filters(page)

        visited_pages = set()
        downloaded_keys = set()

        while True:
            page_num = current_page_number(page)
            print(f"Processing filtered page: {page_num}")

            if page_num in visited_pages:
                print("Already visited this page; stopping to avoid loop.")
                break
            visited_pages.add(page_num)

            page.wait_for_selector("#rpi tbody tr", timeout=120000)
            items = get_download_items_on_current_page(page)
            print(f"Found {len(items)} matching files on page {page_num}")

            for item in items:
                dedupe_key = (item["url"], item["through_date"], item["document_type"])
                if dedupe_key in downloaded_keys:
                    continue
                downloaded_keys.add(dedupe_key)
                download_file(item)

            moved = click_next(page)
            if not moved:
                print("No more filtered pages.")
                break

        browser.close()

if __name__ == "__main__":
    main()