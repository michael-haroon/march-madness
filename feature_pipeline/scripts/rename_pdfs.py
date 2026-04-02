# RENAME PDFs BY EXTRACTING DATES!
import re
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # PyMuPDF

BASE_DIR = Path("/Users/michaelharoon/Projects/tasty/march-madness/data/raw/rpi_archive")

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan\\.|Feb\\.|Mar\\.|Apr\\.|Jun\\.|Jul\\.|Aug\\.|Sep\\.|Sept\\.|Oct\\.|Nov\\.|Dec\\."
)

FULL_DATE_RE = re.compile(
    rf"\b(?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}\b",
    re.IGNORECASE
)

FINAL_YEAR_RE = re.compile(
    r"\bFinal\s+(20\d{2}|19\d{2})\b",
    re.IGNORECASE
)

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def normalize_date_string(raw: str) -> str:
    raw = normalize_spaces(raw)

    month_map = {
        "january": "01", "jan.": "01",
        "february": "02", "feb.": "02",
        "march": "03", "mar.": "03",
        "april": "04", "apr.": "04",
        "may": "05",
        "june": "06", "jun.": "06",
        "july": "07", "jul.": "07",
        "august": "08", "aug.": "08",
        "september": "09", "sep.": "09", "sept.": "09",
        "october": "10", "oct.": "10",
        "november": "11", "nov.": "11",
        "december": "12", "dec.": "12",
    }

    m = FULL_DATE_RE.search(raw)
    if m:
        token = m.group(0)
        parts = token.replace(",", "").split()
        month = month_map[parts[0].lower()]
        day = f"{int(parts[1]):02d}"
        year = parts[2]
        return f"{year}-{month}-{day}"

    m = FINAL_YEAR_RE.search(raw)
    if m:
        return f"Final_{m.group(1)}"

    return None

def make_unique_path(path: Path) -> Path:
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

def extract_candidate_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    try:
        if doc.page_count == 0:
            return ""
        page = doc[0]

        # Fast whole-page text.
        full_text = page.get_text("text")

        # Also try top-right-ish text blocks, since you said date is often there.
        blocks = page.get_text("blocks")
        page_width = page.rect.width
        top_blocks = []
        for b in blocks:
            x0, y0, x1, y1, text = b[:5]
            if y0 < 160 and x0 > page_width * 0.45:
                top_blocks.append((y0, x0, text))

        top_blocks.sort()
        top_right_text = "\n".join(t[2] for t in top_blocks if t[2])

        combined = "\n".join([
            top_right_text,
            "\n".join(full_text.splitlines()[:12])
        ])
        return combined
    finally:
        doc.close()

def process_pdf(pdf_path_str: str):
    pdf_path = Path(pdf_path_str)
    try:
        text = extract_candidate_text(pdf_path)
        text = normalize_spaces(text)

        normalized_date = normalize_date_string(text)
        if not normalized_date:
            return {
                "file": str(pdf_path),
                "status": "no_date_found",
                "new_name": None,
            }

        new_name = f"{normalized_date}_Team_Sheets.pdf"
        new_path = make_unique_path(pdf_path.with_name(new_name))

        if pdf_path.resolve() != new_path.resolve():
            pdf_path.rename(new_path)

        return {
            "file": str(pdf_path),
            "status": "renamed",
            "new_name": str(new_path),
        }

    except Exception as e:
        return {
            "file": str(pdf_path),
            "status": f"error: {e}",
            "new_name": None,
        }

def main():
    pdfs = sorted(BASE_DIR.glob("Team_Sheets*.pdf"))
    print(f"Found {len(pdfs)} PDFs to process")

    max_workers = min(os.cpu_count() or 4, 12)
    print(f"Using {max_workers} workers")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_pdf, str(pdf)) for pdf in pdfs]
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            print(result)

    renamed = sum(r["status"] == "renamed" for r in results)
    skipped = sum(r["status"] == "no_date_found" for r in results)
    errored = len(results) - renamed - skipped

    print(f"\nRenamed: {renamed}")
    print(f"Skipped (no date found): {skipped}")
    print(f"Errors: {errored}")

if __name__ == "__main__":
    main()