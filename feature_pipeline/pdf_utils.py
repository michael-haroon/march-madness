"""
feature_pipeline/pdf_utils.py
------------------------------
Shared PDF extraction utilities used by both parse_team_sheet_pdfs.py
and resolve_ambiguous_teams.py.
"""

import re
import fitz  # PyMuPDF


_SKIP_WORDS = {"THROUGH", "GAMES", "OFFICIAL", "PAGE"}
_SPACED_CHARS = re.compile(r"^([A-Za-z&.'] [A-Za-z&.'])")


def get_spatial_team_name(words: list[dict], page_height: float) -> str | None:
    """
    Extract team name from the top-left header of a PDF page using word
    coordinates. Works for all observed PDF layouts (2005–2026).

    The team name is always on the LEFT side of the page (x < 300).
    The date/OFFICIAL header is on the RIGHT side (x > 400) and is ignored.

    words: list of dicts with keys x0, top, x1, bottom, text
           (from page.get_text("words") mapped to dicts)
    page_height: page.rect.height
    """
    # Team name is always on the LEFT side of the page.
    # The "(OFFICIAL) Through Games Of DATE" header is on the right (x > 480).
    # For some PDFs (e.g. 2018+) the team name is at y≈0, x≈27.
    # For other PDFs (e.g. 2015) the team name is spaced-out chars at y≈30, x=205-385.
    # Use x < 450 to capture all formats while excluding the right-side header.
    header_words = [
        w for w in words
        if w["top"] < 8
        and w["x0"] < 450
        and not any(x in w["text"].upper() for x in _SKIP_WORDS)
        and not re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", w["text"])
    ]
    if not header_words:
        # Fallback for PDFs where team name is lower on the page (e.g. y≈30)
        header_words = [
            w for w in words
            if w["top"] < page_height * 0.12
            and w["x0"] < 450
            and not any(x in w["text"].upper() for x in _SKIP_WORDS)
            and not re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", w["text"])
        ]
    if not header_words:
        return None

    header_words.sort(key=lambda w: (w["top"], w["x0"]))

    # Build first line (words within 3pt of same y)
    current_line = [header_words[0]]
    for i in range(1, len(header_words)):
        if abs(header_words[i]["top"] - header_words[i - 1]["top"]) < 3:
            current_line.append(header_words[i])
        else:
            break

    raw = " ".join(w["text"] for w in current_line)
    clean = re.split(r"\(|:|Through Games|\(OFFICIAL\)", raw)[0].strip()

    # Compact spaced-out characters: "V i r g i n i a" → "Virginia"
    if _SPACED_CHARS.match(clean):
        clean = clean.replace(" ", "")

    return clean if clean else None


def page_words_to_dicts(raw_words: list) -> list[dict]:
    """
    Convert PyMuPDF word tuples (x0, y0, x1, y1, text, ...) to the dict
    format expected by get_spatial_team_name().
    """
    return [
        {"x0": w[0], "top": w[1], "x1": w[2], "bottom": w[3], "text": w[4]}
        for w in raw_words
    ]


def _extract_spaced_chars(text: str) -> str | None:
    """
    Some PDF pages encode the team name as individual characters on separate lines:
      'S\\na\\ni\\nn\\nt\\nF\\nr\\na\\nn\\nc\\ni\\ns\\n(\\nP\\nA\\n)'  →  'Saint Francis (PA)'
    Detect and reconstruct such names.

    Picks the FIRST run of 4+ single-char lines that starts with an uppercase letter
    followed by a lowercase letter — this is the proper-name signature.
    Falls back to the longest run if no proper-name run is found.
    """
    lines = text.split("\n")

    # Collect all runs of single-char lines (length >= 4)
    all_runs: list[list[str]] = []
    run: list[str] = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) == 1:
            run.append(stripped)
        else:
            if len(run) >= 4:
                all_runs.append(list(run))
            run = []
    if len(run) >= 4:
        all_runs.append(run)

    if not all_runs:
        return None

    # Prefer a run whose first half equals its second half AND the half is
    # not purely H/A/N/S location codes.
    # The team name is encoded twice: "BaylorBaylor", "BYUBYU", "SanDiegoSt.SanDiegoSt."
    # H/A/N location indicators ("NANAS", "HHANNAS") also sometimes repeat, but
    # are caught by the length-3 minimum and all-HAN exclusion.
    _han = set("HANS")
    best_run = None
    for r in all_runs:
        mid = len(r) // 2
        half = r[:mid]
        if (mid >= 3
                and r[mid:mid + mid] == half
                and not all(c in _han for c in half)):
            best_run = r
            break

    if best_run is None:
        # Fallback: first run not purely H/A/N/S
        for r in all_runs:
            if not all(c in _han for c in r):
                best_run = r
                break

    if best_run is None:
        best_run = max(all_runs, key=len)

    max_run = best_run
    if len(max_run) < 4:
        return None
    joined = "".join(max_run)
    # "SaintFrancis(PA)" → insert spaces at case boundaries and before "("
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", joined)      # camelCase split
    name = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", name)  # "LIUBrooklyn" → "LIU Brooklyn"
    name = re.sub(r"\.(?=[A-Za-z])", ". ", name)            # "Mt.St." → "Mt. St."
    name = re.sub(r"([A-Za-z])\(", r"\1 (", name)
    name = re.sub(r"\)([A-Za-z])", r") \1", name)
    name = name.strip()
    # Deduplicate: "Saint Francis (PA) Saint Francis (PA)" → "Saint Francis (PA)"
    mid = len(name) // 2
    if name[:mid].strip() == name[mid:].strip():
        name = name[:mid].strip()
    elif " " in name:
        # Try splitting on repeated half
        half = name[: (len(name) + 1) // 2].strip()
        if name.startswith(half) and name[len(half):].strip() == half:
            name = half
    return name if re.match(r"^[A-Za-z]", name) else None


def extract_team_name_from_page(page: fitz.Page) -> str | None:
    """
    Extract the team name from a single PDF page.
    Tries spatial extraction first; falls back to spaced-character reconstruction
    for pages where the team name is encoded character-by-character.
    Single authoritative implementation — used by both parse and resolve scripts.
    """
    # Patterns that indicate the spatial result is a column header, not a team name
    _NOT_TEAM = re.compile(
        r"^(NET|SOS|RPI|KPI|BPI|SOR|POM|SAG|WON|HOME|AWAY|ROAD|RECORD|LOSS|NON)\b"
        r"|Opponent|Score\s*$"
        r"|^Of\s+\w+[\s,]+\d+,?\s*$"  # "Of March 1," with no team name after
        r"|\d$",                        # ends with digit — ranking bled into compacted name
        re.IGNORECASE
    )

    raw_words = page.get_text("words")
    if raw_words:
        words = page_words_to_dicts(raw_words)
        name = get_spatial_team_name(words, page.rect.height)
        if name and not _NOT_TEAM.search(name):
            return name

    # Fallback: some PDFs store the team name as individual characters per line
    # (e.g. "S\na\ni\nn\nt\nF\nr\na\nn\nc\ni\ns\n" → "Saint Francis")
    text = page.get_text()
    return _extract_spaced_chars(text)


def _team_name_text_based(page: fitz.Page) -> str | None:
    """
    Text-based extraction that preserves state abbreviations like (FL), (OH), (PA).
    Used ONLY for disambiguation — do not use for the main name list.
    Splits on '(NET:' / '(RPI:' but NOT on '(FL)' or '(OH)'.
    """
    text = page.get_text()
    if not text:
        return None
    m = re.search(r"GmDte\n(.+?)\n", text)
    if m:
        cand = m.group(1).strip()
        if not re.match(r"^\d", cand):
            return cand
    # Capture everything up to "(NET:" or "(RPI:" — preserves "(FL)", "(OH)" etc.
    m = re.search(r"\n(.+?) \((?:NET|RPI|KPI|BPI|POM|SAG|SOR):", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"\n(.+?) \d+[-\u2013]\s*\d+\s+", text)
    if m:
        cand = m.group(1).strip()
        if re.match(r"^[A-Za-z]", cand):
            return cand
    return None


def load_pdf_team_names_for_disambiguation(pdf_path: str) -> dict[int, str]:
    """
    Like load_pdf_team_names() but preserves state abbreviations in names
    (e.g. 'Miami (FL)', 'Miami (OH)', 'Saint Francis (PA)') so that
    resolve_via_pdf can distinguish ambiguous teams.
    Falls back to spatial extraction if text patterns return nothing.
    """
    import os
    result: dict[int, str] = {}
    if not os.path.exists(pdf_path):
        return result
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    [WARN] cannot open {pdf_path}: {e}")
        return result
    for pg in range(1, len(doc) + 1):
        page = doc[pg - 1]
        name = _team_name_text_based(page)
        if not name:
            name = extract_team_name_from_page(page)
        if name:
            result[pg] = name
    doc.close()
    return result


def load_pdf_team_names(pdf_path: str) -> dict[int, str]:
    """
    Return {page_number (1-based): team_name} for every page in the PDF
    where a team name can be extracted.
    """
    import os
    result: dict[int, str] = {}
    if not os.path.exists(pdf_path):
        return result
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    [WARN] cannot open {pdf_path}: {e}")
        return result
    for pg in range(1, len(doc) + 1):
        name = extract_team_name_from_page(doc[pg - 1])
        if name:
            result[pg] = name
    doc.close()
    return result
