import fitz  # PyMuPDF
import glob
import re
import os
import sys
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from feature_pipeline.pdf_utils import get_spatial_team_name, page_words_to_dicts

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pdf_processing.log")
    ]
)

# --- Spatial Functions (Updated for compatibility) ---

def extract_spatial_value(words, anchor_texts, x_tolerance=30, index=0, exclude_text=None):
    if isinstance(anchor_texts, str):
        anchor_texts = [anchor_texts]
    
    candidates = [w for w in words if any(t.upper() in w['text'].upper() for t in anchor_texts)]
    valid_anchors = []
    
    for cand in candidates:
        if exclude_text:
            preceded_by_excluded = any(
                exclude_text.upper() in w['text'].upper() 
                and abs(w['top'] - cand['top']) < 5
                and 0 < (cand['x0'] - w['x1']) < 80
                for w in words
            )
            if preceded_by_excluded: continue
        if "OPP." in cand['text'].upper() and "OPPONENT" in cand['text'].upper(): continue
        valid_anchors.append(cand)
        
    if not valid_anchors: return None
    valid_anchors.sort(key=lambda x: (x['top'], x['x0']))
    anchor = valid_anchors[0]
    
    # Extract digits in the same vertical column
    column_digits = [w for w in words if abs(w['x0'] - anchor['x0']) < x_tolerance and w['top'] > anchor['bottom'] and re.match(r'^\d+$', w['text'])]
    column_digits.sort(key=lambda x: x['top'])
    return column_digits[index]['text'] if len(column_digits) > index else None

def get_header_metric(words, label_text):
    label_anchor = next((w for w in words if label_text.upper() in w['text'].upper()), None)
    if not label_anchor: return None
    
    candidates = [w for w in words if re.match(r'^\d+$', w['text']) and (
        (abs(w['top'] - label_anchor['top']) < 10 and w['x0'] > label_anchor['x1']) or 
        (abs(w['x0'] - label_anchor['x0']) < 30 and w['top'] > label_anchor['bottom'])
    )]
    candidates.sort(key=lambda x: (x['top'], x['x0']))
    return candidates[0]['text'] if candidates else None

# get_spatial_team_name is imported from feature_pipeline.pdf_utils

# --- Processing Logic ---

def process_single_pdf(file_info):
    """
    file_info: tuple (full_pdf_path, relative_dir_path)
    """
    pdf_path, rel_dir = file_info
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    all_teams = []
    
    try:
        doc = fitz.open(pdf_path)
        logging.info(f"Processing: {pdf_path}")
        
        for page_num, page in enumerate(doc):
            # PyMuPDF word tuple: (x0, y0, x1, y1, "text", block_no, line_no, word_no)
            # Map to dict to maintain compatibility with original logic
            raw_words = page.get_text("words")
            words = page_words_to_dicts(raw_words)
            
            full_text = " ".join([w['text'] for w in words])
            team_name = get_spatial_team_name(words, page.rect.height)
            if not team_name:
                logging.debug(f"Skipped page {page_num + 1} in {file_name}: no team name found")
                continue
            
            record_match = re.search(r"(\d{1,2}\s*-\s*\d{1,2})", full_text)
            metrics = {
                "Team": team_name,
                "Record": record_match.group(1).replace(" ", "") if record_match else None,
                "Avg_RPI_Win": get_header_metric(words, "Average RPI Win"),
                "Avg_RPI_Loss": get_header_metric(words, "Average RPI Loss"),
                "RPI_Rank_D1": extract_spatial_value(words, "TEAM", index=0),
                "RPI_Rank_NonConf": extract_spatial_value(words, "TEAM", index=1),
                "SOS_D1": extract_spatial_value(words, ["SUCCESS", "STRENGTH"], index=0, exclude_text="OPP"),
                "SOS_NonConf": extract_spatial_value(words, ["SUCCESS", "STRENGTH"], index=1, exclude_text="OPP"),
                "Opp_SOS_D1": extract_spatial_value(words, "OPP.", index=0),
                "Opp_SOS_NonConf": extract_spatial_value(words, "OPP.", index=1),
            }
            
            # Rank Metric Extraction
            for key, pattern in [("NET", r"NET:\s*(\d+)"), ("KPI", r"KPI:\s*(\d+)"), 
                                 ("SOR", r"SOR:\s*(\d+)"), ("POM", r"POM:\s*(\d+)"), 
                                 ("SAG", r"SAG:\s*(\d+)"), ("BPI", r"BPI:\s*(\d+)")]:
                match = re.search(pattern, full_text)
                metrics[key] = match.group(1) if match else None
            
            all_teams.append(metrics)
        doc.close()
        
        return file_name, all_teams, rel_dir

    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}")
        return None

def clean_and_reconcile(year_key, raw_data, base_path, rel_dir):
    if not raw_data: return pd.DataFrame()
    df = pd.DataFrame(raw_data)
    cols_to_check = [c for c in df.columns if c != 'Team']
    df = df.dropna(subset=cols_to_check, how='all').reset_index(drop=True)

    # For years ≤ 2014, all files within a year are consistently alphabetically ordered.
    # Use any CSV from rpi_archive_copy/{year}/ as a 1-to-1 row-position reference for
    # clean team names, since the copy has the same alphabetical ordering.
    folder_year = None
    for part in rel_dir.replace("\\", "/").split("/"):
        if part.isdigit():
            folder_year = int(part)
            break

    if folder_year is not None and folder_year <= 2014:
        copy_dir = os.path.join(base_path, "raw", "test", "rpi_archive copy", str(folder_year))
        ref_csvs = sorted(glob.glob(os.path.join(copy_dir, "*.csv")))
        if ref_csvs:
            ref_df = pd.read_csv(ref_csvs[0])
            if "Team" in ref_df.columns:
                # Extract clean name from raw Team column values in the reference.
                # Handles both observed formats:
                #   Suffix: "TEAM Of Final YYYY" → take before " Of "
                #   Prefix: "Of DOW, Month D, YYYY TEAM [W-L SYS]" → strip prefix/record
                _record = re.compile(r"\s+\d+-\d+\s+[A-Za-z]{2,}$")
                def _ref_name(s: str) -> str:
                    s = str(s).strip()
                    if " Of " in s and not s.startswith("Of "):
                        return s.split(" Of ")[0].strip()
                    if s.startswith("Of "):
                        rest = s[3:]
                        # "DOW, Month D, YYYY TEAM" or "Month D, YYYY TEAM"
                        m = re.match(r'^(?:\w+,\s+)?(?:\w+\.?\s+\d+,?\s+\d{4})\s+(.+)$', rest)
                        if m:
                            return _record.sub("", m.group(1)).strip()
                        # "Final YYYY TEAM" or "YYYY Final TEAM"
                        m = re.match(r'^(?:Final\s+\d{4}|\d{4}\s+Final)\s+(.+)$', rest)
                        if m:
                            return _record.sub("", m.group(1)).strip()
                    return _record.sub("", s).strip()
                ref_names = [_ref_name(v) for v in ref_df["Team"].astype(str)]
                if len(ref_names) >= len(df):
                    df["Team"] = ref_names[:len(df)]
                else:
                    logging.warning(
                        f"rpi_archive_copy reference too short for {year_key}: "
                        f"copy has {len(ref_names)} rows, parsed has {len(df)}"
                    )
        else:
            logging.warning(f"No reference CSV found in rpi_archive copy for year {folder_year}")

    return df

# --- Main Execution ---

if __name__ == "__main__":
    # Scoped to rpi_archive only — do not walk the full men/team_sheets/ tree
    pdf_dir = "/Users/michaelharoon/Projects/tasty/march-madness/data/raw/pdf/men/team_sheets/rpi_archive/"
    base_proj_path = "/Users/michaelharoon/Projects/tasty/march-madness/data/"
    output_base_dir = "/Users/michaelharoon/Projects/tasty/march-madness/data/raw/test/rpi_archive/"

    # 1. Collect all PDF paths recursively
    pdf_tasks = []
    for root, dirs, files in os.walk(pdf_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                full_path = os.path.join(root, f)
                # Keep the subfolder structure relative to the rpi_archive root
                rel_dir = os.path.relpath(root, pdf_dir)
                pdf_tasks.append((full_path, rel_dir))

    logging.info(f"Found {len(pdf_tasks)} PDF files to process.")

    # 2. Process in Parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_single_pdf, pdf_tasks))

    skipped = 0
    saved = 0

    # 3. Reconcile and Save with structure
    for result in results:
        if result is None:
            skipped += 1
            continue
        year_key, raw_teams, rel_dir = result

        logging.info(f"Cleaning and saving: {year_key}")
        cleaned_df = clean_and_reconcile(year_key, raw_teams, base_proj_path, rel_dir)

        if not cleaned_df.empty:
            target_output_path = os.path.join(output_base_dir, rel_dir)
            os.makedirs(target_output_path, exist_ok=True)

            save_path = os.path.join(target_output_path, f"{year_key}_Cleaned.csv")
            cleaned_df.to_csv(save_path, index=False)
            logging.info(f"Saved: {save_path}")
            saved += 1
        else:
            logging.warning(f"No valid data extracted for {year_key}")
            skipped += 1

    logging.info(
        f"\nSummary: {saved} CSVs saved, {skipped} PDFs skipped/empty "
        f"(out of {len(pdf_tasks)} total)"
    )