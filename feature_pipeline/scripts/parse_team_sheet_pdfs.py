import fitz  # PyMuPDF
import re
import os
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor

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

def get_spatial_team_name(words, page_height):
    header_words = [w for w in words if w['top'] < page_height * 0.12 and 
                    not any(x in w['text'].upper() for x in ["THROUGH", "GAMES", "OFFICIAL", "PAGE"]) 
                    and not re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', w['text'])]
    if not header_words: return None
    
    header_words.sort(key=lambda x: (x['top'], x['x0']))
    lines = []
    if not header_words: return None
    
    current_line = [header_words[0]]
    for i in range(1, len(header_words)):
        if abs(header_words[i]['top'] - header_words[i-1]['top']) < 3:
            current_line.append(header_words[i])
        else:
            lines.append(" ".join([w['text'] for w in current_line]))
            current_line = [header_words[i]]
    lines.append(" ".join([w['text'] for w in current_line]))
    
    clean = re.split(r'\(|:|Through Games|\(OFFICIAL\)', lines[0])[0].strip()
    if re.match(r"^(\w\s)+\w$", clean): clean = clean.replace(" ", "")
    return clean

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
            words = [
                {'x0': w[0], 'top': w[1], 'x1': w[2], 'bottom': w[3], 'text': w[4]}
                for w in raw_words
            ]
            
            full_text = " ".join([w['text'] for w in words])
            team_name = get_spatial_team_name(words, page.rect.height)
            if not team_name: continue
            
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

def clean_and_reconcile(year_key, raw_data, base_path):
    if not raw_data: return pd.DataFrame()
    df = pd.DataFrame(raw_data)
    cols_to_check = [c for c in df.columns if c != 'Team']
    df = df.dropna(subset=cols_to_check, how='all').reset_index(drop=True)
    
    # Reference path check
    ref_path = os.path.join(base_path, "team_sheets", f"{year_key}_Team_Sheets_Final.csv")
    if os.path.exists(ref_path):
        ref_df = pd.read_csv(ref_path)
        # Map names if counts match
        if len(df) <= len(ref_df):
            df['Team'] = ref_df.iloc[:len(df), 0].values
        else:
            logging.warning(f"Row mismatch in {year_key}: Parsed {len(df)} > Reference {len(ref_df)}")
    else:
        logging.warning(f"Reference CSV not found: {ref_path}")
    
    return df

# --- Main Execution ---

if __name__ == "__main__":
    pdf_dir = "/Users/michaelharoon/Projects/tasty/march-madness/data/raw/pdf/men/team_sheets/"
    base_proj_path = "/Users/michaelharoon/Projects/tasty/march-madness/data/"
    output_base_dir = "/Users/michaelharoon/Projects/tasty/march-madness/data/raw/test/"

    # 1. Collect all PDF paths recursively
    pdf_tasks = []
    for root, dirs, files in os.walk(pdf_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                full_path = os.path.join(root, f)
                # Keep the subfolder structure relative to the root pdf_dir
                rel_dir = os.path.relpath(root, pdf_dir)
                pdf_tasks.append((full_path, rel_dir))

    logging.info(f"Found {len(pdf_tasks)} PDF files to process.")

    # 2. Process in Parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_single_pdf, pdf_tasks))

    # 3. Reconcile and Save with structure
    for result in results:
        if result is None: continue
        year_key, raw_teams, rel_dir = result
        
        logging.info(f"Cleaning and saving: {year_key}")
        cleaned_df = clean_and_reconcile(year_key, raw_teams, base_proj_path)
        
        if not cleaned_df.empty:
            # Recreate subfolder structure in the output directory
            target_output_path = os.path.join(output_base_dir, rel_dir)
            os.makedirs(target_output_path, exist_ok=True)
            
            save_path = os.path.join(target_output_path, f"{year_key}_Cleaned.csv")
            cleaned_df.to_csv(save_path, index=False)
            logging.info(f"Saved: {save_path}")
        else:
            logging.warning(f"No valid data extracted for {year_key}")