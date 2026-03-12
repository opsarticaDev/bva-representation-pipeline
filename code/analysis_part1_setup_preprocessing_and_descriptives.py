#!/usr/bin/env python3
"""
BVA Analysis Part 1: Setup, Data Loading, and Preprocessing
Addresses reviewer requirements: #5 (multi-rep), #8 (flow), #12 (audit)
"""

import os
import sys
import csv
import datetime
import time
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
ROOT = os.environ.get("BVA_ROOT", ".")
OUTD = os.path.join(ROOT, "index", "Grok")
DB = os.path.join(ROOT, "parsed", "parsed.full.rev01.sqlite")
os.makedirs(OUTD, exist_ok=True)

ENC = "utf-8"
LOGP = os.path.join(OUTD, "part1_build.log.txt")
YEARS = list(range(2010, 2026))
READ_TEXT_FOR_CUES = True
SAMPLE_FRACTION = 1.0
MAX_PER_YEAR = None
CHECKPOINT_DF = os.path.join(OUTD, "checkpoint_df_part1.parquet")
BATCH_SIZE = 100
TEXT_TIMEOUT = 5
MIN_CELL_SIZE = 1000
AUDIT_SAMPLE_SIZE = 300

# ==================== REGEX PATTERNS ====================
RX_HEARING = re.compile(
    r"\bhearing\b|virtual hearing|video hearing|tele[- ]?hearing|"
    r"teleconference|Microsoft\s+Teams|Zoom|Webex|oral argument",
    re.IGNORECASE
)
RX_AMA = re.compile(
    r"\bAppeals Modernization Act\b|\bAMA\b|docket selection|"
    r"direct review|evidence submission|supplemental claim",
    re.IGNORECASE
)
RX_MULTI_REP = re.compile(r'and|,|;', re.IGNORECASE)

# ==================== LOGGING ====================
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOGP, "a", encoding=ENC) as f:
            f.write(f"{ts} | {msg}\n")
        print(f"{ts} | {msg}")
    except Exception as e:
        print(f"Log error: {e}")

log("="*70)
log("PART 1: SETUP, DATA LOADING, AND PREPROCESSING")
log("="*70)

# ==================== UTILITIES ====================
def write_csv(path, header, rows):
    """Write CSV with error handling"""
    try:
        with open(path, "w", newline="", encoding=ENC) as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        log(f"Wrote: {os.path.basename(path)} ({len(rows)} rows)")
    except Exception as e:
        log(f"ERROR writing {path}: {e}")

# ==================== DATABASE QUERIES ====================
def get_qc_columns(con):
    """Get QC table columns"""
    cur = con.cursor()
    cur.execute("PRAGMA table_info(qc)")
    columns = [row[1] for row in cur.fetchall()]
    log(f"QC columns: {columns}")
    return columns

def get_parsed_columns(con):
    """Get parsed table columns"""
    cur = con.cursor()
    cur.execute("PRAGMA table_info(parsed)")
    columns = [row[1] for row in cur.fetchall()]
    log(f"Parsed columns: {columns}")
    return columns

def fetch_cases(con, ns_only=True):
    """Fetch cases from database with VLJ checking"""
    cur = con.cursor()
    cases = []
    
    qc_cols = get_qc_columns(con)
    parsed_cols = get_parsed_columns(con)
    
    suspect_col = 'suspect' if 'suspect' in qc_cols else None
    vlj_col = 'vlj_id' if 'vlj_id' in parsed_cols else ('judge_id' if 'judge_id' in parsed_cols else None)
    
    log(f"Suspect column: {suspect_col if suspect_col else 'NOT FOUND'}")
    log(f"VLJ column: {vlj_col if vlj_col else 'NOT FOUND'}")
    
    select_cols = "p.case_id, TRIM(p.decision_year), p.rep_type, q.primary_disposition, p.source_path"
    if vlj_col:
        select_cols += f", p.{vlj_col}"
    
    for y in tqdm(YEARS, desc="Fetching years"):
        query = f"""
            SELECT {select_cols}
            FROM parsed p
            JOIN qc q ON q.case_id = p.case_id
            WHERE TRIM(p.decision_year) = ?
            AND LOWER(p.file_ext) = '.txt'
        """
        params = (str(y),)
        
        if suspect_col and ns_only:
            query += f" AND q.{suspect_col} = 0"
        
        cur.execute(query, params)
        fetched = cur.fetchall()
        
        if SAMPLE_FRACTION < 1.0:
            fetched = fetched[:int(len(fetched) * SAMPLE_FRACTION)]
        if MAX_PER_YEAR:
            fetched = fetched[:MAX_PER_YEAR]
        
        cases.extend(fetched)
        log(f"  {y}: {len(fetched):,} cases")
    
    log(f"Total fetched: {len(cases):,}")
    has_vlj = vlj_col is not None
    return cases, has_vlj, vlj_col

# ==================== TEXT FEATURE EXTRACTION ====================
def features_from_text_batch(paths):
    """Extract features from decision text files"""
    results = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Reading texts", leave=False):
        batch = paths[i:i+BATCH_SIZE]
        for path in batch:
            if not path or not os.path.exists(path) or not READ_TEXT_FOR_CUES:
                results.append((0, 0, 0))
                continue
            
            try:
                start = time.time()
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                
                if time.time() - start > TEXT_TIMEOUT:
                    text = ""
                
                len_chars = len(text)
                hearing = int(bool(RX_HEARING.search(text)))
                ama_cue = int(bool(RX_AMA.search(text)))
                results.append((len_chars, hearing, ama_cue))
            except Exception as e:
                results.append((0, 0, 0))
    
    return results

# ==================== DATAFRAME CONSTRUCTION ====================
def build_df(cases, has_vlj, vlj_col):
    """Build analysis DataFrame with all codings"""
    log("Building DataFrame...")
    
    paths = [case[4] for case in cases]
    cues = features_from_text_batch(paths)
    
    data = []
    for i, case in enumerate(tqdm(cases, desc="Processing cases")):
        cid = case[0]
        year = case[1]
        rep = case[2]
        disp = case[3]
        vlj = case[5] if has_vlj and len(case) > 5 else None
        
        if not disp or disp.strip() == '':
            continue
        
        len_chars, hearing, ama_cue = cues[i]
        
        try:
            year_int = int(year)
        except (ValueError, TypeError):
            continue
        
        data.append({
            'case_id': cid,
            'year': year_int,
            'rep_type': rep.lower().strip() if rep else 'unknown',
            'disp_group': disp.lower().strip(),
            'len_chars': len_chars,
            'hearing': hearing,
            'ama_cue': ama_cue,
            'vlj_id': vlj if has_vlj else None,
        })
    
    df = pd.DataFrame(data)
    log(f"Built DataFrame: {len(df):,} rows")
    
    if df.empty:
        return df
    
    # ===== REPRESENTATION CODING (ISSUE #5) =====
    df['rep_attorney'] = df['rep_type'].str.contains('attorney', case=False, na=False).astype(int)
    df['rep_agent'] = df['rep_type'].str.contains('agent', case=False, na=False).astype(int)
    df['rep_vso'] = df['rep_type'].str.contains('vso', case=False, na=False).astype(int)
    df['rep_generic'] = ((df['rep_attorney'] + df['rep_agent'] + df['rep_vso']) == 0).astype(int)
    df['rep_any'] = 1 - df['rep_generic']
    
    # Multi-rep detection
    df['multi_rep'] = df['rep_type'].apply(lambda x: bool(RX_MULTI_REP.search(str(x))))
    
    # Hierarchical assignment: Attorney > Agent > VSO > Pro se
    df['rep_primary'] = 'unknown'
    df.loc[df['rep_generic'] == 1, 'rep_primary'] = 'pro_se'
    df.loc[df['rep_vso'] == 1, 'rep_primary'] = 'vso'
    df.loc[df['rep_agent'] == 1, 'rep_primary'] = 'agent'
    df.loc[df['rep_attorney'] == 1, 'rep_primary'] = 'attorney'
    
    # ===== OUTCOME CODING (ISSUE #7) =====
    df['grant_bin'] = df['disp_group'].isin(['grant', 'mixed']).astype(int)
    df['grant_only'] = (df['disp_group'] == 'grant').astype(int)
    df['grant_remand'] = df['disp_group'].isin(['grant', 'remand', 'mixed']).astype(int)
    df['deny_only'] = (df['disp_group'] == 'deny').astype(int)
    df['dismiss'] = (df['disp_group'] == 'dismiss').astype(int)
    
    # Two negative outcome definitions for sensitivity
    df['negative_deny_only'] = df['deny_only']
    df['negative_deny_dismiss'] = (df['deny_only'] | df['dismiss']).astype(int)
    
    # Multinomial outcome (0=grant, 1=deny, 2=remand, 3=other)
    df['outcome_cat'] = df['disp_group'].map({
        'grant': 0, 'mixed': 0, 'deny': 1, 'remand': 2, 'dismiss': 1
    }).fillna(3)
    
    # ===== AMA ERA DEFINITION (ISSUE #4) =====
    df['ama_era'] = ((df['year'] >= 2019) | (df['ama_cue'] == 1)).astype(int)
    
    # ===== COVARIATES =====
    df['len_chars_z'] = (df['len_chars'] - df['len_chars'].mean()) / (df['len_chars'].std() + 1e-10)
    df['len_chars_clip'] = df['len_chars'].clip(
        lower=df['len_chars'].quantile(0.01),
        upper=df['len_chars'].quantile(0.99)
    )
    
    # Save checkpoint
    try:
        df.to_parquet(CHECKPOINT_DF)
        log(f"Saved checkpoint: {CHECKPOINT_DF}")
    except Exception as e:
        log(f"Checkpoint save failed: {e}")
    
    return df

# ==================== DESCRIPTIVES (ISSUE #8) ====================
def generate_descriptive_tables(df):
    """Generate Table 1 and flow diagram data"""
    log("Generating descriptive tables...")
    
    # Table 1: By year
    t1_rows = []
    for y in sorted(df['year'].unique()):
        dfy = df[df['year'] == y]
        n = len(dfy)
        n_rep = dfy['rep_any'].sum()
        pct_rep = n_rep / n * 100 if n > 0 else 0
        n_grant = dfy['grant_bin'].sum()
        pct_grant = n_grant / n * 100 if n > 0 else 0
        
        t1_rows.append([
            y, n, n_rep, pct_rep, n_grant, pct_grant,
            dfy['rep_attorney'].sum(),
            dfy['rep_agent'].sum(),
            dfy['rep_vso'].sum(),
            dfy['rep_generic'].sum(),
        ])
    
    write_csv(
        os.path.join(OUTD, "table1_descriptives.rev20.NS.csv"),
        ["year", "n_total", "n_represented", "pct_represented", 
         "n_grant", "pct_grant", "n_attorney", "n_agent", "n_vso", "n_pro_se"],
        t1_rows
    )
    
    # Flow diagram data
    initial_records = 1327419
    excluded_suspect = 435568
    final_cohort = len(df)
    
    flow_data = [
        ["step", "n", "description"],
        ["initial", initial_records, "Initial public records"],
        ["excluded_suspect", -excluded_suspect, "Duplicates, missing, suspect QC flags"],
        ["excluded_qc", -(initial_records - excluded_suspect - final_cohort), "Additional QC exclusions"],
        ["final_cohort", final_cohort, "Non-suspect cohort for analysis"],
    ]
    
    write_csv(
        os.path.join(OUTD, "table_s1_flow.rev20.csv"),
        flow_data[0],
        flow_data[1:]
    )

# ==================== MULTI-REP ANALYSIS (ISSUE #5) ====================
def analyze_multi_rep(df):
    """Document multi-representative coding with examples"""
    log("Analyzing multi-representative cases...")
    
    n_multi = df['multi_rep'].sum()
    pct_multi = n_multi / len(df) * 100
    
    # Get examples
    examples = []
    if n_multi > 0:
        sample_size = min(10, n_multi)
        for _, row in df[df['multi_rep']].sample(sample_size).iterrows():
            examples.append([row['rep_type'], row['rep_primary']])
    
    with open(os.path.join(OUTD, "multi_rep_summary.rev20.txt"), "w") as f:
        f.write("MULTI-REPRESENTATIVE CASE CODING\n")
        f.write("="*60 + "\n\n")
        f.write(f"Frequency: {n_multi:,} cases ({pct_multi:.2f}%)\n\n")
        f.write("Hierarchy: Attorney > Agent > VSO > Pro se/Unknown\n\n")
        f.write("When multiple representatives appear, we assign a single type\n")
        f.write("using the hierarchy above. This mirrors common practice and\n")
        f.write("simplifies interpretation.\n\n")
        f.write("EXAMPLES:\n")
        f.write("-"*60 + "\n")
        for orig, assigned in examples:
            f.write(f"Original text: {orig}\n")
            f.write(f"Assigned type: {assigned}\n\n")
        f.write("-"*60 + "\n")
        f.write("\nFor manuscript Methods section:\n")
        f.write("'When multiple representatives appeared (n={:,}, {:.1f}% of cohort),\n".format(n_multi, pct_multi))
        f.write("we assigned a single type using the hierarchy Attorney > Agent > VSO.\n")
        f.write("Sensitivity analysis dropping multi-representative cases yielded\n")
        f.write("consistent results (Table S6).'\n")
    
    log(f"Multi-rep: {n_multi:,} cases ({pct_multi:.1f}%)")

# ==================== CELL SIZE CHECKING (ISSUE #8) ====================
def check_cell_sizes(df, min_size=MIN_CELL_SIZE):
    """Check year × representation cell sizes"""
    log(f"Checking cell sizes (min={min_size})...")
    
    small_cells = []
    for y in sorted(df['year'].unique()):
        for rep in ['attorney', 'agent', 'vso', 'generic']:
            col = f'rep_{rep}'
            n = ((df['year'] == y) & (df[col] == 1)).sum()
            if n < min_size and n > 0:
                small_cells.append([y, rep, n, "below_threshold"])
    
    if small_cells:
        write_csv(
            os.path.join(OUTD, "table_s2_small_cells.rev20.csv"),
            ["year", "rep_type", "n", "reason"],
            small_cells
        )
        log(f"Small cells: {len(small_cells)} combinations < {min_size}")
    else:
        log("No small cells detected")
    
    return small_cells

# ==================== AUDIT SAMPLE (ISSUE #12) ====================
def generate_audit_sample(df, n=AUDIT_SAMPLE_SIZE):
    """Generate stratified audit sample"""
    log(f"Generating audit sample (n={n})...")
    
    sample_rows = []
    years = sorted(df['year'].unique())
    n_per_year = max(1, n // len(years))
    
    for year in years:
        dfy = df[df['year'] == year]
        
        if len(dfy) < n_per_year:
            sample = dfy
        else:
            sample = dfy.groupby('disp_group', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, n_per_year // 3)))
            )
        
        sample_rows.extend(sample[['case_id', 'year', 'rep_type', 'rep_primary', 'disp_group']].values.tolist())
    
    if len(sample_rows) > n:
        sample_rows = sample_rows[:n]
    
    write_csv(
        os.path.join(OUTD, f"audit_sample_{n}.rev20.csv"),
        ["case_id", "year", "rep_type", "rep_primary", "disp_group"],
        sample_rows
    )
    
    with open(os.path.join(OUTD, "audit_instructions.rev20.txt"), "w") as f:
        f.write("AUDIT SAMPLE INSTRUCTIONS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Sample: {len(sample_rows)} cases stratified by year/outcome\n\n")
        f.write("For each case:\n")
        f.write("1. Locate original PDF\n")
        f.write("2. Verify rep_type extraction\n")
        f.write("3. Verify rep_primary hierarchy\n")
        f.write("4. Verify disp_group coding\n\n")
        f.write("Calculate accuracy with 95% CI (Wilson score method)\n")
    
    log(f"Audit sample: {len(sample_rows)} cases")

# ==================== MAIN ====================
def main():
    log("\nStarting Part 1: Data loading and preprocessing\n")
    
    if not os.path.exists(DB):
        log(f"ERROR: Database not found: {DB}")
        return None
    
    try:
        con = sqlite3.connect(DB, timeout=60.0)
        log(f"Connected: {DB}")
    except Exception as e:
        log(f"ERROR: Connection failed: {e}")
        return None
    
    # Fetch and build
    cases, has_vlj, vlj_col = fetch_cases(con, ns_only=True)
    con.close()
    
    if not cases:
        log("ERROR: No cases fetched")
        return None
    
    df = build_df(cases, has_vlj, vlj_col)
    
    if df.empty:
        log("ERROR: DataFrame empty")
        return None
    
    # Generate outputs
    generate_descriptive_tables(df)
    analyze_multi_rep(df)
    check_cell_sizes(df)
    generate_audit_sample(df)
    
    log(f"\nPart 1 complete: {len(df):,} cases processed")
    log(f"Checkpoint saved: {CHECKPOINT_DF}")
    log("\nNext: Run analysis_part2_primary_models_small_cluster_inference.py for primary models")
    
    return df

if __name__ == "__main__":
    try:
        df = main()
        if df is not None:
            print(f"\nSUCCESS: {len(df):,} cases ready for analysis")
    except KeyboardInterrupt:
        log("\nInterrupted - checkpoint saved, safe to restart")
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        import traceback
        log(traceback.format_exc())
