#!/usr/bin/env python3
"""
Quality review and cleaning for extracted secondary grants.
Generates reports and creates cleaned dataset.
"""
import os
import re
import pandas as pd
from collections import Counter
from pathlib import Path

ROOT = os.environ.get("BVA_ROOT", ".")
INPUT_CSV = os.path.join(ROOT, 'secondary_grants_extraction.csv')
OUTPUT_DIR = os.path.join(ROOT, 'analysis')
CLEANED_CSV = os.path.join(OUTPUT_DIR, 'secondary_grants_cleaned.csv')
REPORT_FILE = os.path.join(OUTPUT_DIR, 'quality_report.txt')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Additional filters for edge cases
ADDITIONAL_BLOCKLIST = re.compile(
    r'\b(claimant|veteran|appellant|service member|board|va|'
    r'evidence|claim|appeal|decision|rating|exam|examination|'
    r'statement|opinion|report|record|file|documents?|'
    r'percentage|percent|effective date|award|'
    r'reasons stated|herein|above|below|foregoing)\b',
    re.I
)

def normalize_condition(s):
    """Normalize condition text for grouping."""
    if pd.isna(s):
        return ''
    s = str(s).lower().strip()
    # Remove parentheticals
    s = re.sub(r'\([^)]*\)', '', s)
    # Normalize spacing
    s = re.sub(r'\s+', ' ', s)
    # Remove trailing punctuation
    s = re.sub(r'[.,;:]+$', '', s)
    return s.strip()

def is_likely_valid(row):
    """Additional validation for edge cases."""
    sec = str(row['secondary_condition']).strip()
    pri = str(row['primary_condition']).strip()
    
    # Length checks
    if len(sec) < 3 or len(pri) < 3:
        return False, 'too_short'
    if len(sec) > 150 or len(pri) > 150:
        return False, 'too_long'
    
    # Check for procedural language
    if ADDITIONAL_BLOCKLIST.search(sec) or ADDITIONAL_BLOCKLIST.search(pri):
        return False, 'procedural_language'
    
    # Check for identical or near-identical primary/secondary
    if normalize_condition(sec) == normalize_condition(pri):
        return False, 'identical_conditions'
    
    # Require some letters
    if sum(c.isalpha() for c in sec) < 3 or sum(c.isalpha() for c in pri) < 3:
        return False, 'insufficient_alpha'
    
    # Check for incomplete extraction (often ends with "is" or "are")
    if re.search(r'\b(is|are|was|were|of|the|and)$', sec, re.I):
        return False, 'incomplete_extraction_sec'
    if re.search(r'\b(is|are|was|were|of|the|and)$', pri, re.I):
        return False, 'incomplete_extraction_pri'
    
    return True, 'valid'

def generate_report(df, df_cleaned, rejection_counts):
    """Generate detailed quality report."""
    report = []
    report.append("=" * 80)
    report.append("SECONDARY GRANTS EXTRACTION - QUALITY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall stats
    report.append(f"Total extracted grants: {len(df):,}")
    report.append(f"After cleaning: {len(df_cleaned):,}")
    report.append(f"Rejected: {len(df) - len(df_cleaned):,} ({100*(len(df)-len(df_cleaned))/len(df):.1f}%)")
    report.append("")
    
    # Rejection reasons
    report.append("REJECTION REASONS:")
    report.append("-" * 40)
    for reason, count in rejection_counts.most_common(20):
        pct = 100 * count / len(df)
        report.append(f"  {reason:30s}: {count:6,} ({pct:5.2f}%)")
    report.append("")
    
    # Relationship type distribution
    report.append("RELATIONSHIP TYPES (cleaned data):")
    report.append("-" * 40)
    rel_counts = df_cleaned['relationship_type'].value_counts()
    for rel, count in rel_counts.items():
        pct = 100 * count / len(df_cleaned)
        report.append(f"  {rel:30s}: {count:6,} ({pct:5.1f}%)")
    report.append("")
    
    # Top secondary conditions
    report.append("TOP 25 SECONDARY CONDITIONS:")
    report.append("-" * 40)
    sec_norm = df_cleaned['secondary_condition'].apply(normalize_condition)
    for cond, count in sec_norm.value_counts().head(25).items():
        report.append(f"  {count:5,} | {cond}")
    report.append("")
    
    # Top primary conditions
    report.append("TOP 25 PRIMARY CONDITIONS:")
    report.append("-" * 40)
    pri_norm = df_cleaned['primary_condition'].apply(normalize_condition)
    for cond, count in pri_norm.value_counts().head(25).items():
        report.append(f"  {count:5,} | {cond}")
    report.append("")
    
    # Top condition pairs
    report.append("TOP 25 CONDITION PAIRS (Secondary -> Primary):")
    report.append("-" * 40)
    df_cleaned['pair_normalized'] = (
        df_cleaned['secondary_condition'].apply(normalize_condition) + 
        ' -> ' + 
        df_cleaned['primary_condition'].apply(normalize_condition)
    )
    for pair, count in df_cleaned['pair_normalized'].value_counts().head(25).items():
        report.append(f"  {count:5,} | {pair}")
    report.append("")
    
    # Unique counts
    report.append("UNIQUE ENTITY COUNTS (cleaned data):")
    report.append("-" * 40)
    sec_unique = sec_norm.nunique()
    pri_unique = pri_norm.nunique()
    pair_unique = df_cleaned['pair_normalized'].nunique()
    report.append(f"  Unique secondary conditions: {sec_unique:,}")
    report.append(f"  Unique primary conditions: {pri_unique:,}")
    report.append(f"  Unique condition pairs: {pair_unique:,}")
    report.append("")
    
    # Temporal analysis (if docket numbers are valid)
    docket_years = df_cleaned['docket_number'].str.extract(r'^(\d{2})', expand=False)
    if docket_years.notna().sum() > 0:
        report.append("GRANTS BY YEAR (from docket number):")
        report.append("-" * 40)
        year_counts = docket_years.value_counts().sort_index()
        for year, count in year_counts.items():
            full_year = f"20{year}" if int(year) <= 25 else f"19{year}"
            report.append(f"  {full_year}: {count:6,}")
        report.append("")
    
    # Source file stats
    report.append("SOURCE FILE COVERAGE:")
    report.append("-" * 40)
    files_with_grants = df_cleaned['source_path'].nunique()
    report.append(f"  Files containing grants: {files_with_grants:,}")
    avg_per_file = len(df_cleaned) / files_with_grants if files_with_grants > 0 else 0
    report.append(f"  Average grants per file: {avg_per_file:.2f}")
    report.append("")
    
    report.append("=" * 80)
    
    return '\n'.join(report)

def main():
    print("Loading extracted grants...")
    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
    
    print(f"Loaded {len(df):,} grants")
    print("Applying quality filters...")
    
    # Apply validation
    results = df.apply(is_likely_valid, axis=1)
    df['is_valid'] = results.apply(lambda x: x[0])
    df['reject_reason'] = results.apply(lambda x: x[1])
    
    # Split into clean and rejected
    df_cleaned = df[df['is_valid']].copy()
    df_rejected = df[~df['is_valid']].copy()
    
    # Drop validation columns from cleaned data
    df_cleaned = df_cleaned.drop(['is_valid', 'reject_reason'], axis=1)
    
    # Count rejections
    rejection_counts = Counter(df_rejected['reject_reason'])
    
    # Generate report
    print("Generating quality report...")
    report = generate_report(df, df_cleaned, rejection_counts)
    
    # Save outputs
    df_cleaned.to_csv(CLEANED_CSV, index=False, encoding='utf-8')
    print(f"[OK] Saved cleaned data: {CLEANED_CSV}")
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[OK] Saved quality report: {REPORT_FILE}")
    
    # Also save rejected for review
    rejected_sample = os.path.join(OUTPUT_DIR, 'rejected_samples.csv')
    df_rejected.head(1000).to_csv(rejected_sample, index=False, encoding='utf-8')
    print(f"[OK] Saved rejected samples: {rejected_sample}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(report)

if __name__ == '__main__':
    main()