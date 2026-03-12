# Emits a complete "Methods Pack" for the BVA study (no DB work).
# Outputs under: {BVA_ROOT}/index/methods_rev17/
import os, sys, json, csv, hashlib, datetime, platform, subprocess, traceback

ROOT = os.environ.get("BVA_ROOT", ".")
IDX  = os.path.join(ROOT, "index")
OUTD = os.path.join(IDX, "methods_rev17")
os.makedirs(OUTD, exist_ok=True)
ENC = "utf-8"

NOW = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---- Helper: hashing & file meta ----
def sha256_of(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def meta_row(path, kind):
    st = os.stat(path)
    return {
        "kind": kind,
        "path": path,
        "bytes": st.st_size,
        "modified": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "sha256": sha256_of(path)
    }

# ---- Codebook (concise but complete for our CSVs) ----
CODEBOOK = {
  "negative_rate_by_decision_year.rev15.*.csv": {
    "description": "Negative-rate rollups by decision year under different sensitivity scenarios.",
    "fields": {
      "decision_year": "YYYY (int).",
      "negative_count": "Count of decisions classified as negative per policy (deny [+ dismiss if *_as_neg]).",
      "total_rows": "Total decisions in the scope (Non-suspects or All rows).",
      "rate_negative": "negative_count / total_rows (float in 0..1)."
    },
    "variants": {
      "NS_D_as_neg": "Non-suspects only; dismissals counted as negative.",
      "NS_D_separate": "Non-suspects only; dismissals reported separately (not in negative_count).",
      "ALL_D_as_neg": "All records (incl. suspects); dismissals counted as negative.",
      "ALL_D_separate": "All records; dismissals separate."
    }
  },
  "rep_presence_by_decision_year.rev15.*.csv": {
    "description": "Representation presence by year.",
    "fields": {
      "decision_year": "YYYY.",
      "any_rep_count": "Decisions with any representative (attorney, agent, VSO, or generic represented).",
      "total_rows": "Total decisions in scope.",
      "rate_any_rep": "any_rep_count / total_rows."
    }
  },
  "dispositions_by_year.rev15.*.csv": {
    "description": "Disposition mix by year.",
    "fields": {
      "decision_year": "YYYY.",
      "deny|dismiss|grant|mixed|remand|vacate|withdraw|unknown": "Label then *_n count column for each disposition.",
      "total_label": "Literal 'total' for readability.",
      "total_n": "Total decisions in scope."
    }
  },
  "rep_type_share_by_year.clean.*.csv": {
    "description": "Share of representation types by year (cleaned pipeline).",
    "fields": {
      "decision_year": "YYYY.",
      "attorney|agent|VSO|unknown|represented_generic": "Counts or shares by type depending on file.",
      "total_rows": "Total decisions considered.",
      "rate_*": "Type share = count / total_rows."
    }
  }
}

# ---- Regex families & rules (documented) ----
REGEX_FAMILIES = {
  "rep_line_detection": {
    "goal": "Find the line stating who represents the Veteran/Appellant.",
    "include_patterns": [
      r"^\\s*Representation[:\\-]",
      r"^\\s*Appellant represented by[:\\-]",
      r"^\\s*Veteran represented by[:\\-]",
      r"\\bThe Veteran is represented by\\b",
      r"\\bAttorney for the Veteran\\b",
      r"\\bAgent for the Veteran\\b",
      r"\\bRepresentative\\b.*\\bfor the Veteran\\b"
    ],
    "exclusions": [
      r"\\bAttorney for the Board\\b",  # Board counsel (not claimant rep)
      r"\\bCounsel for the Board\\b",
      r"\\bBoard Attorney\\b"
    ],
    "notes": "Greedy stoplines: next blank line, section header (ORDER, FINDINGS OF FACT, REASONS AND BASES), or all-caps header."
  },
  "rep_type_classification": {
    "attorney_cues": [
      r"\\battorney\\b", r"\\besq\\.?\\b", r"\\bcounsel\\b", r"\\blaw\\s+office\\b"
    ],
    "agent_cues": [ r"\\bagent\\b", r"\\baccredited agent\\b" ],
    "vso_cues_examples": [
      r"\\bThe American Legion\\b", r"\\bDisabled American Veterans\\b", r"\\bDAV\\b",
      r"\\bVeterans of Foreign Wars\\b", r"\\bVFW\\b", r"\\bAMVETS\\b",
      r"\\bParalyzed Veterans of America\\b", r"\\bPVA\\b",
      r"\\bVietnam Veterans of America\\b", r"\\bVVA\\b",
      r"\\bTexas Veterans Commission\\b", r"\\bFlorida Department of Veterans' Affairs\\b"
    ],
    "generic_presence": [ r"\\brepresented\\b", r"\\brepresentation\\b" ],
    "board_only_exclusions": [ r"\\bAttorney for the Board\\b", r"\\bBoard Attorney\\b" ],
    "decision_logic": "If excluded by board_only_exclusions -> board_only. Else if any vso_cues -> VSO. Else if any attorney_cues -> attorney. Else if any agent_cues -> agent. Else if generic_presence -> represented_generic. Else -> unknown."
  },
  "disposition_mapping": {
    "source": "qc.primary_disposition per case derived from ORDER/CONCLUSION sections.",
    "labels": ["deny","dismiss","grant","mixed","remand","vacate","withdraw","unknown"],
    "deny_equivalent_note": "Sensitivity: optionally treat 'dismiss' as negative."
  },
  "structural_qc": {
    "flags": [
      "very_short_text", "header_only", "missing_rep_section",
      "missing_decision_date", "missing_docket", "parse_error"
    ],
    "purpose": "Exclude structurally suspect decisions from the primary (NS) series."
  }
}

# ---- Reproducibility narrative ----
REPRO = f"""# Reproducibility (rev17)

**Study window:** 2010-2025.  
**DB:** {BVA_ROOT}/parsed\\parsed.full.rev01.sqlite (tables: `parsed`, `qc`).

## 1) Inventory & QC
- Build manifest and parsed index: `bva_indexer.rev02.pyw` (text scan) → `manifest.rev01.*.csv`, `manifest.rev01.sqlite`.
- Structural QC: `bva_disposition_and_qc.rev01.pyw` + `bva_qc_recheck_txt.rev07.pyw`  
  → `qc_structural_flags.rev0*.csv`, full `qc` table with `is_struct_suspect`.

## 2) Representation extraction & cleaning
- Initial parse: `bva_enrich_rep_and_outcome.rev02.pyw` (rep_line, rep_type, outcome lines).
- Backfill/boost (2019-2025): `bva_backfill_rep.rev0*.pyw`, `bva_rep_reclass_2019_2025.rev09.pyw`, `bva_rep_presence_boost_2019_2025.rev10.pyw`.
- Year rollups and fixes: `bva_fix_decision_year_and_rollups.rev01.pyw`, `bva_rep_year_merge.rev01.pyw`, `bva_study_rollups.clean.rev01.pyw`.

## 3) Core study outputs
- Dispositions & negative rate (NS vs ALL, dismiss policy toggles): `bva_sensitivity_cuts.rev15*.pyw`  
  → `negative_rate_by_decision_year.rev15.<TAG>.csv`, `dispositions_by_year.rev15.<TAG>.csv`, `rep_presence_by_decision_year.rev15.<TAG>.csv`
  with TAG in {{NS_D_as_neg, NS_D_separate, ALL_D_as_neg, ALL_D_separate}}.

## 4) Figures & effect sizes
- Figures: `bva_figures_from_sensitivity.rev16*.pyw` → `index\\figs_2010_2025\\rev16\\*.svg/.png`, `figure_sheet.rev16.pdf`.
- Effect sizes: `bva_effect_sizes.rev16d.pyw` → `effect_sizes.rev16d.csv/.txt`.

## 5) Logging & snapshots
- Each step writes a dated log under `{BVA_ROOT}/index\\` (e.g., `sensitivity.rev15f.log.txt`).
- Outputs are immutable by revision suffix; superseded files moved under `_archive` by housekeeping.

**Sensitivity definitions:**
- *Policy:* `as_neg` counts dismissals in negative_count; `separate` does not.
- *QC scope:* `NS` = `q.is_struct_suspect = 0`; `ALL` includes suspects.

**Recommended reporting:**
- Primary: Non-suspects, deny **+ dismiss** (veteran-centric negative rate), with deny-only as a sensitivity overlay.
- Always provide the “ALL rows” series as a secondary sensitivity band.

Generated on: {NOW}.
"""

# ---- README overview ----
README = f"""# Methods Pack (rev17)

This folder contains the documentation and provenance needed to reproduce the BVA 2010-2025 study.

**Files**
- `codebook.rev17.json`: field definitions for the study CSVs.
- `regex_families.rev17.json`: regex cues and classification rules used for representation and QC.
- `lineage.rev17.csv`: file provenance: path, size, mtime, SHA256.
- `environment.rev17.txt`: Python/OS and (when available) key package versions.
- `reproducibility.rev17.md`: end-to-end steps and sensitivity definitions.

Generated on: {NOW}.
"""

def write_text(path, text):
    with open(path, "w", encoding=ENC) as f:
        f.write(text)

def write_json(path, obj):
    with open(path, "w", encoding=ENC) as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def main():
    try:
        # 1) Write codebook & regex families & reproducibility & README
        write_json(os.path.join(OUTD, "codebook.rev17.json"), CODEBOOK)
        write_json(os.path.join(OUTD, "regex_families.rev17.json"), REGEX_FAMILIES)
        write_text(os.path.join(OUTD, "reproducibility.rev17.md"), REPRO)
        write_text(os.path.join(OUTD, "README.rev17.md"), README)

        # 2) Environment capture
        env_lines = []
        env_lines.append(f"Generated: {NOW}")
        env_lines.append(f"Python: {sys.version}")
        env_lines.append(f"Executable: {sys.executable}")
        env_lines.append(f"Platform: {platform.platform()}")
        # Try a minimal pip freeze (best-effort)
        try:
            out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], timeout=30)
            env_lines.append("\n[pip freeze]")
            env_lines.append(out.decode(ENC, errors="ignore"))
        except Exception as e:
            env_lines.append(f"\n[pip freeze] unavailable: {e}")
        write_text(os.path.join(OUTD, "environment.rev17.txt"), "\n".join(env_lines))

        # 3) Lineage: hash important inputs/outputs
        candidates = []
        # Inputs (DB + key intermediates if present)
        for p in [
            os.path.join(ROOT, "parsed", "parsed.full.rev01.sqlite"),
            os.path.join(IDX, "manifest.rev01.part01.csv"),
            os.path.join(IDX, "manifest.rev01.part02.csv"),
            os.path.join(IDX, "manifest.rev01.sqlite"),
        ]:
            if os.path.exists(p): candidates.append(("input", p))

        # Core study CSVs (rev15 sensitivity outputs)
        tags = ["NS_D_as_neg","NS_D_separate","ALL_D_as_neg","ALL_D_separate"]
        for base in ["negative_rate_by_decision_year.rev15",
                     "dispositions_by_year.rev15",
                     "rep_presence_by_decision_year.rev15"]:
            for t in tags:
                p = os.path.join(IDX, f"{base}.{t}.csv")
                if os.path.exists(p): candidates.append(("output", p))

        # Clean rollups & figures
        for p in [
            os.path.join(IDX, "rep_type_share_by_year.clean.2010_2025.rev10.csv"),
            os.path.join(IDX, "rep_type_share_by_year.clean.2010_2025.rev14.csv"),
            os.path.join(IDX, "rep_presence_by_decision_year.clean.2010_2025.rev10.csv"),
            os.path.join(IDX, "rep_presence_by_decision_year.clean.2010_2025.rev14.csv"),
            os.path.join(IDX, "figs_2010_2025", "rev16", "figure_sheet.rev16.pdf"),
        ]:
            if os.path.exists(p): candidates.append(("artifact", p))

        # Logs that anchor runs
        for name in ["sensitivity.rev15f.log.txt", "figs_build.rev16c.log.txt"]:
            p = os.path.join(IDX, name)
            if os.path.exists(p): candidates.append(("log", p))

        # Compute lineage
        rows = []
        for kind, path in candidates:
            try:
                rows.append(meta_row(path, kind))
            except Exception:
                # skip unreadable files
                pass

        # Write lineage CSV
        lin_csv = os.path.join(OUTD, "lineage.rev17.csv")
        with open(lin_csv, "w", encoding=ENC, newline="") as f:
            w = csv.DictWriter(f, fieldnames=["kind","path","bytes","modified","sha256"])
            w.writeheader()
            for r in rows: w.writerow(r)

        # Done
        write_text(os.path.join(OUTD, "done.rev17.txt"), f"Methods pack generated at {NOW} with {len(rows)} lineage entries.")
    except Exception:
        err = traceback.format_exc()
        write_text(os.path.join(OUTD, "ERROR.rev17.txt"), err)

if __name__ == "__main__":
    main()
