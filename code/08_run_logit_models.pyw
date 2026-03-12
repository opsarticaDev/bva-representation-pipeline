# Logistic regression models for the BVA study (2010-2025).
# - Builds NS (primary) and optional ALL cohorts from SQLite
# - Creates lightweight features (rep dummies, hearing & AMA cues, decision length)
# - Fits logit models with year fixed effects & clustered SEs by year
# - Writes Table 1 (descriptives) and Table 2 (logit) CSVs
# Safe for large data; no DB writes.

import os, sys, csv, sqlite3, datetime, traceback, subprocess, math, re

# ---------------- PATHS ----------------
ROOT = os.environ.get("BVA_ROOT", ".")
DB   = os.path.join(ROOT, "parsed", "parsed.full.rev01.sqlite")
OUTD = os.path.join(ROOT, "index", "tables_rev18")
LIBS_MAIN = os.path.join(ROOT, "_libs2")  # prefer the clean portable stack
LIBS_FALL = os.path.join(ROOT, "_libs")   # optional fallback if present
os.makedirs(OUTD, exist_ok=True)

# Prefer portable libs if available
if os.path.isdir(LIBS_MAIN):
    sys.path.insert(0, LIBS_MAIN)
if os.path.isdir(LIBS_FALL):
    sys.path.insert(0, LIBS_FALL)

# ---------------- CONFIG ----------------
YEARS = list(range(2010, 2026))
RUN_ALL_COHORT   = False    # True to also run ALL (includes structural suspects)
SAMPLE_FRACTION  = 1.0      # e.g., 0.10 for a 10% dry run
READ_TEXT_FOR_CUES = True   # parses .txt to build hearing/AMA flags & length
MAX_PER_YEAR     = None     # e.g., 25000 to cap per-year reads for testing
ENC = "utf-8"
# ----------------------------------------

# ---- Ensure deps (portable install only if truly missing) ----
def ensure_lib(mod_name, pip_name=None, target=LIBS_MAIN):
    try:
        __import__(mod_name); return
    except ImportError:
        pass
    if not os.path.exists(target):
        os.makedirs(target, exist_ok=True)
    exe = sys.executable
    pkg = pip_name or mod_name
    cmd = [exe, "-m", "pip", "install", "--no-cache-dir", "--target", target, pkg]
    try:
        subprocess.check_call(cmd)
        sys.path.insert(0, target)
        __import__(mod_name)
    except Exception as e:
        # final try: fallback target
        if target != LIBS_FALL:
            ensure_lib(mod_name, pip_name, target=LIBS_FALL)
        else:
            raise RuntimeError(f"Failed to import/install {mod_name}: {e}")

# Only try to install if the import fails
try:
    import pandas as pd
except Exception:
    ensure_lib("pandas", "pandas==2.2.2")
    import pandas as pd

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except Exception:
    ensure_lib("statsmodels", "statsmodels==0.14.1")
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

# ---- Logging ----
LOGP = os.path.join(OUTD, "rev18_build.log.txt")
def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOGP, "a", encoding=ENC) as f:
        f.write(f"{ts} | {msg}\n")

# ---- Text cues ----
RX_HEARING = re.compile(r"\bhearing\b|virtual hearing|video hearing|tele[- ]?hearing|teleconference|Microsoft\s+Teams|Zoom|Webex", re.IGNORECASE)
RX_AMA     = re.compile(r"\bAppeals Modernization Act\b|\bAMA\b", re.IGNORECASE)

def features_from_text(path):
    """Return (length_chars, hearing_flag, ama_flag)."""
    if not READ_TEXT_FOR_CUES or not path or not os.path.exists(path):
        return 0, 0, 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            t = f.read()
    except Exception:
        return 0, 0, 0
    return len(t), int(bool(RX_HEARING.search(t))), int(bool(RX_AMA.search(t)))

def fetch_cases(con, ns_only=True):
    """Yield tuples: (case_id, decision_year, rep_type, primary_disposition, source_path)."""
    cur = con.cursor()
    total = 0
    for y in YEARS:
        if ns_only:
            cur.execute("""
               SELECT p.case_id, TRIM(p.decision_year), p.rep_type, q.primary_disposition, p.source_path
                 FROM parsed p
                 JOIN qc q ON q.case_id = p.case_id
                WHERE TRIM(p.decision_year) = ?
                  AND LOWER(p.file_ext) = '.txt'
                  AND (q.is_struct_suspect = 0 OR q.is_struct_suspect = '0')
            """, (str(y),))
        else:
            cur.execute("""
               SELECT p.case_id, TRIM(p.decision_year), p.rep_type, q.primary_disposition, p.source_path
                 FROM parsed p
                 JOIN qc q ON q.case_id = p.case_id
                WHERE TRIM(p.decision_year) = ?
                  AND LOWER(p.file_ext) = '.txt'
            """, (str(y),))
        rows = cur.fetchall()
        total += len(rows)
        log(f"Fetched {len(rows):,} rows for {y} (ns_only={ns_only})")
        for r in rows:
            yield r
    log(f"Total fetched (ns_only={ns_only}): {total:,}")


def build_dataframe(row_iter):
    """Builds DataFrame with features & outcomes."""
    recs = []
    for (cid, year, rep, disp, path) in row_iter:
        rep = (rep or "").strip().lower()
        rep_att = 1 if rep == "attorney" else 0
        rep_ag  = 1 if rep == "agent" else 0
        rep_vso = 1 if rep == "vso" else 0
        rep_gen = 1 if rep == "represented_generic" else 0

        length, hearing, ama = features_from_text(path)

        d = (disp or "unknown").strip().lower()
        neg_as  = 1 if d in ("deny", "dismiss") else 0
        neg_sep = 1 if d in ("deny",) else 0
        grant   = 1 if d == "grant" else 0

        # For logit, outcome is grant vs. non-grant (two policy views use same y)
        recs.append({
            "case_id": cid,
            "year": int(year) if str(year).isdigit() else year,
            "rep_type": rep or "unknown",
            "rep_attorney": rep_att, "rep_agent": rep_ag, "rep_vso": rep_vso, "rep_generic": rep_gen,
            "disp": d,
            "grant": grant,
            "neg_as": neg_as, "neg_sep": neg_sep,
            "grant_bin_asneg": 1 if grant == 1 else 0,
            "grant_bin_separate": 1 if grant == 1 else 0,
            "len_chars": length,
            "hearing": hearing,
            "ama_cue": ama
        })
    df = pd.DataFrame.from_records(recs)
    if (SAMPLE_FRACTION is not None) and SAMPLE_FRACTION < 1.0 and len(df) > 0:
        df = df.sample(frac=SAMPLE_FRACTION, random_state=42).reset_index(drop=True)
    return df

def table1_descriptives(df, tag):
    """Writes Table 1 by year; safe if df is empty."""
    outp = os.path.join(OUTD, f"table1_descriptives.rev18.{tag}.csv")
    header = ["year","n","grant_rate","neg_rate_asneg","neg_rate_separate",
              "rep_presence","share_attorney","share_agent","share_vso","share_generic"]
    if df.empty:
        with open(outp, "w", encoding=ENC, newline="") as f:
            w = csv.writer(f); w.writerow(header)
        log(f"Skipped Table 1 ({tag}): no rows.")
        return
    rows = []
    for y, g in df.groupby("year"):
        n = len(g)
        if n == 0:
            continue
        grant_rate   = g["grant"].mean()
        neg_rate_as  = g["neg_as"].mean()
        neg_rate_sep = g["neg_sep"].mean()
        any_rep      = ((g["rep_attorney"]|g["rep_agent"]|g["rep_vso"]|g["rep_generic"])>0).mean()
        mix_att = g["rep_attorney"].mean()
        mix_ag  = g["rep_agent"].mean()
        mix_vso = g["rep_vso"].mean()
        mix_gen = g["rep_generic"].mean()
        rows.append([y, n, grant_rate, neg_rate_as, neg_rate_sep, any_rep, mix_att, mix_ag, mix_vso, mix_gen])
    rows = sorted(rows, key=lambda r: r[0])
    with open(outp, "w", encoding=ENC, newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    log(f"Wrote {outp}")

def fit_logit(df, y_col, tag):
    """Fits logit with year FE and clustered SEs by year; robust to singularities."""
    outp = os.path.join(OUTD, f"table2_logit_{y_col}.rev18.{tag}.csv")
    header = ["term","coef","se","z","pvalue","odds_ratio","note"]

    if df.empty:
        log(f"Skipped logit {y_col} ({tag}): no rows.")
        with open(outp, "w", encoding=ENC, newline="") as f:
            csv.writer(f).writerow(header)
        return

    Xcols = ["rep_attorney","rep_agent","rep_vso","rep_generic","hearing","ama_cue"]
    df0 = df.dropna(subset=[y_col]).copy()

    # Clip extreme length and keep as candidate feature
    q99 = df0["len_chars"].quantile(0.99) if len(df0) > 100 else df0["len_chars"].max()
    df0["len_chars_clip"] = df0["len_chars"].clip(upper=q99)
    Xcols_all = Xcols + ["len_chars_clip"]

    # 1) Drop problematic years: too small or zero variance in outcome
    bad_years = []
    for y, g in df0.groupby("year"):
        if len(g) < 100:
            bad_years.append(y)
            continue
        vy = g[y_col].nunique()
        if vy < 2:  # all 0 or all 1
            bad_years.append(y)
    if bad_years:
        log(f"Dropping years for {y_col} due to size/variance: {sorted(bad_years)}")
        df0 = df0[~df0["year"].isin(bad_years)].copy()

    if df0.empty:
        log(f"After filters, no rows for {y_col} ({tag}): writing empty table.")
        with open(outp, "w", encoding=ENC, newline="") as f:
            csv.writer(f).writerow(header)
        return

    # 2) Drop constant predictors
    keep = []
    for c in Xcols_all:
        if df0[c].nunique() > 1:
            keep.append(c)
        else:
            log(f"Dropping constant predictor: {c}")
    Xcols_eff = keep

    # If all rep indicators dropped, we still fit with remaining controls + year FE
    # Build formula dynamically
    rhs = []
    for c in Xcols_eff:
        rhs.append(c)
    rhs.append("C(year)")
    formula = f"{y_col} ~ " + " + ".join(rhs)

    rows = []

    # 3) Try Logit with clustered SEs
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    model = smf.logit(formula=formula, data=df0)
    try:
        res = model.fit(disp=False, maxiter=200, cov_type="cluster", cov_kwds={"groups": df0["year"]})
        used = "logit_cluster"
    except Exception as e1:
        log(f"Logit failed ({y_col}/{tag}): {e1} - trying GLM Binomial clustered")
        # 4) GLM Binomial with clustered SEs
        try:
            glm = smf.glm(formula=formula, data=df0, family=sm.families.Binomial())
            res = glm.fit(cov_type="cluster", cov_kwds={"groups": df0["year"]})
            used = "glm_binom_cluster"
        except Exception as e2:
            log(f"GLM Binomial clustered failed: {e2} - trying GLM Binomial HC1")
            # 5) Last resort: GLM HC1 (non-clustered robust)
            glm = smf.glm(formula=formula, data=df0, family=sm.families.Binomial())
            res = glm.fit(cov_type="HC1")
            used = "glm_binom_HC1"

    # Build table (omit C(year) rows)
    import math as _math
    for name, coef, se, p in zip(res.params.index, res.params.values, res.bse, res.pvalues):
        if name.startswith("C(year)"):
            continue
        z = (coef / se) if se > 0 else float("nan")
        orr = _math.exp(coef) if _math.isfinite(coef) else float("nan")
        rows.append([name, coef, se, z, p, orr, used])

    with open(outp, "w", encoding=ENC, newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    log(f"Wrote {outp} (fit={used}; n={len(df0):,}; years={df0['year'].nunique()})")

def main():
    # Record run config
    with open(os.path.join(OUTD, "model_notes.rev18.txt"), "w", encoding=ENC) as f:
        f.write(f"rev18 run at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"DB: {DB}\n")
        f.write(f"YEARS: {YEARS}\n")
        f.write(f"RUN_ALL_COHORT: {RUN_ALL_COHORT}\n")
        f.write(f"SAMPLE_FRACTION: {SAMPLE_FRACTION}\n")
        f.write(f"READ_TEXT_FOR_CUES: {READ_TEXT_FOR_CUES}\n")
        f.write(f"MAX_PER_YEAR: {MAX_PER_YEAR}\n")

    log("Starting rev18...")
    if not os.path.exists(DB):
        log(f"ERROR: DB missing: {DB}")
        return

    con = sqlite3.connect(DB, timeout=60.0)
    cur = con.cursor()
    cur.execute("PRAGMA busy_timeout=60000;")
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    # ---- NS cohort (primary) ----
    log("Fetching NS cohort...")
    ns_iter = fetch_cases(con, ns_only=True)
    df_ns = build_dataframe(ns_iter)
    log(f"NS DataFrame: {len(df_ns):,} rows")

    table1_descriptives(df_ns, "NS")
    fit_logit(df_ns, "grant_bin_asneg", "NS")
    fit_logit(df_ns, "grant_bin_separate", "NS")

    # ---- ALL cohort (optional) ----
    if RUN_ALL_COHORT:
        log("Fetching ALL cohort...")
        all_iter = fetch_cases(con, ns_only=False)
        df_all = build_dataframe(all_iter)
        log(f"ALL DataFrame: {len(df_all):,} rows")
        table1_descriptives(df_all, "ALL")
        fit_logit(df_all, "grant_bin_asneg", "ALL")
        fit_logit(df_all, "grant_bin_separate", "ALL")

    con.close()
    log("Done rev18.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(LOGP, "a", encoding=ENC) as f:
            f.write("FATAL:\n" + traceback.format_exc() + "\n")
