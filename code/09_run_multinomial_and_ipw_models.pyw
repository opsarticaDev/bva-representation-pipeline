# Full drop-in replacement for bva_models_multinomial_ipw.rev19.pyw
# Complete: All functions defined, safe_exp for CIs, resume from checkpoint, all models run

import os
import sys
import csv
import datetime
import traceback
import subprocess
import math
import re
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import MNLogit
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
from scipy.stats import norm
from tqdm import tqdm
import sqlite3

ROOT = os.environ.get("BVA_ROOT", ".")
OUTD = os.path.join(ROOT, "index", "Grok")
LIBS_MAIN = os.path.join(ROOT, "_libs2")
os.makedirs(OUTD, exist_ok=True)

ENC = "utf-8"

LOGP = os.path.join(OUTD, "grok_build.log.txt")
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOGP, "a", encoding=ENC) as f:
            f.write(f"{ts} | {msg}\n")
        print(f"{ts} | {msg}")
    except Exception as e:
        print(f"Log error: {e}")

log("Script started - starting analysis")

# ------------- CONFIG -------------
YEARS = list(range(2010, 2026))
READ_TEXT_FOR_CUES = True
SAMPLE_FRACTION = 1.0
MAX_PER_YEAR = None
IPW_TRIM_LO, IPW_TRIM_HI = 0.01, 0.99
BOOTSTRAP_B = 999  # Set 99 for test
DB = os.path.join(ROOT, "parsed", "parsed.full.rev01.sqlite")
CHECKPOINT_DF = os.path.join(OUTD, "checkpoint_df.grok.parquet")
BATCH_SIZE = 100
TEXT_TIMEOUT = 5
# ----------------------------------

RX_HEARING = re.compile(r"\bhearing\b|virtual hearing|video hearing|tele[- ]?hearing|teleconference|Microsoft\s+Teams|Zoom|Webex", re.IGNORECASE)
RX_AMA = re.compile(r"\bAppeals Modernization Act\b|\bAMA\b", re.IGNORECASE)
RX_MULTI_REP = re.compile(r'and|,', re.IGNORECASE)

def features_from_text_batch(paths):
    results = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Reading texts", leave=False):
        batch = paths[i:i+BATCH_SIZE]
        batch_results = []
        for path in batch:
            if not path or not os.path.exists(path):
                batch_results.append((0, 0, 0))
                continue
            if not READ_TEXT_FOR_CUES:
                batch_results.append((0, 0, 0))
                continue
            try:
                start = time.time()
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    t = f.read()
                elapsed = time.time() - start
                if elapsed > TEXT_TIMEOUT:
                    log(f"Timeout on {os.path.basename(path)} ({elapsed:.1f}s) - skipping")
                    t = ""
                len_chars = len(t)
                hearing = int(bool(RX_HEARING.search(t)))
                ama_cue = int(bool(RX_AMA.search(t)))
                batch_results.append((len_chars, hearing, ama_cue))
            except Exception as e:
                log(f"Error reading {os.path.basename(path)}: {e}")
                batch_results.append((0, 0, 0))
        results.extend(batch_results)
    return results

def get_qc_columns(con):
    cur = con.cursor()
    cur.execute("PRAGMA table_info(qc)")
    columns = [row[1] for row in cur.fetchall()]
    log(f"'qc' table columns: {columns}")
    return columns

def fetch_cases(con, ns_only=True):
    cur = con.cursor()
    total = 0
    cases = []
    qc_columns = get_qc_columns(con)
    suspect_col = 'suspect' if 'suspect' in qc_columns else None
    log(f"Suspect column: {suspect_col} (using {'non-suspect' if suspect_col else 'full'} cohort)")
    try:
        for y in tqdm(YEARS, desc="Fetching years"):
            query = """
                SELECT p.case_id, TRIM(p.decision_year), p.rep_type, q.primary_disposition, p.source_path
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
            total += len(fetched)
            log(f"Fetched {len(fetched):,} rows for {y}")
        log(f"Total fetched: {total:,}")
    except Exception as e:
        log(f"DB query failed: {e}")
    return cases

def build_df(cases):
    log("Building DataFrame...")
    paths = [case[4] for case in cases]
    cues = features_from_text_batch(paths)
    data = []
    for i, (cid, year, rep, disp, _) in enumerate(tqdm(cases, desc="Building rows")):
        if i % 1000 == 0:
            log(f"Processed {i:,} rows")
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
        })
    df = pd.DataFrame(data)
    if df.empty:
        log("Build_df produced empty DataFrame")
        return df
    log(f"Built DataFrame: {len(df):,} rows")
    df['rep_attorney'] = (df['rep_type'].str.contains('attorney', case=False, na=False)).astype(int)
    df['rep_agent'] = (df['rep_type'].str.contains('agent', case=False, na=False)).astype(int)
    df['rep_vso'] = (df['rep_type'].str.contains('vso', case=False, na=False)).astype(int)
    df['rep_generic'] = ((df['rep_attorney'] + df['rep_agent'] + df['rep_vso']) == 0).astype(int)
    df['rep_any'] = 1 - df['rep_generic']
    df['multi_rep'] = df['rep_type'].apply(lambda x: bool(RX_MULTI_REP.search(str(x))))
    df['grant_bin'] = df['disp_group'].isin(['grant', 'mixed']).astype(int)
    df['grant_only'] = (df['disp_group'] == 'grant').astype(int)
    df['grant_remand'] = df['disp_group'].isin(['grant', 'remand', 'mixed']).astype(int)
    df['outcome_cat'] = df['disp_group'].map({'grant': 0, 'mixed': 0, 'deny': 1, 'remand': 2, 'dismiss': 1}).fillna(3)
    df['len_chars_z'] = (df['len_chars'] - df['len_chars'].mean()) / df['len_chars'].std()
    df['len_chars_clip'] = df['len_chars'].clip(lower=df['len_chars'].quantile(0.01), upper=df['len_chars'].quantile(0.99))
    try:
        df.to_parquet(CHECKPOINT_DF)
        log(f"Saved DataFrame checkpoint: {CHECKPOINT_DF}")
    except Exception as e:
        log(f"Failed to save checkpoint: {e}")
    return df

def load_checkpoint_if_exists():
    if os.path.exists(CHECKPOINT_DF):
        log("Loading DataFrame checkpoint to resume...")
        try:
            df = pd.read_parquet(CHECKPOINT_DF)
            log(f"Loaded checkpoint: {len(df):,} rows")
            return df
        except Exception as e:
            log(f"Checkpoint load failed: {e} - re-building from scratch")
    return None

def write_csv(path, header, rows):
    try:
        with open(path, "w", newline="", encoding=ENC) as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        log(f"Wrote CSV: {path}")
    except Exception as e:
        log(f"Failed to write CSV {path}: {e}")

def safe_exp(x):
    if np.isnan(x):
        return np.nan
    if x > 700:
        return np.inf
    if x < -700:
        return 0
    return math.exp(x)

def wild_bootstrap(model, df, cluster_col, B=BOOTSTRAP_B):
    coefs = []
    clusters = df[cluster_col].unique()
    orig_params = model.params
    resid = model.resid_response
    fitted = model.fittedvalues
    for b in tqdm(range(B), desc="Bootstrap", leave=False):
        signs = np.random.choice([-1, 1], size=len(clusters), replace=True)
        df_b = df.copy()
        cluster_map = dict(zip(clusters, signs))
        df_b['residual_b'] = resid * df_b[cluster_col].map(cluster_map)
        y_b = fitted + df_b['residual_b']
        model_b = model.model.__class__(y_b, model.model.exog)
        try:
            r_b = model_b.fit(disp=False, start_params=orig_params)
            coefs.append(r_b.params)
        except Exception as e:
            log(f"Bootstrap iter {b} failed: {e}")
    if not coefs:
        log("All bootstrap iterations failed, using original SEs/p-values")
        return model.bse, model.pvalues
    coefs = np.array(coefs)
    se_boot = np.std(coefs, axis=0)
    p_boot = 2 * (1 - norm.cdf(np.abs(orig_params / se_boot)))
    return se_boot, p_boot

def fit_multinomial(df, tag):
    outp = os.path.join(OUTD, f"table3_multinomial.grok.{tag}.csv")
    if os.path.exists(outp):
        log(f"Skipping {os.path.basename(outp)} - already exists")
        return
    header = ["outcome_vs_deny","term","coef","se","lower_ci","upper_ci","z","pvalue","RRR","note"]
    if df.empty:
        write_csv(outp, header, [])
        return
    dfm = df[df['outcome_cat'].isin([0,1,2])].copy()
    years_keep = [y for y in YEARS if (dfm['year'] == y).sum() > 1000 and dfm[dfm['year'] == y]['outcome_cat'].var() > 0]
    dropped_years = list(set(YEARS) - set(years_keep))
    log(f"MNLogit dropping years (size/variance): {dropped_years}")
    dfm = dfm[dfm['year'].isin(years_keep)]
    fml = "outcome_cat ~ rep_attorney + rep_agent + rep_vso + rep_generic + len_chars_z + hearing + ama_cue + C(year)"
    m = MNLogit.from_formula(fml, data=dfm)
    try:
        r = m.fit(disp=False, maxiter=200)
        used = "mnlogit_default"
    except np.linalg.LinAlgError as e:
        log(f"MNLogit failed: {e} - falling back to separate logits vs deny")
        df_gd = dfm[dfm['outcome_cat'].isin([0,1])].copy()
        df_gd['grant_vs_deny'] = (df_gd['outcome_cat'] == 0).astype(int)
        fml_g = "grant_vs_deny ~ rep_attorney + rep_agent + rep_vso + rep_generic + len_chars_z + hearing + ama_cue + C(year)"
        mg = smf.logit(fml_g, data=df_gd)
        df_rd = dfm[dfm['outcome_cat'].isin([1,2])].copy()
        df_rd['remand_vs_deny'] = (df_rd['outcome_cat'] == 2).astype(int)
        fml_r = "remand_vs_deny ~ rep_attorney + rep_agent + rep_vso + rep_generic + len_chars_z + hearing + ama_cue + C(year)"
        mr = smf.logit(fml_r, data=df_rd)
        try:
            rg = mg.fit(disp=False, cov_type="cluster", cov_kwds={"groups": df_gd["year"]})
            rr = mr.fit(disp=False, cov_type="cluster", cov_kwds={"groups": df_rd["year"]})
            used = "logit_cluster (fallback)"
            se_boot_g, p_boot_g = wild_bootstrap(rg, df_gd, "year")
            rg.bse[:] = se_boot_g
            rg.pvalues[:] = p_boot_g
            se_boot_r, p_boot_r = wild_bootstrap(rr, df_rd, "year")
            rr.bse[:] = se_boot_r
            rr.pvalues[:] = p_boot_r
        except Exception as e:
            log(f"Fallback cluster failed: {e} - using default")
            rg = mg.fit(disp=False, method="lbfgs", maxiter=300)
            rr = mr.fit(disp=False, method="lbfgs", maxiter=300)
            used = "logit_default (fallback)"
        rows = []
        for outcome, res in zip(['grant_vs_deny', 'remand_vs_deny'], [rg, rr]):
            for name, coef, se, p in zip(res.params.index, res.params.values, res.bse, res.pvalues):
                if name.startswith("C(year)"): continue
                z = coef / se if se > 0 else np.nan
                rrr = safe_exp(coef)
                lower = safe_exp(coef - 1.96 * se) if se > 0 else np.nan
                upper = safe_exp(coef + 1.96 * se) if se > 0 else np.nan
                rows.append([outcome, name, coef, se, lower, upper, z, p, rrr, used])
    else:
        rows = []
        for i in range(r.params.shape[1]):
            outcome = f"outcome_{i}_vs_deny"
            for name, coef, se, p in zip(r.params.index, r.params.iloc[:, i], r.bse.iloc[:, i], r.pvalues.iloc[:, i]):
                if name.startswith("C(year)"): continue
                z = coef / se if se > 0 else np.nan
                rrr = safe_exp(coef)
                lower = safe_exp(coef - 1.96 * se) if se > 0 else np.nan
                upper = safe_exp(coef + 1.96 * se) if se > 0 else np.nan
                rows.append([outcome, name, coef, se, lower, upper, z, p, rrr, used])
    write_csv(outp, header, rows)
    s5p = os.path.join(OUTD, "table_s5_small_cells.grok.csv")
    s5_header = ["year", "reason", "n_excluded"]
    s5_rows = [[y, "small size or zero variance", (df['year'] == y).sum()] for y in dropped_years]
    write_csv(s5p, s5_header, s5_rows)
    log(f"Multinomial complete for {tag}")

def fit_heterogeneity(df, tag):
    outp = os.path.join(OUTD, f"table3_heterogeneity.grok.{tag}.csv")
    if os.path.exists(outp):
        log(f"Skipping {os.path.basename(outp)} - already exists")
        return
    header = ["term","coef","se","lower_ci","upper_ci","z","pvalue","odds_ratio","note"]
    if df.empty:
        write_csv(outp, header, [])
        return
    dfh = df.copy()
    years_keep = [y for y in YEARS if (dfh['year'] == y).sum() > 1000 and dfh[dfh['year'] == y]['grant_bin'].var() > 0]
    log(f"Heterogeneity dropping years (size/variance): {list(set(YEARS) - set(years_keep))}")
    dfh = dfh[dfh['year'].isin(years_keep)]
    fml = "grant_bin ~ rep_attorney * (hearing + ama_cue + len_chars_z) + rep_agent * (hearing + ama_cue + len_chars_z) + rep_vso * (hearing + ama_cue + len_chars_z) + rep_generic * (hearing + ama_cue + len_chars_z) + C(year)"
    m = smf.logit(fml, data=dfh)
    try:
        r = m.fit(disp=False, cov_type="cluster", cov_kwds={"groups": dfh["year"]})
        used = "logit_cluster"
        se_boot, p_boot = wild_bootstrap(r, dfh, "year")
        r.bse[:] = se_boot
        r.pvalues[:] = p_boot
    except Exception as e:
        log(f"Heterogeneity fit failed: {e} - using default")
        r = m.fit(disp=False, method="lbfgs", maxiter=300)
        used = "logit_default"
    rows = []
    for name, coef, se, p in zip(r.params.index, r.params.values, r.bse, r.pvalues):
        if name.startswith("C(year)"): continue
        z = coef / se if se > 0 else np.nan
        odds = safe_exp(coef)
        lower = safe_exp(coef - 1.96 * se) if se > 0 else np.nan
        upper = safe_exp(coef + 1.96 * se) if se > 0 else np.nan
        rows.append([name, coef, se, lower, upper, z, p, odds, used])
    write_csv(outp, header, rows)
    log(f"Heterogeneity complete for {tag}")

def fit_covariate_specs(df, tag):
    specs = {
        'a': 'grant_bin ~ rep_attorney + rep_agent + rep_vso + C(year)',
        'b': 'grant_bin ~ rep_attorney + rep_agent + rep_vso + hearing + len_chars_clip + ama_cue + C(year)',
    }
    for key, fml in specs.items():
        outp = os.path.join(OUTD, f"table_s3_spec_{key}.grok.{tag}.csv")
        if os.path.exists(outp):
            log(f"Skipping {os.path.basename(outp)} - already exists")
            continue
        header = ["term","coef","se","lower_ci","upper_ci","z","pvalue","odds_ratio","note"]
        m = smf.logit(fml, data=df)
        try:
            r = m.fit(disp=False, cov_type="cluster", cov_kwds={"groups": df["year"]})
            used = "logit_cluster"
            se_boot, p_boot = wild_bootstrap(r, df, "year")
            r.bse[:] = se_boot
            r.pvalues[:] = p_boot
        except Exception as e:
            log(f"Spec {key} failed: {e} - using default")
            r = m.fit(disp=False, method="lbfgs", maxiter=300)
            used = "logit_default"
        rows = []
        for name, coef, se, p in zip(r.params.index, r.params.values, r.bse, r.pvalues):
            if name.startswith("C(year)"): continue
            z = coef / se if se > 0 else np.nan
            odds = safe_exp(coef)
            lower = safe_exp(coef - 1.96 * se) if se > 0 else np.nan
            upper = safe_exp(coef + 1.96 * se) if se > 0 else np.nan
            rows.append([name, coef, se, lower, upper, z, p, odds, used])
        write_csv(outp, header, rows)
        log(f"Spec {key} complete for {tag}")

def fit_outcome_sensitivity(df, tag):
    outcomes = ['grant_only', 'grant_remand']
    for outc in outcomes:
        outp = os.path.join(OUTD, f"table_sens_{outc}.grok.{tag}.csv")
        if os.path.exists(outp):
            log(f"Skipping {os.path.basename(outp)} - already exists")
            continue
        header = ["term","coef","se","lower_ci","upper_ci","z","pvalue","odds_ratio","note"]
        fml = f"{outc} ~ rep_attorney + rep_agent + rep_vso + len_chars_clip + hearing + ama_cue + C(year)"
        m = smf.logit(fml, data=df)
        try:
            r = m.fit(disp=False, cov_type="cluster", cov_kwds={"groups": df["year"]})
            used = "logit_cluster"
            se_boot, p_boot = wild_bootstrap(r, df, "year")
            r.bse[:] = se_boot
            r.pvalues[:] = p_boot
        except Exception as e:
            log(f"Sensitivity {outc} failed: {e} - using default")
            r = m.fit(disp=False, method="lbfgs", maxiter=300)
            used = "logit_default"
        rows = []
        for name, coef, se, p in zip(r.params.index, r.params.values, r.bse, r.pvalues):
            if name.startswith("C(year)"): continue
            z = coef / se if se > 0 else np.nan
            odds = safe_exp(coef)
            lower = safe_exp(coef - 1.96 * se) if se > 0 else np.nan
            upper = safe_exp(coef + 1.96 * se) if se > 0 else np.nan
            rows.append([name, coef, se, lower, upper, z, p, odds, used])
        write_csv(outp, header, rows)
        log(f"Sensitivity {outc} complete for {tag}")

def fit_multi_rep_sensitivity(df, tag):
    multi_freq = df['multi_rep'].sum() / len(df) * 100
    log(f"Multi-rep frequency: {multi_freq:.1f}% (n={df['multi_rep'].sum()})")
    with open(os.path.join(OUTD, "multi_rep_summary.grok.txt"), "w") as f:
        f.write(f"Multi-rep frequency: {multi_freq:.1f}% (n={df['multi_rep'].sum()})\n")
    df_dropped = df[~df['multi_rep']]
    log(f"Dropped multi-rep: {len(df) - len(df_dropped):,} cases")
    fit_multinomial(df_dropped, f"{tag}_dropped_multi")
    outp = os.path.join(OUTD, f"table2_logit_grant_bin_dropped_multi.grok.{tag}.csv")
    if os.path.exists(outp):
        log(f"Skipping {os.path.basename(outp)} - already exists")
        return
    header = ["term","coef","se","lower_ci","upper_ci","z","pvalue","odds_ratio","note"]
    fml = "grant_bin ~ rep_attorney + rep_agent + rep_vso + len_chars_clip + hearing + ama_cue + C(year)"
    m = smf.logit(fml, data=df_dropped)
    try:
        r = m.fit(disp=False, cov_type="cluster", cov_kwds={"groups": df_dropped["year"]})
        used = "logit_cluster"
        se_boot, p_boot = wild_bootstrap(r, df_dropped, "year")
        r.bse[:] = se_boot
        r.pvalues[:] = p_boot
    except Exception as e:
        log(f"Multi-rep logit failed: {e} - using default")
        r = m.fit(disp=False, method="lbfgs", maxiter=300)
        used = "logit_default"
    rows = []
    for name, coef, se, p in zip(r.params.index, r.params.values, r.bse, r.pvalues):
        if name.startswith("C(year)"): continue
        z = coef / se if se > 0 else np.nan
        odds = safe_exp(coef)
        lower = safe_exp(coef - 1.96 * se) if se > 0 else np.nan
        upper = safe_exp(coef + 1.96 * se) if se > 0 else np.nan
        rows.append([name, coef, se, lower, upper, z, p, odds, used])
    write_csv(outp, header, rows)
    log(f"Multi-rep sensitivity complete for {tag}")

def fit_ama_split(df, tag):
    df_pre = df[df['ama_cue'] == 0]
    df_post = df[df['ama_cue'] == 1]
    log(f"Pre-AMA N: {len(df_pre):,}; Post-AMA N: {len(df_post):,}")
    fit_multinomial(df_pre, f"{tag}_pre_ama")
    fit_multinomial(df_post, f"{tag}_post_ama")
    fit_ipw_binary(df_post, f"{tag}_post_ama")
    log(f"AMA split complete for {tag}")

def fit_ipw_binary(df, tag):
    outp = os.path.join(OUTD, f"table3_ipw_binary.grok.{tag}.csv")
    if os.path.exists(outp):
        log(f"Skipping {os.path.basename(outp)} - already exists")
        return
    header = ["term","coef","se","lower_ci","upper_ci","z","pvalue","odds_ratio","note"]
    if df.empty:
        write_csv(outp, header, [])
        return
    dfw, bal, summ = fit_propensity_and_weights(df)
    bal.to_csv(os.path.join(OUTD, "balance_before_after.grok.csv"), index=False)
    with open(os.path.join(OUTD, "weights_summary.grok.txt"), "w", encoding=ENC) as f:
        f.write(summ + "\n")
        f.write(f"w min/med/max: {dfw['w'].min():.3f}/{dfw['w'].median():.3f}/{dfw['w'].max():.3f}\n")
    fml = "grant_bin ~ rep_attorney + rep_agent + rep_vso + rep_generic + len_chars_z + hearing + ama_cue + C(year)"
    glm = smf.glm(formula=fml, data=dfw, family=sm.families.Binomial(), freq_weights=dfw["w"])
    try:
        r = glm.fit(cov_type="cluster", cov_kwds={"groups": dfw["year"]})
        used = "glm_binom_cluster_ipw"
        se_boot, p_boot = wild_bootstrap(r, dfw, "year")
        r.bse[:] = se_boot
        r.pvalues[:] = p_boot
    except Exception as e:
        log(f"IPW cluster failed: {e} - using HC1")
        r = glm.fit(cov_type="HC1")
        used = "glm_binom_HC1_ipw"
    rows = []
    for name, coef, se, p in zip(r.params.index, r.params.values, r.bse, r.pvalues):
        if name.startswith("C(year)"): continue
        z = coef / se if se > 0 else np.nan
        odds = safe_exp(coef)
        lower = safe_exp(coef - 1.96 * se) if se > 0 else np.nan
        upper = safe_exp(coef + 1.96 * se) if se > 0 else np.nan
        rows.append([name, coef, se, lower, upper, z, p, odds, used])
    write_csv(outp, header, rows)
    log(f"IPW binary complete for {tag}")

def fit_propensity_and_weights(df):
    fml_ps = "rep_any ~ len_chars_z + hearing + ama_cue + C(year)"
    psm = smf.logit(fml_ps, data=df)
    try:
        psr = psm.fit(disp=False)
        ps = psr.predict(df)
        summ = "Propensity model converged"
    except np.linalg.LinAlgError as e:
        log(f"IPW propensity failed: {e} - dropping ama_cue")
        fml_ps = "rep_any ~ len_chars_z + hearing + C(year)"
        psm = smf.logit(fml_ps, data=df)
        psr = psm.fit(disp=False)
        ps = psr.predict(df)
        summ = "Propensity model converged after dropping ama_cue"
    df['ps'] = ps
    pt = df['rep_any'].mean()
    df['w'] = np.where(df['rep_any'] == 1, pt / df['ps'], (1 - pt) / (1 - df['ps']))
    trim_lo, trim_hi = df['w'].quantile([IPW_TRIM_LO, IPW_TRIM_HI])
    df['w'] = df['w'].clip(lower=trim_lo, upper=trim_hi)
    summ += f"; pt={pt:.3f}; weights trimmed to [{trim_lo:.3f},{trim_hi:.3f}]"
    covs = ['len_chars_z', 'hearing', 'ama_cue']
    smd_pre = [(df[cov][df['rep_any']==1].mean() - df[cov][df['rep_any']==0].mean()) / np.sqrt((df[cov][df['rep_any']==1].var() + df[cov][df['rep_any']==0].var())/2) for cov in covs if cov in df]
    smd_post = []
    for cov in covs:
        if cov not in df: continue
        mu1 = np.average(df[cov][df['rep_any']==1], weights=df['w'][df['rep_any']==1])
        mu0 = np.average(df[cov][df['rep_any']==0], weights=df['w'][df['rep_any']==0])
        var1 = np.average((df[cov][df['rep_any']==1] - mu1)**2, weights=df['w'][df['rep_any']==1])
        var0 = np.average((df[cov][df['rep_any']==0] - mu0)**2, weights=df['w'][df['rep_any']==0])
        smd = (mu1 - mu0) / np.sqrt((var1 + var0)/2) if (var1 + var0) > 0 else np.nan
        smd_post.append(smd)
    bal = pd.DataFrame({'covariate': covs, 'smd_pre': smd_pre, 'smd_post': smd_post})
    ess = (df['w'].sum() ** 2) / (df['w'] ** 2).sum()
    summ += f"\nESS: {ess:.0f} ({ess / len(df):.2%} of original)"
    plt.figure()
    plt.hist(df['ps'][df['rep_any']==1], alpha=0.5, label='Represented', bins=30)
    plt.hist(df['ps'][df['rep_any']==0], alpha=0.5, label='Pro se', bins=30)
    plt.legend()
    plt.title('Propensity Score Overlap')
    try:
        plt.savefig(os.path.join(OUTD, 'prop_score_overlap.grok.png'))
        log("Wrote prop_score_overlap.grok.png")
    except Exception as e:
        log(f"Failed to save overlap plot: {e}")
    plt.close()
    return df, bal, summ

def generate_flow_diagram():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sankey = Sankey(ax=ax, scale=1.0 / 1327419)
    sankey.add(flows=[1327419, -435568, -1191446],
               labels=['Initial Public Records', 'Excluded (Duplicates, Missing, Suspect)', 'Full Cohort'],
               orientations=[0, 1, -1],
               trunklength=0.5)
    sankey.finish()
    plt.title('Cohort Flow Diagram')
    try:
        plt.savefig(os.path.join(OUTD, 'cohort_flow.grok.png'))
        log("Wrote cohort_flow.grok.png")
    except Exception as e:
        log(f"Failed to save cohort flow: {e}")
    plt.close()

def main():
    log("Starting analysis...")
    if not os.path.exists(DB):
        log(f"ERROR: DB missing: {DB}")
        return
    try:
        con = sqlite3.connect(DB, timeout=60.0)
        log(f"Connected to DB: {DB}")
    except Exception as e:
        log(f"DB connection failed: {e}")
        return
    df = load_checkpoint_if_exists()
    if df is None:
        log("No checkpoint - fetching and building DataFrame...")
        df = build_df(fetch_cases(con, ns_only=True))
        if df.empty:
            log("No data - exiting")
            con.close()
            return
    log(f"Using DataFrame with {len(df):,} rows")
    generate_flow_diagram()
    fit_multinomial(df, "NS")
    fit_heterogeneity(df, "NS")
    fit_covariate_specs(df, "NS")
    fit_outcome_sensitivity(df, "NS")
    fit_multi_rep_sensitivity(df, "NS")
    fit_ama_split(df, "NS")
    fit_ipw_binary(df, "NS")
    con.close()
    log("Done analysis - check {ROOT}/index/Grok for outputs")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user - safe to restart")
    except Exception as e:
        log(f"FATAL: {traceback.format_exc()}")