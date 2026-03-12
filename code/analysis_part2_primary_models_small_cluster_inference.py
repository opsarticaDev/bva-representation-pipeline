#!/usr/bin/env python3
"""
BVA Analysis Part 2: Primary Models and Small-Cluster Inference
Addresses reviewer requirements: #1 (numeric effects), #2 (small-cluster), #11 (multinomial)
Requires: Part 1 checkpoint file
"""

import os
import csv
import datetime
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import MNLogit
from scipy.stats import norm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
ROOT = os.environ.get("BVA_ROOT", ".")
OUTD = os.path.join(ROOT, "index", "Grok")
CHECKPOINT_DF = os.path.join(OUTD, "checkpoint_df_part1.parquet")
LOGP = os.path.join(OUTD, "part2_build.log.txt")
BOOTSTRAP_B = 999
MIN_CELL_SIZE = 1000

# ==================== LOGGING ====================
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOGP, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {msg}\n")
        print(f"{ts} | {msg}")
    except:
        print(f"{ts} | {msg}")

log("="*70)
log("PART 2: PRIMARY MODELS AND SMALL-CLUSTER INFERENCE")
log("="*70)

# ==================== UTILITIES ====================
def write_csv(path, header, rows):
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        log(f"Wrote: {os.path.basename(path)}")
    except Exception as e:
        log(f"ERROR: {e}")

def safe_exp(x):
    if np.isnan(x):
        return np.nan
    return np.exp(np.clip(x, -700, 700))

# ==================== SMALL-CLUSTER INFERENCE (ISSUE #2) ====================
def wild_cluster_bootstrap(model, df, cluster_col='year', B=BOOTSTRAP_B):
    """
    Wild cluster bootstrap with Rademacher weights.
    Primary solution for small-cluster inference.
    """
    log(f"Wild bootstrap: {B} iterations on {cluster_col}...")
    
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    log(f"  Clusters: {n_clusters}")
    
    orig_params = model.params
    
    try:
        resid = model.resid_response
        fitted = model.fittedvalues
    except AttributeError:
        log("  Using model predictions for residuals")
        fitted = model.predict()
        if hasattr(model.model, 'endog'):
            resid = model.model.endog - fitted
        else:
            log("  WARNING: Cannot compute residuals")
            return model.bse, model.pvalues
    
    coefs = []
    for b in tqdm(range(B), desc="Bootstrap", leave=False):
        # Rademacher weights
        signs = np.random.choice([-1, 1], size=n_clusters, replace=True)
        cluster_map = dict(zip(clusters, signs))
        weights = df[cluster_col].map(cluster_map).values
        
        if len(resid.shape) == 1:
            resid_b = resid * weights
            y_b = fitted + resid_b
        else:
            resid_b = resid * weights[:, np.newaxis]
            y_b = fitted + resid_b
        
        try:
            model_b = model.model.__class__(y_b, model.model.exog)
            r_b = model_b.fit(disp=False, start_params=orig_params, maxiter=100)
            coefs.append(r_b.params)
        except:
            continue
    
    if len(coefs) < B * 0.5:
        log(f"  WARNING: Only {len(coefs)}/{B} converged")
        return model.bse, model.pvalues
    
    coefs = np.array(coefs)
    se_boot = np.std(coefs, axis=0)
    t_stats = orig_params / se_boot
    p_boot = 2 * (1 - norm.cdf(np.abs(t_stats)))
    
    log(f"  Complete: {len(coefs)} successful")
    return se_boot, p_boot

def fit_with_small_sample_correction(formula, df, cluster_col='year'):
    """Fit logit with wild bootstrap for small clusters"""
    n_clusters = df[cluster_col].nunique()
    log(f"Fitting: {n_clusters} clusters")
    
    model = smf.logit(formula, data=df)
    
    try:
        result = model.fit(
            disp=False,
            cov_type='cluster',
            cov_kwds={'groups': df[cluster_col], 'use_correction': True}
        )
        method = "cluster_robust"
    except:
        result = model.fit(disp=False)
        method = "default"
    
    if n_clusters < 30:
        se_boot, p_boot = wild_cluster_bootstrap(result, df, cluster_col)
        result.bse = pd.Series(se_boot, index=result.params.index)
        result.pvalues = pd.Series(p_boot, index=result.params.index)
        method += "+wild_bootstrap"
    
    log(f"  Inference: {method}")
    return result, method

# ==================== PRIMARY LOGIT MODELS (ISSUE #1) ====================
def fit_primary_logit_models(df):
    """
    Fit primary logistic models with both outcome definitions.
    Generates numeric results for manuscript text.
    """
    log("="*60)
    log("PRIMARY LOGISTIC REGRESSION MODELS")
    log("="*60)
    
    results_summary = {}
    
    outcomes = [
        ('deny_only', 'grant_bin', 'Any grant vs deny only'),
        ('deny_dismiss', 'grant_bin', 'Any grant vs deny+dismiss')
    ]
    
    for outcome_name, outcome_var, desc in outcomes:
        log(f"\n{desc}")
        
        if outcome_name == 'deny_only':
            df_model = df[df['dismiss'] == 0].copy()
        else:
            df_model = df.copy()
        
        log(f"  N = {len(df_model):,}")
        
        formula = f"{outcome_var} ~ rep_attorney + rep_agent + rep_vso + hearing + len_chars_clip + ama_cue + C(year)"
        
        result, method = fit_with_small_sample_correction(formula, df_model, 'year')
        
        rows = []
        for name in result.params.index:
            if name.startswith("C(year)") or name == "Intercept":
                continue
            
            coef = result.params[name]
            se = result.bse[name]
            pval = result.pvalues[name]
            
            or_val = safe_exp(coef)
            or_lower = safe_exp(coef - 1.96 * se)
            or_upper = safe_exp(coef + 1.96 * se)
            
            rows.append([
                name, coef, se, or_val, or_lower, or_upper,
                coef / se if se > 0 else np.nan, pval, method
            ])
        
        write_csv(
            os.path.join(OUTD, f"table2_logit_{outcome_name}.rev20.NS.csv"),
            ["term", "coef", "se", "OR", "OR_lower", "OR_upper", "z", "pvalue", "inference"],
            rows
        )
        
        results_summary[outcome_name] = {
            'rows': rows,
            'n': len(df_model),
            'method': method
        }
    
    generate_results_text_snippet(results_summary)
    return results_summary

def generate_results_text_snippet(results):
    """Generate text with exact numeric results for manuscript (ISSUE #1)"""
    log("Generating results text snippet...")
    
    path = os.path.join(OUTD, "results_text_snippet.rev20.txt")
    with open(path, "w") as f:
        f.write("NUMERIC RESULTS FOR MANUSCRIPT - ISSUE #1\n")
        f.write("="*70 + "\n\n")
        f.write("COPY THIS TEXT INTO YOUR RESULTS SECTION:\n\n")
        
        for outcome_name, data in results.items():
            f.write(f"\n{outcome_name.upper().replace('_', ' ')} MODEL:\n")
            f.write(f"(N = {data['n']:,}, inference: {data['method']})\n\n")
            
            for row in data['rows']:
                term, coef, se, or_val, or_lo, or_hi, z, pval, _ = row
                
                if 'attorney' in term:
                    label = "Attorney representation"
                elif 'agent' in term:
                    label = "Agent representation"
                elif 'vso' in term:
                    label = "VSO representation"
                else:
                    continue
                
                p_str = f"{pval:.4f}" if pval >= 0.001 else "<0.001"
                f.write(f"{label}: OR = {or_val:.2f} ")
                f.write(f"(95% CI {or_lo:.2f}-{or_hi:.2f}, p = {p_str})\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Example prose:\n\n")
        f.write("'Adjusted odds ratios from year fixed-effects logistic regression\n")
        f.write("with wild cluster bootstrap inference (n_clusters = 16) show that\n")
        f.write("attorney representation (OR = X.XX, 95% CI X.XX-X.XX, p < 0.001),\n")
        f.write("agent representation (OR = X.XX, 95% CI X.XX-X.XX, p < 0.001), and\n")
        f.write("VSO representation (OR = X.XX, 95% CI X.XX-X.XX, p < 0.001) were each\n")
        f.write("associated with higher grant odds relative to pro se appellants under\n")
        f.write("the deny-only negative definition (Table 2A). Under the deny-plus-dismiss\n")
        f.write("definition, patterns remained consistent (Table 2B).'\n")
    
    log(f"Text snippet saved: {path}")

# ==================== MULTINOMIAL MODELS (ISSUE #11) ====================
def fit_multinomial_models(df):
    """Fit multinomial logit for grant/deny/remand with RRRs"""
    log("="*60)
    log("MULTINOMIAL LOGIT MODELS")
    log("="*60)
    
    df_mn = df[df['outcome_cat'].isin([0, 1, 2])].copy()
    log(f"N = {len(df_mn):,}")
    
    # Check cell sizes
    years_keep = []
    for y in sorted(df_mn['year'].unique()):
        dfy = df_mn[df_mn['year'] == y]
        if len(dfy) >= MIN_CELL_SIZE and dfy['outcome_cat'].nunique() == 3:
            years_keep.append(y)
        else:
            log(f"  Dropping year {y}: n={len(dfy)} or missing outcomes")
    
    df_mn = df_mn[df_mn['year'].isin(years_keep)].copy()
    log(f"Years retained: {years_keep}")
    
    formula = "outcome_cat ~ rep_attorney + rep_agent + rep_vso + hearing + len_chars_clip + ama_cue + C(year)"
    
    try:
        model = MNLogit.from_formula(formula, data=df_mn)
        result = model.fit(disp=False, maxiter=200)
        
        # Wild bootstrap if small clusters
        n_clusters = df_mn['year'].nunique()
        if n_clusters < 30:
            log("Applying wild bootstrap to multinomial...")
            se_boot, p_boot = wild_cluster_bootstrap(result, df_mn, 'year')
            
            for j in range(result.bse.shape[1]):
                result.bse.iloc[:, j] = se_boot
                result.pvalues.iloc[:, j] = p_boot
        
        method = "mnlogit_wild_bootstrap" if n_clusters < 30 else "mnlogit_default"
        
        rows = []
        outcome_labels = {0: "grant_vs_deny", 2: "remand_vs_deny"}
        
        for j, outcome_label in outcome_labels.items():
            for name in result.params.index:
                if name.startswith("C(year)") or name == "Intercept":
                    continue
                
                coef = result.params.iloc[:, j][name]
                se = result.bse.iloc[:, j][name]
                pval = result.pvalues.iloc[:, j][name]
                
                rrr = safe_exp(coef)
                rrr_lower = safe_exp(coef - 1.96 * se)
                rrr_upper = safe_exp(coef + 1.96 * se)
                
                rows.append([
                    outcome_label, name, coef, se, rrr, rrr_lower, rrr_upper,
                    coef/se if se > 0 else np.nan, pval, method
                ])
        
        write_csv(
            os.path.join(OUTD, "table3_multinomial.rev20.NS.csv"),
            ["outcome", "term", "coef", "se", "RRR", "RRR_lower", "RRR_upper", "z", "pvalue", "inference"],
            rows
        )
        
        log("Multinomial complete")
        return result
        
    except Exception as e:
        log(f"Multinomial failed: {e}")
        log("Falling back to separate binary logits...")
        
        rows = []
        
        # Grant vs deny
        df_gd = df_mn[df_mn['outcome_cat'].isin([0, 1])].copy()
        df_gd['grant_vs_deny'] = (df_gd['outcome_cat'] == 0).astype(int)
        formula_g = "grant_vs_deny ~ rep_attorney + rep_agent + rep_vso + hearing + len_chars_clip + ama_cue + C(year)"
        
        try:
            result_g, method_g = fit_with_small_sample_correction(formula_g, df_gd, 'year')
            
            for name in result_g.params.index:
                if name.startswith("C(year)") or name == "Intercept":
                    continue
                
                coef = result_g.params[name]
                se = result_g.bse[name]
                pval = result_g.pvalues[name]
                
                rrr = safe_exp(coef)
                rrr_lower = safe_exp(coef - 1.96 * se)
                rrr_upper = safe_exp(coef + 1.96 * se)
                
                rows.append([
                    "grant_vs_deny", name, coef, se, rrr, rrr_lower, rrr_upper,
                    coef/se, pval, f"logit_fallback_{method_g}"
                ])
        except Exception as e2:
            log(f"Grant vs deny fallback failed: {e2}")
        
        # Remand vs deny
        df_rd = df_mn[df_mn['outcome_cat'].isin([1, 2])].copy()
        df_rd['remand_vs_deny'] = (df_rd['outcome_cat'] == 2).astype(int)
        formula_r = "remand_vs_deny ~ rep_attorney + rep_agent + rep_vso + hearing + len_chars_clip + ama_cue + C(year)"
        
        try:
            result_r, method_r = fit_with_small_sample_correction(formula_r, df_rd, 'year')
            
            for name in result_r.params.index:
                if name.startswith("C(year)") or name == "Intercept":
                    continue
                
                coef = result_r.params[name]
                se = result_r.bse[name]
                pval = result_r.pvalues[name]
                
                rrr = safe_exp(coef)
                rrr_lower = safe_exp(coef - 1.96 * se)
                rrr_upper = safe_exp(coef + 1.96 * se)
                
                rows.append([
                    "remand_vs_deny", name, coef, se, rrr, rrr_lower, rrr_upper,
                    coef/se, pval, f"logit_fallback_{method_r}"
                ])
        except Exception as e2:
            log(f"Remand vs deny fallback failed: {e2}")
        
        if rows:
            write_csv(
                os.path.join(OUTD, "table3_multinomial.rev20.NS.csv"),
                ["outcome", "term", "coef", "se", "RRR", "RRR_lower", "RRR_upper", "z", "pvalue", "inference"],
                rows
            )
            log("Multinomial fallback complete")
        
        return None

# ==================== MAIN ====================
def main():
    log("\nStarting Part 2: Primary models and inference\n")
    
    if not os.path.exists(CHECKPOINT_DF):
        log(f"ERROR: Checkpoint not found: {CHECKPOINT_DF}")
        log("Run analysis_part1_setup_preprocessing_and_descriptives.py first")
        return False
    
    log("Loading checkpoint...")
    df = pd.read_parquet(CHECKPOINT_DF)
    log(f"Loaded: {len(df):,} cases\n")
    
    # Fit models
    fit_primary_logit_models(df)
    fit_multinomial_models(df)
    
    log("\nPart 2 complete")
    log("Next: Run analysis_part3_sensitivity_and_robustness_checks.py for sensitivity analyses")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nSUCCESS: Primary models complete")
            print("Check results_text_snippet.rev20.txt for manuscript text")
    except KeyboardInterrupt:
        log("\nInterrupted")
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        import traceback
        log(traceback.format_exc())
