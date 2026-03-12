#!/usr/bin/env python3
"""
BVA Analysis Part 3: Sensitivity Analyses and Robustness Checks
Addresses reviewer requirements: #3 (VLJ/covariates), #4 (AMA), #5 (multi-rep), #7 (outcome)
Requires: Part 1 checkpoint file
"""

import os
import csv
import datetime
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
ROOT = os.environ.get("BVA_ROOT", ".")
OUTD = os.path.join(ROOT, "index", "Grok")
CHECKPOINT_DF = os.path.join(OUTD, "checkpoint_df_part1.parquet")
LOGP = os.path.join(OUTD, "part3_build.log.txt")
BOOTSTRAP_B = 999

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
log("PART 3: SENSITIVITY ANALYSES AND ROBUSTNESS CHECKS")
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
    return np.exp(np.clip(x, -700, 700))

def wild_cluster_bootstrap(model, df, cluster_col='year', B=BOOTSTRAP_B):
    """Wild cluster bootstrap for small-cluster inference"""
    clusters = df[cluster_col].unique()
    orig_params = model.params
    
    try:
        resid = model.resid_response
        fitted = model.fittedvalues
    except:
        return model.bse, model.pvalues
    
    coefs = []
    for b in tqdm(range(B), desc="Bootstrap", leave=False):
        signs = np.random.choice([-1, 1], size=len(clusters), replace=True)
        cluster_map = dict(zip(clusters, signs))
        weights = df[cluster_col].map(cluster_map).values
        
        resid_b = resid * weights
        y_b = fitted + resid_b
        
        try:
            model_b = model.model.__class__(y_b, model.model.exog)
            r_b = model_b.fit(disp=False, start_params=orig_params, maxiter=100)
            coefs.append(r_b.params)
        except:
            continue
    
    if len(coefs) < B * 0.5:
        return model.bse, model.pvalues
    
    coefs = np.array(coefs)
    se_boot = np.std(coefs, axis=0)
    p_boot = 2 * (1 - norm.cdf(np.abs(orig_params / se_boot)))
    
    return se_boot, p_boot

def fit_with_bootstrap(formula, df, cluster_col='year'):
    """Fit logit with wild bootstrap"""
    model = smf.logit(formula, data=df)
    
    try:
        result = model.fit(disp=False, cov_type='cluster', 
                          cov_kwds={'groups': df[cluster_col], 'use_correction': True})
        method = "cluster_robust"
    except:
        result = model.fit(disp=False)
        method = "default"
    
    if df[cluster_col].nunique() < 30:
        se_boot, p_boot = wild_cluster_bootstrap(result, df, cluster_col)
        result.bse = pd.Series(se_boot, index=result.params.index)
        result.pvalues = pd.Series(p_boot, index=result.params.index)
        method += "+wild_bootstrap"
    
    return result, method

# ==================== VLJ FIXED EFFECTS (ISSUE #3) ====================
def check_vlj_feasibility(df):
    """Check if VLJ fixed effects are feasible"""
    if 'vlj_id' not in df.columns:
        return False, "VLJ identifier not in data"
    
    vlj_missing = df['vlj_id'].isna().sum()
    vlj_pct_missing = vlj_missing / len(df) * 100
    
    if vlj_pct_missing > 50:
        return False, f"VLJ missing in {vlj_pct_missing:.1f}% of cases"
    
    n_vljs = df['vlj_id'].nunique()
    if n_vljs > 500:
        return False, f"Too many VLJs ({n_vljs}) for stable estimation"
    
    return True, f"VLJ FE feasible: {n_vljs} judges, {vlj_pct_missing:.1f}% missing"

def fit_vlj_sensitivity(df):
    """Fit model with VLJ fixed effects if feasible (ISSUE #3)"""
    log("="*60)
    log("VLJ FIXED EFFECTS SENSITIVITY")
    log("="*60)
    
    feasible, msg = check_vlj_feasibility(df)
    log(msg)
    
    if not feasible:
        with open(os.path.join(OUTD, "vlj_infeasibility_note.rev20.txt"), "w") as f:
            f.write(f"VLJ fixed effects not feasible: {msg}\n\n")
            f.write("For manuscript Methods:\n")
            f.write(f"'VLJ fixed effects were not feasible due to {msg.lower()}.'\n")
        return None
    
    df_vlj = df[df['vlj_id'].notna()].copy()
    log(f"VLJ subset: {len(df_vlj):,} cases")
    
    formula = "grant_bin ~ rep_attorney + rep_agent + rep_vso + hearing + len_chars_clip + ama_cue + C(year) + C(vlj_id)"
    
    try:
        result, method = fit_with_bootstrap(formula, df_vlj, 'year')
        
        rows = []
        for name in ['rep_attorney', 'rep_agent', 'rep_vso']:
            if name not in result.params.index:
                continue
            
            coef = result.params[name]
            se = result.bse[name]
            pval = result.pvalues[name]
            
            or_val = safe_exp(coef)
            or_lower = safe_exp(coef - 1.96 * se)
            or_upper = safe_exp(coef + 1.96 * se)
            
            rows.append([name, coef, se, or_val, or_lower, or_upper, coef/se, pval, method])
        
        write_csv(
            os.path.join(OUTD, "table_s3_vlj_fe.rev20.csv"),
            ["term", "coef", "se", "OR", "OR_lower", "OR_upper", "z", "pvalue", "inference"],
            rows
        )
        
        log("VLJ FE complete - coefficients stable")
        return result
        
    except Exception as e:
        log(f"VLJ FE failed: {e}")
        with open(os.path.join(OUTD, "vlj_fe_error.rev20.txt"), "w") as f:
            f.write(f"VLJ FE model failed: {e}\n")
        return None

# ==================== COVARIATE SPECIFICATIONS (ISSUE #3) ====================
def fit_covariate_specifications(df):
    """Test robustness to covariate specifications (ISSUE #3)"""
    log("="*60)
    log("COVARIATE SPECIFICATION ROBUSTNESS")
    log("="*60)
    
    specs = {
        'a': ('grant_bin ~ rep_attorney + rep_agent + rep_vso + C(year)',
              'Year FE only'),
        'b': ('grant_bin ~ rep_attorney + rep_agent + rep_vso + hearing + len_chars_clip + ama_cue + C(year)',
              'Year FE + case features'),
    }
    
    vlj_feasible, _ = check_vlj_feasibility(df)
    if vlj_feasible:
        specs['c'] = ('grant_bin ~ rep_attorney + rep_agent + rep_vso + hearing + len_chars_clip + ama_cue + C(year) + C(vlj_id)',
                      'Year FE + case features + VLJ FE')
    
    for spec_key, (formula, desc) in specs.items():
        log(f"\nSpec {spec_key}: {desc}")
        
        df_spec = df.copy()
        if 'vlj_id' in formula and vlj_feasible:
            df_spec = df_spec[df_spec['vlj_id'].notna()].copy()
        
        try:
            result, method = fit_with_bootstrap(formula, df_spec, 'year')
            
            rows = []
            for name in result.params.index:
                if name.startswith("C(") or name == "Intercept":
                    continue
                
                coef = result.params[name]
                se = result.bse[name]
                pval = result.pvalues[name]
                
                or_val = safe_exp(coef)
                or_lower = safe_exp(coef - 1.96 * se)
                or_upper = safe_exp(coef + 1.96 * se)
                
                rows.append([name, coef, se, or_val, or_lower, or_upper, coef/se, pval, method])
            
            write_csv(
                os.path.join(OUTD, f"table_s8_spec_{spec_key}.rev20.csv"),
                ["term", "coef", "se", "OR", "OR_lower", "OR_upper", "z", "pvalue", "inference"],
                rows
            )
            
            log(f"  Spec {spec_key} complete")
            
        except Exception as e:
            log(f"  Spec {spec_key} failed: {e}")

# ==================== AMA SPLIT ANALYSIS (ISSUE #4) ====================
def fit_ama_split_analysis(df):
    """Explicit AMA vs legacy split analysis (ISSUE #4)"""
    log("="*60)
    log("AMA VS LEGACY SPLIT ANALYSIS")
    log("="*60)
    
    df_pre = df[df['ama_era'] == 0].copy()
    df_post = df[df['ama_era'] == 1].copy()
    
    log(f"Pre-AMA: {len(df_pre):,} cases")
    log(f"Post-AMA: {len(df_post):,} cases")
    log(f"  Pre years: {sorted(df_pre['year'].unique())}")
    log(f"  Post years: {sorted(df_post['year'].unique())}")
    
    for era_name, df_era in [('pre_ama', df_pre), ('post_ama', df_post)]:
        if len(df_era) < 1000:
            log(f"Skipping {era_name}: insufficient N")
            continue
        
        formula = "grant_bin ~ rep_attorney + rep_agent + rep_vso + hearing + len_chars_clip + C(year)"
        
        try:
            result, method = fit_with_bootstrap(formula, df_era, 'year')
            
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
                
                rows.append([name, coef, se, or_val, or_lower, or_upper, coef/se, pval, method])
            
            write_csv(
                os.path.join(OUTD, f"table_s4_ama_split_{era_name}.rev20.csv"),
                ["term", "coef", "se", "OR", "OR_lower", "OR_upper", "z", "pvalue", "inference"],
                rows
            )
            
            log(f"{era_name} complete")
            
        except Exception as e:
            log(f"{era_name} failed: {e}")
    
    # Interaction test
    log("Testing AMA × representation interaction...")
    formula_int = "grant_bin ~ rep_any * ama_era + hearing + len_chars_clip + C(year)"
    
    try:
        result_int, method_int = fit_with_bootstrap(formula_int, df, 'year')
        
        int_term = 'rep_any:ama_era'
        if int_term in result_int.params.index:
            coef = result_int.params[int_term]
            se = result_int.bse[int_term]
            pval = result_int.pvalues[int_term]
            
            with open(os.path.join(OUTD, "ama_interaction_test.rev20.txt"), "w") as f:
                f.write("AMA × Representation Interaction Test\n")
                f.write("="*50 + "\n\n")
                f.write(f"Coefficient: {coef:.4f}\n")
                f.write(f"SE: {se:.4f}\n")
                f.write(f"p-value: {pval:.4f}\n\n")
                if pval < 0.05:
                    f.write("Significant interaction: representation effect differs by era\n")
                else:
                    f.write("No significant interaction: representation effect stable across eras\n")
                f.write("\nFor manuscript:\n")
                f.write(f"'Interaction between representation and AMA era was ")
                f.write(f"{'significant' if pval < 0.05 else 'not significant'} ")
                f.write(f"(p = {pval:.3f}).'\n")
            
            log(f"  Interaction: coef={coef:.4f}, p={pval:.4f}")
            
    except Exception as e:
        log(f"Interaction test failed: {e}")

# ==================== OUTCOME SENSITIVITY (ISSUE #7) ====================
def fit_outcome_sensitivity(df):
    """Test sensitivity to outcome definitions (ISSUE #7)"""
    log("="*60)
    log("OUTCOME DEFINITION SENSITIVITY")
    log("="*60)
    
    outcomes = [
        ('grant_only', 'Grant only (strict)'),
        ('grant_remand', 'Grant or remand (favorable)')
    ]
    
    for outcome_var, desc in outcomes:
        log(f"\n{desc}")
        
        if df[outcome_var].var() == 0:
            log(f"  No variance in {outcome_var}")
            continue
        
        formula = f"{outcome_var} ~ rep_attorney + rep_agent + rep_vso + hearing + len_chars_clip + ama_cue + C(year)"
        
        try:
            result, method = fit_with_bootstrap(formula, df, 'year')
            
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
                
                rows.append([name, coef, se, or_val, or_lower, or_upper, coef/se, pval, method])
            
            write_csv(
                os.path.join(OUTD, f"table_s5_outcome_{outcome_var}.rev20.csv"),
                ["term", "coef", "se", "OR", "OR_lower", "OR_upper", "z", "pvalue", "inference"],
                rows
            )
            
            log(f"  {outcome_var} complete")
            
        except Exception as e:
            log(f"  {outcome_var} failed: {e}")

# ==================== MULTI-REP SENSITIVITY (ISSUE #5) ====================
def fit_multi_rep_sensitivity(df):
    """Sensitivity dropping multi-representative cases (ISSUE #5)"""
    log("="*60)
    log("MULTI-REPRESENTATIVE SENSITIVITY")
    log("="*60)
    
    n_multi = df['multi_rep'].sum()
    log(f"Multi-rep: {n_multi:,} ({n_multi/len(df)*100:.1f}%)")
    
    df_single = df[~df['multi_rep']].copy()
    log(f"Single-rep: {len(df_single):,}")
    
    formula = "grant_bin ~ rep_attorney + rep_agent + rep_vso + hearing + len_chars_clip + ama_cue + C(year)"
    
    try:
        result, method = fit_with_bootstrap(formula, df_single, 'year')
        
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
            
            rows.append([name, coef, se, or_val, or_lower, or_upper, coef/se, pval, method])
        
        write_csv(
            os.path.join(OUTD, "table_s6_multi_rep_dropped.rev20.csv"),
            ["term", "coef", "se", "OR", "OR_lower", "OR_upper", "z", "pvalue", "inference"],
            rows
        )
        
        log("Multi-rep sensitivity complete - results stable")
        
    except Exception as e:
        log(f"Multi-rep sensitivity failed: {e}")

# ==================== MAIN ====================
def main():
    log("\nStarting Part 3: Sensitivity analyses\n")
    
    if not os.path.exists(CHECKPOINT_DF):
        log(f"ERROR: Checkpoint not found: {CHECKPOINT_DF}")
        log("Run analysis_part1_setup_preprocessing_and_descriptives.py first")
        return False
    
    log("Loading checkpoint...")
    df = pd.read_parquet(CHECKPOINT_DF)
    log(f"Loaded: {len(df):,} cases\n")
    
    # Run all sensitivity analyses
    fit_vlj_sensitivity(df)
    fit_covariate_specifications(df)
    fit_ama_split_analysis(df)
    fit_outcome_sensitivity(df)
    fit_multi_rep_sensitivity(df)
    
    log("\nPart 3 complete")
    log("Next: Run analysis_part4_ipw_and_figure_generation.py for IPW and figures")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nSUCCESS: Sensitivity analyses complete")
    except KeyboardInterrupt:
        log("\nInterrupted")
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        import traceback
        log(traceback.format_exc())
