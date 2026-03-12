# bva_rev21_merge_only.pyw: robust merge/report for rev21 audit
import os, pandas as pd, math

_ROOT = os.environ.get("BVA_ROOT", ".")
OUTD = os.path.join(_ROOT, "index", "rev21")
ENC  = "utf-8"
sample_csv = os.path.join(OUTD, "audit_sample_300.rev21.csv")
results_csv= os.path.join(OUTD, "audit_results.rev21.csv")
report_md  = os.path.join(OUTD, "audit_report.rev21.md")

def norm_cols(df):
    m = {c.lower().strip(): c for c in df.columns}
    need = ["case_id","header_ok","rep_line_correct","outcome_correct","notes"]
    # Try to find reasonable fallbacks
    alias = {
        "header_ok": ["header ok","header_is_ok","header-correct","header_correct"],
        "rep_line_correct": ["rep_line_is_correct","rep line correct","rep_correct","representation_correct"],
        "outcome_correct": ["outcome is correct","outcome-correct","disposition_correct","primary_disposition_correct"],
        "notes": ["comment","remarks"]
    }
    out = {}
    for k in need:
        if k in m: out[k] = m[k]; continue
        found = None
        for a in alias.get(k, []):
            if a in m: found = m[a]; break
        if not found:
            # leave missing; we'll add empty
            out[k] = None
        else:
            out[k] = found
    return out

def as_bool(x):
    s = str(x).strip().lower()
    return 1 if s in ("1","true","t","y","yes","ok","correct") else 0

if not (os.path.exists(sample_csv) and os.path.exists(results_csv)):
    raise SystemExit("Missing sample or results CSV in index/rev21")

sdf = pd.read_csv(sample_csv, encoding=ENC)
rdf = pd.read_csv(results_csv, encoding=ENC)

# Normalize columns
maps = norm_cols(rdf)
for need, src in maps.items():
    if need == "case_id": continue
    if src is None:
        rdf[need] = ""
    elif need != src:
        rdf[need] = rdf[src]

if "case_id" not in rdf.columns:
    # try common variants
    for alt in ["CaseID","case id","id"]:
        if alt in rdf.columns: rdf["case_id"] = rdf[alt]
if "case_id" not in rdf.columns:
    raise SystemExit("audit_results is missing a case_id column")

m = sdf.merge(rdf[["case_id","header_ok","rep_line_correct","outcome_correct","notes"]], on="case_id", how="left")
for col in ["header_ok","rep_line_correct","outcome_correct"]:
    if col not in m.columns: m[col] = 0
    m[col] = m[col].map(as_bool)

n = len(m)
rates = {
    "header_ok_rate":        m["header_ok"].mean() if n else float("nan"),
    "rep_line_correct_rate": m["rep_line_correct"].mean() if n else float("nan"),
    "outcome_correct_rate":  m["outcome_correct"].mean() if n else float("nan"),
}

grp = m.groupby(["rep_group","year_bucket"], dropna=False)[["header_ok","rep_line_correct","outcome_correct"]].mean().reset_index()

with open(report_md, "w", encoding=ENC) as f:
    f.write("# Audit Report: rev21\n\n")
    f.write(f"Sample size: **{n}**\n\n")
    f.write("## Overall accuracy\n\n")
    for k,v in rates.items():
        vv = f"{v:.3f}" if isinstance(v, (int,float)) and math.isfinite(v) else "n/a"
        f.write(f"- {k.replace('_',' ')}: **{vv}**\n")
    f.write("\n## Accuracy by rep group × year bucket\n\n")
    f.write("| rep_group | year_bucket | header_ok | rep_line_correct | outcome_correct |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for _,r in grp.iterrows():
        f.write(f"| {r['rep_group']} | {r['year_bucket']} | {r['header_ok']:.3f} | {r['rep_line_correct']:.3f} | {r['outcome_correct']:.3f} |\n")

print("Wrote:", report_md)
