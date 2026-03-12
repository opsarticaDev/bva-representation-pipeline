# pyw\bva_audit_and_snapshot.rev21.pyw
# rev21: Audit sample + results merge + pretty balance + replication bundle (zip)
# - Non-destructive; reads your existing SQLite and rev18/19 outputs
# - Writes everything under {BVA_ROOT}/index/rev21/
# - Deterministic (fixed RNG seed)

import os, sys, csv, sqlite3, zipfile, datetime, math, random, shutil, traceback

ROOT   = os.environ.get("BVA_ROOT", ".")
DB     = os.path.join(ROOT, "parsed", "parsed.full.rev01.sqlite")
IDX    = os.path.join(ROOT, "index")
OUTD   = os.path.join(IDX, "rev21")
REV18D = os.path.join(IDX, "tables_rev18")
REV19D = os.path.join(IDX, "tables_rev19")
FIGS   = os.path.join(IDX, "figs_2010_2025")  # include if present
LOGP   = os.path.join(OUTD, "rev21_build.log.txt")
ENC    = "utf-8"
os.makedirs(OUTD, exist_ok=True)

# ---------- small logger ----------
def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOGP, "a", encoding=ENC) as f:
        f.write(f"{ts} | {msg}\n")

# ---------- safe import ----------
def ensure_pd():
    try:
        import pandas as pd
        return pd
    except Exception:
        raise SystemExit("pandas is required for rev21")

pd = ensure_pd()

RNG = random.Random(20210921)  # deterministic seed

# ---------- helpers ----------
def q(cur, sql, params=()):
    cur.execute(sql, params)
    return cur.fetchall()

def write_csv(path, header, rows):
    with open(path, "w", encoding=ENC, newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    log(f"Wrote {path}")

# ---------- 1) stratified audit sample (n=300) ----------
def build_audit_sample():
    """Stratify by (year bucket) × (rep_group) × (disposition group). Sample ~300 proportionally."""
    con = sqlite3.connect(DB, timeout=60.0)
    cur = con.cursor()
    cur.execute("PRAGMA busy_timeout=60000;")

    # Pull NS cohort (same gates as rev18/19)
    # rep_group: attorney/agent/vso/prose_unknown (represented_generic folded into represented)
    rows = q(cur, """
        SELECT p.case_id,
               TRIM(p.decision_year) AS year,
               LOWER(p.rep_type)     AS rep_type,
               q.primary_disposition AS disp,
               p.source_path         AS path
          FROM parsed p
          JOIN qc q ON q.case_id = p.case_id
         WHERE LOWER(p.file_ext) = '.txt'
           AND (q.is_struct_suspect = 0 OR q.is_struct_suspect = '0')
    """)

    con.close()
    if not rows:
        raise SystemExit("No NS rows found for audit sampling.")

    # Bucket year coarsely to spread review
    def year_bucket(y):
        try: y = int(y)
        except: return "other"
        if 2010 <= y <= 2013: return "2010-2013"
        if 2014 <= y <= 2017: return "2014-2017"
        if y == 2018:         return "2018"
        if 2019 <= y <= 2020: return "2019-2020"
        if 2021 <= y <= 2025: return "2021-2025"
        return "other"

    def rep_group(rt):
        rt = (rt or "").strip().lower()
        if rt in ("attorney",): return "attorney"
        if rt in ("agent",):    return "agent"
        if rt in ("vso",):      return "vso"
        # represented_generic and unknown collapse into pro se/unknown for audit check on header accuracy
        return "prose_or_unknown"

    def disp_group(d):
        d = (d or "unknown").strip().lower()
        if d in ("grant",):   return "grant"
        if d in ("deny",):    return "deny"
        if d in ("remand",):  return "remand"
        if d in ("dismiss",): return "dismiss"
        return "other"

    records = []
    for cid, y, rt, disp, path in rows:
        records.append({
            "case_id": cid,
            "year": y,
            "year_bucket": year_bucket(y),
            "rep_type": (rt or ""),
            "rep_group": rep_group(rt),
            "primary_disposition": (disp or ""),
            "disp_group": disp_group(disp),
            "source_path": path
        })

    df = pd.DataFrame.from_records(records)
    N = len(df)
    log(f"NS rows available for audit sampling: {N:,}")

    # Stratification
    df["stratum"] = df["year_bucket"] + " | " + df["rep_group"] + " | " + df["disp_group"]
    counts = df["stratum"].value_counts().to_dict()

    target_n = 300
    # Proportional allocation with floor=1 for strata with at least 10 rows
    picks = {}
    for s, c in counts.items():
        if c <= 0: continue
        n_s = max(1 if c >= 10 else 0, round(target_n * (c / N)))
        picks[s] = n_s

    # Adjust to hit exactly 300
    current = sum(picks.values())
    # If too few, top up biggest strata; if too many, trim smallest n>1 strata
    if current < target_n:
        for s,_ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            if current >= target_n: break
            picks[s] = picks.get(s,0) + 1
            current += 1
    elif current > target_n:
        for s,_ in sorted(counts.items(), key=lambda kv: kv[1]):  # trim small first
            if current <= target_n: break
            if picks.get(s,0) > 1:
                picks[s] -= 1
                current -= 1

    # Sample within each stratum deterministically
    sample_rows = []
    for s, n_s in picks.items():
        if n_s <= 0: continue
        block = df[df["stratum"] == s]
        if len(block) == 0: continue
        # deterministic sample using our RNG
        idxs = list(block.index)
        RNG.shuffle(idxs)
        take = idxs[:min(n_s, len(idxs))]
        sample_rows.extend(block.loc[take].to_dict("records"))

    out_cols = [
        "case_id","source_path","year","year_bucket",
        "rep_type","rep_group","primary_disposition","disp_group","stratum"
    ]
    out_path = os.path.join(OUTD, "audit_sample_300.rev21.csv")
    pd.DataFrame(sample_rows, columns=out_cols).to_csv(out_path, index=False, encoding=ENC)
    log(f"Audit sample written: {out_path}")
    return out_path

# ---------- 2) audit template & merge ----------
def make_audit_template(sample_csv):
    tpl = os.path.join(OUTD, "audit_results_template.rev21.csv")
    df = pd.read_csv(sample_csv, encoding=ENC)
    # Add reviewer fields
    for col in ["header_ok","rep_line_correct","outcome_correct","notes"]:
        df[col] = ""
    df.to_csv(tpl, index=False, encoding=ENC)
    log(f"Audit template written: {tpl}")
    return tpl

def merge_and_score_if_present(sample_csv):
    filled = os.path.join(OUTD, "audit_results.rev21.csv")
    if not os.path.exists(filled):
        log("audit_results.rev21.csv not found: skipping merge/score for now.")
        return None
    sdf = pd.read_csv(sample_csv, encoding=ENC)
    rdf = pd.read_csv(filled, encoding=ENC)
    # Basic validation: must contain same case_ids
    if "case_id" not in rdf.columns:
        log("audit_results.rev21.csv missing 'case_id': cannot merge.")
        return None
    m = sdf.merge(rdf, on="case_id", how="left", suffixes=("",""))
    # Normalize reviewer booleans
    def as_bool(x):
        s = str(x).strip().lower()
        return 1 if s in ("1","true","t","y","yes","ok","correct") else 0
    for col in ("header_ok","rep_line_correct","outcome_correct"):
        if col in m.columns:
            m[col] = m[col].map(as_bool)
        else:
            m[col] = 0
    n = len(m)
    rates = {
        "header_ok_rate":          m["header_ok"].mean() if n else float("nan"),
        "rep_line_correct_rate":   m["rep_line_correct"].mean() if n else float("nan"),
        "outcome_correct_rate":    m["outcome_correct"].mean() if n else float("nan"),
    }
    # Stratified rates (by rep_group and year_bucket)
    grp = m.groupby(["rep_group","year_bucket"])[["header_ok","rep_line_correct","outcome_correct"]].mean().reset_index()
    # Write markdown report
    md = os.path.join(OUTD, "audit_report.rev21.md")
    with open(md, "w", encoding=ENC) as f:
        f.write("# Audit Report: rev21\n\n")
        f.write(f"Sample size: **{n}**\n\n")
        f.write("## Overall accuracy\n\n")
        for k,v in rates.items():
            f.write(f"- {k.replace('_',' ')}: **{v:.3f}**\n")
        f.write("\n## Accuracy by rep group × year bucket\n\n")
        f.write("| rep_group | year_bucket | header_ok | rep_line_correct | outcome_correct |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for _,r in grp.iterrows():
            f.write(f"| {r['rep_group']} | {r['year_bucket']} | {r['header_ok']:.3f} | {r['rep_line_correct']:.3f} | {r['outcome_correct']:.3f} |\n")
    log(f"Audit report written: {md}")
    return md

# ---------- 3) pretty balance table ----------
def make_balance_md():
    src = os.path.join(REV19D, "balance_before_after.rev19.csv")
    if not os.path.exists(src):
        log("rev19 balance file not found: skipping balance_table_pretty.")
        return None
    df = pd.read_csv(src, encoding=ENC)
    outp = os.path.join(OUTD, "balance_table_pretty.rev21.md")
    with open(outp, "w", encoding=ENC) as f:
        f.write("# Covariate Balance (IPW): rev19\n\n")
        f.write("_Standardized Mean Differences (SMD); |SMD| < 0.10 is commonly considered acceptable._\n\n")
        f.write("| Covariate | SMD (pre) | SMD (post) | Improvement |\n")
        f.write("|---|---:|---:|---:|\n")
        for _,r in df.iterrows():
            pre = r.get("smd_pre", float("nan"))
            post= r.get("smd_post", float("nan"))
            imp = abs(pre) - abs(post) if all(map(math.isfinite,[pre,post])) else float("nan")
            f.write(f"| {r['covariate']} | {pre:.3f} | {post:.3f} | {imp:.3f} |\n")
    log(f"Pretty balance table written: {outp}")
    return outp

# ---------- 4) replication bundle ----------
def build_replication_zip():
    zip_path = os.path.join(OUTD, "replication_bundle.rev21.zip")
    manifest = []

    def add_if_exists(zf, abs_path, arcname):
        if os.path.exists(abs_path):
            zf.write(abs_path, arcname)
            manifest.append(arcname)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Methods & codebook if present
        add_if_exists(zf, os.path.join(IDX, "codebook.rev01.json"), "methods/codebook.rev01.json")

        # Tables & logs
        if os.path.isdir(REV18D):
            for name in os.listdir(REV18D):
                add_if_exists(zf, os.path.join(REV18D, name), f"tables_rev18/{name}")
        if os.path.isdir(REV19D):
            for name in os.listdir(REV19D):
                add_if_exists(zf, os.path.join(REV19D, name), f"tables_rev19/{name}")

        # rev21 outputs
        for name in os.listdir(OUTD):
            if name.endswith(".csv") or name.endswith(".md") or name.endswith(".txt"):
                add_if_exists(zf, os.path.join(OUTD, name), f"rev21/{name}")

        # Figures (only small pointers; avoid huge zips: include SVG/PDF manifests if present)
        if os.path.isdir(FIGS):
            for name in os.listdir(FIGS):
                if name.endswith(".svg") or name.endswith(".pdf") or name.endswith(".txt"):
                    add_if_exists(zf, os.path.join(FIGS, name), f"figs_2010_2025/{name}")

        # Logs (rev18/19)
        for p in [os.path.join(REV18D, "rev18_build.log.txt"),
                  os.path.join(REV19D, "rev19_build.log.txt")]:
            if os.path.exists(p):
                add_if_exists(zf, p, os.path.relpath(p, IDX))

        # Add a manifest file inside the zip
        manifest_txt = "MANIFEST.rev21.txt"
        with open(os.path.join(OUTD, manifest_txt), "w", encoding=ENC) as mf:
            mf.write("# Replication bundle manifest: rev21\n")
            for item in manifest:
                mf.write(item + "\n")
        zf.write(os.path.join(OUTD, manifest_txt), f"rev21/{manifest_txt}")
    log(f"Replication bundle written: {zip_path}")
    return zip_path

# ---------------- MAIN ----------------
def main():
    log("Starting rev21…")
    if not os.path.exists(DB):
        log(f"FATAL: DB missing: {DB}")
        return

    # 1) Sample
    sample_csv = build_audit_sample()

    # 2) Template + (optional) merge
    tpl_csv = make_audit_template(sample_csv)
    merge_and_score_if_present(sample_csv)  # runs only if you already saved audit_results.rev21.csv

    # 3) Pretty balance table
    make_balance_md()

    # 4) Replication zip
    build_replication_zip()

    log("Done rev21.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(LOGP, "a", encoding=ENC) as f:
            f.write("FATAL:\n"+traceback.format_exc()+"\n")
