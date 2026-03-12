# bva_effect_sizes.rev16d.pyw
# Build a compact "effect-size" table from rev15 outputs (no DB, headless).
# Writes:
#   {BVA_ROOT}/index/figs_2010_2025/rev16/effect_sizes.rev16d.csv
#   {BVA_ROOT}/index/figs_2010_2025/rev16/effect_sizes.rev16d.txt

import os, csv, datetime

ROOT = os.environ.get("BVA_ROOT", ".")
SRC  = os.path.join(ROOT, "index")
OUT  = os.path.join(SRC, "figs_2010_2025", "rev16")
os.makedirs(OUT, exist_ok=True)

ENC = "utf-8"
YEARS = [2010, 2015, 2020, 2024, 2025]

# Source files (rev15)
NEG_NS_AS   = os.path.join(SRC, "negative_rate_by_decision_year.rev15.NS_D_as_neg.csv")
NEG_NS_SEP  = os.path.join(SRC, "negative_rate_by_decision_year.rev15.NS_D_separate.csv")
NEG_ALL_AS  = os.path.join(SRC, "negative_rate_by_decision_year.rev15.ALL_D_as_neg.csv")
NEG_ALL_SEP = os.path.join(SRC, "negative_rate_by_decision_year.rev15.ALL_D_separate.csv")
REP_NS      = os.path.join(SRC, "rep_presence_by_decision_year.rev15.NS_D_as_neg.csv")  # policy-independent
REP_ALL     = os.path.join(SRC, "rep_presence_by_decision_year.rev15.ALL_D_as_neg.csv")

def read_rate_csv(path, rate_col):
    """Return dict: year -> float(rate) from a rev15 CSV."""
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, "r", encoding=ENC) as f:
        r = csv.DictReader(f)
        for row in r:
            y = row.get("decision_year", "").strip()
            if not y.isdigit(): continue
            y = int(y)
            try:
                d[y] = float(row.get(rate_col, "0") or 0.0)
            except:
                d[y] = 0.0
    return d

def pp(x):  # percent points
    return round(100.0 * x, 1)

def main():
    # Load rates
    neg_ns_as   = read_rate_csv(NEG_NS_AS,   "rate_negative")
    neg_ns_sep  = read_rate_csv(NEG_NS_SEP,  "rate_negative")
    neg_all_as  = read_rate_csv(NEG_ALL_AS,  "rate_negative")
    neg_all_sep = read_rate_csv(NEG_ALL_SEP, "rate_negative")
    rep_ns      = read_rate_csv(REP_NS,      "rate_any_rep")
    rep_all     = read_rate_csv(REP_ALL,     "rate_any_rep")

    # Compose rows
    rows = []
    header = [
        "year",
        # Negative rate: baselines
        "NR_NS_asneg_%", "NR_NS_sep_%",
        "Δ_policy_pp",   # (NS_asneg - NS_sep) * 100
        "NR_ALL_asneg_%","NR_ALL_sep_%",
        "Δ_QC_pp",       # (ALL_asneg - NS_asneg) * 100
        # Representation presence
        "REP_NS_%", "REP_ALL_%",
        "Δ_rep_QC_pp"    # (ALL - NS) * 100
    ]
    for y in YEARS:
        ns_as   = neg_ns_as.get(y, 0.0)
        ns_sep  = neg_ns_sep.get(y, 0.0)
        all_as  = neg_all_as.get(y, 0.0)
        all_sep = neg_all_sep.get(y, 0.0)
        repns   = rep_ns.get(y, 0.0)
        repall  = rep_all.get(y, 0.0)

        delta_policy = pp(ns_as - ns_sep)
        delta_qc     = pp(all_as - ns_as)
        delta_rep_qc = pp(repall - repns)

        row = [
            y,
            pp(ns_as), pp(ns_sep),
            delta_policy,
            pp(all_as), pp(all_sep),
            delta_qc,
            pp(repns), pp(repall),
            delta_rep_qc
        ]
        rows.append(row)

    # Write CSV
    csv_path = os.path.join(OUT, "effect_sizes.rev16d.csv")
    with open(csv_path, "w", encoding=ENC, newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

    # Write a pretty text summary
    txt_path = os.path.join(OUT, "effect_sizes.rev16d.txt")
    lines = []
    lines.append(f"Effect-size table (built {datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    lines.append("Definitions:")
    lines.append("  Δ_policy_pp = (Negative rate NS_D_as_neg − NS_D_separate) in percentage points.")
    lines.append("  Δ_QC_pp     = (Negative rate ALL_D_as_neg − NS_D_as_neg) in percentage points.")
    lines.append("  Δ_rep_QC_pp = (Rep presence ALL − NS) in percentage points.")
    lines.append("")
    lines.append(" year | NR_NS_asneg% | NR_NS_sep% | Δ_policy_pp | NR_ALL_asneg% | NR_ALL_sep% | Δ_QC_pp | REP_NS% | REP_ALL% | Δ_rep_QC_pp")
    lines.append("-----+---------------+------------+-------------+---------------+-------------+---------+---------+----------+------------")
    for r in rows:
        lines.append(f"{r[0]:>4} | {r[1]:>13.1f} | {r[2]:>10.1f} | {r[3]:>11.1f} | {r[4]:>13.1f} | {r[5]:>11.1f} | {r[6]:>7.1f} | {r[7]:>7.1f} | {r[8]:>8.1f} | {r[9]:>10.1f}")
    with open(txt_path, "w", encoding=ENC) as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
