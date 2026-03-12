# bva_study_rollups_2010_2025.rev03.pyw
# Build study-ready CSVs for 2010-2025 using NON-SUSPECT rows only.

import os, sqlite3, csv, datetime, threading, queue, tkinter as tk
from tkinter.scrolledtext import ScrolledText

ROOT = os.environ.get("BVA_ROOT", ".")
DB   = os.path.join(ROOT, "parsed", "parsed.full.rev01.sqlite")
OUT  = os.path.join(ROOT, "index")

# Outputs (scoped to 2010-2025)
REP_ANY   = os.path.join(OUT, "rep_presence_by_decision_year.clean.2010_2025.rev03.csv")
REP_TYPE  = os.path.join(OUT, "rep_type_share_by_year.clean.2010_2025.rev03.csv")
DISP_YEAR = os.path.join(OUT, "disposition_by_decision_year.clean.2010_2025.rev03.csv")
NEG_YEAR  = os.path.join(OUT, "negative_rate_by_decision_year.clean.2010_2025.rev03.csv")
COVERAGE  = os.path.join(OUT, "coverage.clean.2010_2025.rev03.csv")
SAMPLE    = os.path.join(OUT, "sample_non_suspect_stratified.clean.2010_2025.rev03.csv")
YEARCHK   = os.path.join(OUT, "year_check.clean.2010_2025.rev03.txt")

ENC="utf-8"

def wcsv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding=ENC, newline="") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

WHERE_YEARS = """
  q.is_struct_suspect=0
  AND p.decision_year GLOB '[0-9][0-9][0-9][0-9]'
  AND CAST(p.decision_year AS INTEGER) BETWEEN 2010 AND 2025
"""

def build(q):
    if not os.path.exists(DB):
        q.put(f"Missing DB: {DB}"); return
    conn = sqlite3.connect(DB)
    cur  = conn.cursor()

    # ---- Year sanity
    cur.execute(f"""
        SELECT COUNT(*)
          FROM parsed p JOIN qc q ON q.case_id=p.case_id
         WHERE {WHERE_YEARS}
    """)
    total_clean = cur.fetchone()[0]

    cur.execute(f"""
        SELECT DISTINCT CAST(p.decision_year AS INTEGER)
          FROM parsed p JOIN qc q ON q.case_id=p.case_id
         WHERE {WHERE_YEARS}
         ORDER BY 1
    """)
    years = [r[0] for r in cur.fetchall()]

    with open(YEARCHK, "w", encoding=ENC) as f:
        f.write("Year Check (clean 2010-2025) rev03\n")
        f.write(f"Timestamp: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Non-suspect rows in window: {total_clean}\n")
        f.write(f"Distinct years ({len(years)}): {', '.join(str(y) for y in years)}\n")
        if years:
            f.write(f"Min year: {min(years)}\nMax year: {max(years)}\n")
    q.put(f"Wrote {YEARCHK}")

    # ---- Rep presence and type shares
    cur.execute(f"""
        SELECT p.decision_year, p.rep_type, COUNT(*)
          FROM parsed p JOIN qc q ON q.case_id=p.case_id
         WHERE {WHERE_YEARS}
         GROUP BY p.decision_year, p.rep_type
         ORDER BY CAST(p.decision_year AS INTEGER), p.rep_type
    """)
    rows = cur.fetchall()
    totals, anyrep, type_by_year = {}, {}, {}
    for y,t,n in rows:
        totals[y] = totals.get(y,0)+n
        if t in ("attorney","agent","VSO"):
            anyrep[y] = anyrep.get(y,0)+n
        type_by_year[(y,t)] = n

    rep_any_rows=[]
    for y in sorted(totals.keys(), key=lambda x:int(x)):
        tot=totals[y]; ar=anyrep.get(y,0)
        rep_any_rows.append([y, ar, tot, f"{(ar/tot) if tot else 0:.3f}"])
    wcsv(REP_ANY, ["decision_year","any_rep_count","total_rows","rate_any_rep"], rep_any_rows)

    rep_type_rows=[]
    for y in sorted(totals.keys(), key=lambda x:int(x)):
        tot=totals[y]
        for t in ("attorney","agent","VSO","none","unknown"):
            n=type_by_year.get((y,t),0)
            rep_type_rows.append([y, t, n, f"{(n/tot) if tot else 0:.3f}"])
    wcsv(REP_TYPE, ["decision_year","rep_type","count","share"], rep_type_rows)

    # ---- Dispositions (grant/deny/remand/dismiss/vacate/withdraw/mixed/unknown)
    cur.execute(f"""
        SELECT p.decision_year, q.primary_disposition, COUNT(*)
          FROM qc q JOIN parsed p ON p.case_id=q.case_id
         WHERE {WHERE_YEARS}
         GROUP BY p.decision_year, q.primary_disposition
         ORDER BY CAST(p.decision_year AS INTEGER), q.primary_disposition
    """)
    disp = cur.fetchall()
    wcsv(DISP_YEAR, ["decision_year","primary_disposition","n"], disp)

    # ---- Negative rate (deny + dismiss)
    cur.execute(f"""
        SELECT p.decision_year, 
               SUM(CASE WHEN q.deny_equivalent=1 THEN 1 ELSE 0 END),
               COUNT(*)
          FROM qc q JOIN parsed p ON p.case_id=q.case_id
         WHERE {WHERE_YEARS}
         GROUP BY p.decision_year
         ORDER BY CAST(p.decision_year AS INTEGER)
    """)
    neg = cur.fetchall()
    neg2=[(y, n, t, f"{(n/t) if t else 0:.3f}") for (y,n,t) in neg]
    wcsv(NEG_YEAR, ["decision_year","negative_count_deny_or_dismiss","total","negative_rate"], neg2)

    # ---- Coverage (in-window, non-suspects)
    cur.execute(f"""
        SELECT 
          SUM(CASE WHEN COALESCE(p.decision_year,'')<>'' THEN 1 ELSE 0 END),
          SUM(CASE WHEN COALESCE(p.docket_no,'')<>'' THEN 1 ELSE 0 END),
          SUM(CASE WHEN COALESCE(p.rep_type,'')<>'' THEN 1 ELSE 0 END),
          COUNT(*)
        FROM parsed p JOIN qc q ON q.case_id=p.case_id
        WHERE {WHERE_YEARS}
    """)
    dyear_ok, docket_ok, rep_ok, tot_ok = cur.fetchone()
    cov = [["decision_year_present", dyear_ok, f"{(dyear_ok/tot_ok) if tot_ok else 0:.3f}"],
           ["docket_present",       docket_ok, f"{(docket_ok/tot_ok) if tot_ok else 0:.3f}"],
           ["rep_type_present",     rep_ok,    f"{(rep_ok/tot_ok) if tot_ok else 0:.3f}"],
           ["rows_included",        tot_ok,    "1.000"]]
    wcsv(COVERAGE, ["field","count","share"], cov)

    # ---- Stratified spot-check sample (6 per year)
    sample=[]
    for y in years:
        cur.execute(f"""
            SELECT p.case_id, p.source_path, p.rep_type, q.primary_disposition
              FROM parsed p JOIN qc q ON q.case_id=p.case_id
             WHERE q.is_struct_suspect=0 AND CAST(p.decision_year AS INTEGER)=?
             ORDER BY RANDOM()
             LIMIT 6
        """, (y,))
        for row in cur.fetchall():
            sample.append((y,)+row)
    sample.sort(key=lambda r: r[0])
    wcsv(SAMPLE, ["decision_year","case_id","source_path","rep_type","primary_disposition"], sample)

    conn.close()
    q.put(f"Wrote:\n- {REP_ANY}\n- {REP_TYPE}\n- {DISP_YEAR}\n- {NEG_YEAR}\n- {COVERAGE}\n- {SAMPLE}")
    q.put("Done.")

def pump():
    try:
        while True:
            msg = Q.get_nowait()
            LOG.insert(tk.END, msg + "\n"); LOG.see(tk.END)
    except queue.Empty:
        pass
    root.after(200, pump)

root = tk.Tk()
root.title("BVA Study Rollups • 2010-2025 • rev03")
root.geometry("900x420")
frm = tk.Frame(root); frm.pack(fill="both", expand=True, padx=10, pady=10)
tk.Label(frm, text=f"DB: {DB}").pack(anchor="w", pady=(0,6))
LOG = ScrolledText(frm, wrap="word", height=20); LOG.pack(fill="both", expand=True)

Q = queue.Queue()
threading.Thread(target=build, args=(Q,), daemon=True).start()
root.after(200, pump)
root.mainloop()
