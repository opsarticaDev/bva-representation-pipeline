# bva_make_stats_2010_2025.rev01.pyw
# Zero-dependency stats pack for 2010-2025 clean rev03 CSVs.
# Outputs CSVs and a findings text file you can paste into the study.

import os, csv, math, datetime, tkinter as tk
from tkinter.scrolledtext import ScrolledText

ROOT = os.environ.get("BVA_ROOT", ".")
IDX  = os.path.join(ROOT, "index")
OUT  = os.path.join(IDX, "stats_2010_2025")
os.makedirs(OUT, exist_ok=True)

F_REP_ANY = os.path.join(IDX, "rep_presence_by_decision_year.clean.2010_2025.rev03.csv")
F_REP_TYP = os.path.join(IDX, "rep_type_share_by_year.clean.2010_2025.rev03.csv")
F_DISP    = os.path.join(IDX, "disposition_by_decision_year.clean.2010_2025.rev03.csv")
F_NEG     = os.path.join(IDX, "negative_rate_by_decision_year.clean.2010_2025.rev03.csv")

def rint(x): 
    try: return int(x)
    except: return None
def rfloat(x):
    try: return float(x)
    except: return None

def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def wcsv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

# ---------- simple OLS for y vs year ----------
def linreg_year(points):
    # points: list of (year:int, y:float), year treated as numeric
    xs = [float(x) for x,_ in points]
    ys = [float(y) for _,y in points]
    n = len(xs)
    if n < 3: 
        return {"slope":0.0, "intercept":ys[0] if ys else 0.0, "slope_se":None, "ci_low":None, "ci_high":None}
    xbar = sum(xs)/n; ybar = sum(ys)/n
    Sxx = sum((x-xbar)**2 for x in xs)
    Sxy = sum((x-xbar)*(y-ybar) for x,y in zip(xs,ys))
    slope = Sxy / (Sxx if Sxx else 1.0)
    intercept = ybar - slope*xbar
    # residuals and SE
    rss = sum((y - (intercept + slope*x))**2 for x,y in zip(xs,ys))
    df = max(1, n-2)
    s2 = rss/df
    slope_se = math.sqrt(s2 / (Sxx if Sxx else 1.0))
    # 95% CI using ~1.96 (approx)
    z = 1.96
    ci_low  = slope - z*slope_se
    ci_high = slope + z*slope_se
    return {"slope":slope, "intercept":intercept, "slope_se":slope_se, "ci_low":ci_low, "ci_high":ci_high}

# ---------- two-proportion z-test for YoY deltas ----------
def two_prop_z(n1, N1, n2, N2):
    # compare p2 - p1
    if N1==0 or N2==0: return 0.0, 1.0
    p1 = n1/N1; p2 = n2/N2
    p  = (n1+n2)/(N1+N2)
    denom = math.sqrt(p*(1-p)*(1/N1 + 1/N2)) if p*(1-p)>0 else 0.0
    if denom==0: return 0.0, 1.0
    z = (p2 - p1) / denom
    # two-sided p ~ normal
    pval = 2*(1 - 0.5*(1 + math.erf(abs(z)/math.sqrt(2))))
    return z, pval

def pct(x): 
    return f"{x*100:.1f}%"

def run(log):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"Starting stats build @ {ts}")

    # ---------- Load core tables ----------
    rep_any = read_csv(F_REP_ANY)
    neg     = read_csv(F_NEG)
    rep_typ = read_csv(F_REP_TYP)
    disp    = read_csv(F_DISP)

    # ---------- Build series (per year) ----------
    # Any rep presence
    rep_points = []  # (year, rate), and keep counts for YoY tests
    rep_counts = {}  # year -> (anyrep, total)
    for r in rep_any:
        y = rint(r["decision_year"])
        if y is None: continue
        rep_points.append((y, rfloat(r["rate_any_rep"]) or 0.0))
        rep_counts[y] = (int(float(r["any_rep_count"])), int(float(r["total_rows"])))
    rep_points.sort()

    # Negative rate (deny + dismiss)
    neg_points = []
    neg_counts = {}
    for r in neg:
        y = rint(r["decision_year"])
        if y is None: continue
        neg_points.append((y, rfloat(r["negative_rate"]) or 0.0))
        neg_counts[y] = (int(float(r["negative_count_deny_or_dismiss"])), int(float(r["total"])))
    neg_points.sort()

    # ---------- YoY tables ----------
    rep_rows = []
    for i,(y,rate) in enumerate(rep_points):
        anyrep, tot = rep_counts[y]
        if i==0:
            rep_rows.append([y, anyrep, tot, f"{rate:.3f}", "", "", ""])
        else:
            y0, _ = rep_points[i-1]
            n1,N1 = rep_counts[y0]
            z,pv = two_prop_z(n1,N1, anyrep, tot)
            delta = rate - rep_points[i-1][1]
            rep_rows.append([y, anyrep, tot, f"{rate:.3f}", f"{delta:+.3f}", f"{z:.2f}", f"{pv:.3f}"])
    wcsv(os.path.join(OUT,"rep_presence.stats.rev01.csv"),
         ["year","any_rep","total","rate","yoy_delta","yoy_z","yoy_p"], rep_rows)
    log("wrote rep_presence.stats.rev01.csv")

    neg_rows = []
    for i,(y,rate) in enumerate(neg_points):
        negc, tot = neg_counts[y]
        if i==0:
            neg_rows.append([y, negc, tot, f"{rate:.3f}", "", "", ""])
        else:
            y0,_ = neg_points[i-1]
            n1,N1 = neg_counts[y0]
            z,pv = two_prop_z(n1,N1, negc, tot)
            delta = rate - neg_points[i-1][1]
            neg_rows.append([y, negc, tot, f"{rate:.3f}", f"{delta:+.3f}", f"{z:.2f}", f"{pv:.3f}"])
    wcsv(os.path.join(OUT,"neg_rate.stats.rev01.csv"),
         ["year","deny_or_dismiss","total","rate","yoy_delta","yoy_z","yoy_p"], neg_rows)
    log("wrote neg_rate.stats.rev01.csv")

    # ---------- Rep type shares ----------
    # Just aggregate as given (already per-year shares). Also compute "represented share" = sum(attorney,agent,VSO)
    types = ["attorney","agent","VSO","none","unknown"]
    rep_year = {}
    for r in rep_typ:
        y = rint(r["decision_year"])
        if y is None: continue
        t = r["rep_type"]
        s = rfloat(r["share"]) or 0.0
        c = int(float(r["count"]))
        rec = rep_year.get(y, {"counts":{k:0 for k in types}, "shares":{k:0.0 for k in types}})
        if t in types:
            rec["counts"][t] = c
            rec["shares"][t] = s
        rep_year[y] = rec
    rep_type_rows=[]
    for y in sorted(rep_year.keys()):
        rec = rep_year[y]
        represented_share = rec["shares"]["attorney"]+rec["shares"]["agent"]+rec["shares"]["VSO"]
        rep_type_rows.append([y] + 
            [rec["counts"][k] for k in types] +
            [f"{rec['shares'][k]:.3f}" for k in types] +
            [f"{represented_share:.3f}"])
    wcsv(os.path.join(OUT,"rep_type.stats.rev01.csv"),
         ["year","attorney_n","agent_n","VSO_n","none_n","unknown_n",
          "attorney_share","agent_share","VSO_share","none_share","unknown_share","represented_share"], rep_type_rows)
    log("wrote rep_type.stats.rev01.csv")

    # ---------- Dispositions: counts + shares ----------
    disp_year = {}
    cats = ["grant","deny","dismiss","remand","vacate","withdraw","mixed","unknown"]
    for r in disp:
        y = rint(r["decision_year"])
        if y is None: continue
        d = r["primary_disposition"]
        n = int(float(r["n"]))
        rec = disp_year.get(y, {k:0 for k in cats})
        if d in rec:
            rec[d] += n
        disp_year[y] = rec
    disp_rows=[]
    for y in sorted(disp_year.keys()):
        rec = disp_year[y]
        tot = sum(rec[k] for k in cats)
        shares = [(rec[k]/tot if tot else 0.0) for k in cats]
        disp_rows.append([y] + [rec[k] for k in cats] + [tot] + [f"{s:.3f}" for s in shares])
    wcsv(os.path.join(OUT,"dispositions.stats.rev01.csv"),
         ["year"] + [f"{k}_n" for k in cats] + ["total"] + [f"{k}_share" for k in cats], disp_rows)
    log("wrote dispositions.stats.rev01.csv")

    # ---------- Linear trends with 95% CI ----------
    # slope units: change in rate per calendar year; also report per decade = slope*10
    trends=[]
    def trend_row(name, pts):
        if len(pts)>=3:
            t = linreg_year(pts)
            slope = t["slope"]; lo=t["ci_low"]; hi=t["ci_high"]
            trends.append([name, f"{slope:.5f}", f"{(slope*10):.5f}", 
                           f"{lo:.5f}" if lo is not None else "",
                           f"{hi:.5f}" if hi is not None else ""])
    trend_row("any_rep_rate", rep_points)
    trend_row("negative_rate", neg_points)
    wcsv(os.path.join(OUT,"trends.rev01.csv"),
         ["series","slope_per_year","slope_per_decade","ci95_low","ci95_high"], trends)
    log("wrote trends.rev01.csv")

    # ---------- Simple change-point candidates (max |YoY|) ----------
    def max_yoy(name, rows):
        # rows: list [year, ..., rate, delta, z, p]
        best = None
        for r in rows:
            year=r[0]; delta=r[4]
            if delta=="": continue
            d=float(delta)
            if (best is None) or (abs(d)>abs(best[1])):
                best=(year,d)
        return [name, best[0] if best else "", f"{best[1]:+.3f}" if best else ""]
    cp = []
    cp.append(max_yoy("any_rep_rate", rep_rows))
    cp.append(max_yoy("negative_rate", neg_rows))
    wcsv(os.path.join(OUT,"changepoints.rev01.csv"),
         ["series","year","max_abs_yoy_delta"], cp)
    log("wrote changepoints.rev01.csv")

    # ---------- Findings text ----------
    def first_last(points):
        return (points[0][0], points[0][1], points[-1][0], points[-1][1]) if points else ("","","","")
    ay0, av0, ay1, av1 = first_last(rep_points)
    ny0, nv0, ny1, nv1 = first_last(neg_points)

    # load slopes for prose
    any_slope = next((float(r[1]) for r in trends if r[0]=="any_rep_rate"), 0.0)
    any_lo    = next((float(r[3]) for r in trends if r[0]=="any_rep_rate" and r[3]!=""), None)
    any_hi    = next((float(r[4]) for r in trends if r[0]=="any_rep_rate" and r[4]!=""), None)
    neg_slope = next((float(r[1]) for r in trends if r[0]=="negative_rate"), 0.0)
    neg_lo    = next((float(r[3]) for r in trends if r[0]=="negative_rate" and r[3]!=""), None)
    neg_hi    = next((float(r[4]) for r in trends if r[0]=="negative_rate" and r[4]!=""), None)

    # biggest YoY deltas (for language)
    def biggest(rows):
        best=None
        for r in rows:
            if r[4]=="":
                continue
            y=r[0]; d=float(r[4]); z=r[5]; p=r[6]
            if (best is None) or abs(d)>abs(best[1]):
                best=(y,d,z,p)
        return best
    b_rep = biggest(rep_rows)
    b_neg = biggest(neg_rows)

    FTXT = os.path.join(OUT, "findings.rev01.txt")
    with open(FTXT, "w", encoding="utf-8") as f:
        f.write("BVA Study: 2010-2025 Clean (Non-Suspects)\n")
        f.write(f"Generated: {ts}\n\n")

        f.write("Headline metrics\n")
        f.write(f"• Any-rep rate: {ay0}={pct(av0)} → {ay1}={pct(av1)} "
                f"(slope≈{any_slope*100:+.2f} pp/year")
        if any_lo is not None and any_hi is not None:
            f.write(f", 95% CI [{any_lo*100:+.2f}, {any_hi*100:+.2f}] pp/year")
        f.write(")\n")
        f.write(f"• Negative rate (deny+dismiss): {ny0}={pct(nv0)} → {ny1}={pct(nv1)} "
                f"(slope≈{neg_slope*100:+.2f} pp/year")
        if neg_lo is not None and neg_hi is not None:
            f.write(f", 95% CI [{neg_lo*100:+.2f}, {neg_hi*100:+.2f}] pp/year")
        f.write(")\n\n")

        if b_rep:
            f.write(f"Largest YoY change: Any-rep: {int(b_rep[0])-1}→{b_rep[0]} "
                    f"{b_rep[1]*100:+.1f} pp (z={float(b_rep[2]):.2f}, p={float(b_rep[3]):.3f})\n")
        if b_neg:
            f.write(f"Largest YoY change: Negative rate: {int(b_neg[0])-1}→{b_neg[0]} "
                    f"{b_neg[1]*100:+.1f} pp (z={float(b_neg[2]):.2f}, p={float(b_neg[3]):.3f})\n")
        f.write("\n")

        f.write("Figure captions (drop-in)\n")
        f.write("Fig. 1: Any Representative Rate by Decision Year (2010-2025). "
                "Share of appeals in which the Veteran is represented by a VSO, accredited agent, or attorney.\n")
        f.write("Fig. 2: Negative Outcome Rate (Deny + Dismiss) by Year (2010-2025). "
                "Computed from Board decisions’ primary disposition; ‘dismiss’ is treated as negative.\n")
        f.write("Fig. 3: Representative Type Shares by Year (2010-2025). "
                "Stacked shares for VSOs, attorneys, and agents versus none/unknown.\n")
        f.write("Fig. 4: Dispositions by Year (2010-2025). "
                "Counts of grant, deny, dismiss, remand, vacate, withdraw, and mixed decisions.\n")

    log("wrote findings.rev01.txt")

    log("All stats complete.")

# ----- tiny UI -----
def pump():
    try:
        while True:
            msg = Q.get_nowait()
            LOG.insert(tk.END, msg+"\n"); LOG.see(tk.END)
    except queue.Empty:
        pass
    root.after(200, pump)

def ui_log(m):
    Q.put(m)

root = tk.Tk()
root.title("BVA Stats 2010-2025 • rev01")
root.geometry("760x360")
frm = tk.Frame(root); frm.pack(fill="both", expand=True, padx=10, pady=10)
tk.Label(frm, text=f"CSV folder: {IDX}").pack(anchor="w")
tk.Label(frm, text=f"Output folder: {OUT}").pack(anchor="w", pady=(0,6))
LOG = ScrolledText(frm, wrap="word", height=16); LOG.pack(fill="both", expand=True)

import queue, datetime
Q = queue.Queue()
root.after(200, lambda: run(ui_log))
root.after(250, pump)
root.mainloop()
