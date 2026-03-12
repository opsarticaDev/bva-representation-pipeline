# bva_figures_from_sensitivity.rev16c.pyw
# Robust figure builder: picks first writable output dir, logs early to index folder.
import os, sys, csv, datetime, traceback
sys.path.insert(0, os.path.join(os.environ.get("BVA_ROOT", "."), "_libs"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.environ.get("BVA_ROOT", ".")
SRC  = os.path.join(ROOT, "index")

# Early log (always under index so you see *something*)
EARLY_LOG = os.path.join(SRC, "figs_build.rev16c.bootstrap.log.txt")
def blog(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(SRC, exist_ok=True)
    with open(EARLY_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts} | {msg}\n")

blog("START rev16c")

# choose first writable OUT
CANDIDATE_OUTS = [
    os.path.join(SRC, "figs_2010_2025", "rev16"),
    os.path.join(SRC, "rev16_fallback"),
    os.path.join(ROOT, "output", "figures", "rev16"),
]
OUT = None
for path in CANDIDATE_OUTS:
    try:
        os.makedirs(path, exist_ok=True)
        testfile = os.path.join(path, "_write_test.txt")
        with open(testfile, "w", encoding="utf-8") as tf:
            tf.write("ok\n")
        os.remove(testfile)
        OUT = path
        blog(f"Selected OUT: {OUT}")
        break
    except Exception as e:
        blog(f"OUT not writable: {path} ({e})")

if not OUT:
    blog("ABORT: No writable output directory found.")
    raise SystemExit("No writable output directory found. See bootstrap log.")

LOG = os.path.join(OUT, "figs_build.rev16c.log.txt")
ENC = "utf-8"
YEARS = list(range(2010, 2026))

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts} | {msg}\n")

def read_neg(path):
    d={}
    with open(path,"r",encoding=ENC) as f:
        r=csv.DictReader(f)
        for row in r:
            y=row.get("decision_year","")
            if y.isdigit():
                d[int(y)]=(int(row["negative_count"]), int(row["total_rows"]), float(row["rate_negative"]))
    return d

def read_rep(path):
    d={}
    with open(path,"r",encoding=ENC) as f:
        r=csv.DictReader(f)
        for row in r:
            y=row.get("decision_year","")
            if y.isdigit():
                anyr=int(row["any_rep_count"]); tot=int(row["total_rows"])
                rate=float(row.get("rate_any_rep") or (anyr/tot if tot else 0))
                d[int(y)]=(anyr,tot,rate)
    return d

def read_disp(path):
    d={}
    with open(path,"r",encoding=ENC,newline="") as f:
        rd=csv.reader(f); header=next(rd,[])
        disp_cols=[]; i=1
        while i < len(header)-2:
            name=header[i].strip(); disp_cols.append((name,i+1)); i+=2
        for row in rd:
            if not row: continue
            y=row[0].strip()
            if not y.isdigit(): continue
            y=int(y); m={}
            for (name,idx) in disp_cols:
                try: m[name]=int(row[idx] or 0)
                except: m[name]=0
            d[y]=m
    return d

def pct(x): return 100.0*x

def save(fig, base):
    svg=os.path.join(OUT, base+".svg")
    png=os.path.join(OUT, base+".png")
    fig.tight_layout()
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {base}.svg/.png")

def main():
    try:
        log("START rev16c proper")
        needed = {
          "NEG_NS_AS":   os.path.join(SRC,"negative_rate_by_decision_year.rev15.NS_D_as_neg.csv"),
          "NEG_NS_SEP":  os.path.join(SRC,"negative_rate_by_decision_year.rev15.NS_D_separate.csv"),
          "NEG_ALL_AS":  os.path.join(SRC,"negative_rate_by_decision_year.rev15.ALL_D_as_neg.csv"),
          "NEG_ALL_SEP": os.path.join(SRC,"negative_rate_by_decision_year.rev15.ALL_D_separate.csv"),
          "REP_NS_AS":   os.path.join(SRC,"rep_presence_by_decision_year.rev15.NS_D_as_neg.csv"),
          "REP_ALL_AS":  os.path.join(SRC,"rep_presence_by_decision_year.rev15.ALL_D_as_neg.csv"),
          "DISP_NS_SEP": os.path.join(SRC,"dispositions_by_year.rev15.NS_D_separate.csv"),
          "DISP_ALL_SEP":os.path.join(SRC,"dispositions_by_year.rev15.ALL_D_separate.csv"),
        }
        for k,p in needed.items():
            log(f"CHECK {k}: {'OK' if os.path.exists(p) else 'MISSING'} | {p}")

        missing=[k for k,p in needed.items() if not os.path.exists(p)]
        if missing:
            log("ABORT: Missing sources: "+", ".join(missing))
            with open(os.path.join(OUT,"figs_readme.rev16.txt"),"w",encoding=ENC) as f:
                f.write("Missing sources:\n"); 
                for k in missing: f.write(f"{k} -> {needed[k]}\n")
            return

        # Load sources
        neg_ns_as   = read_neg(needed["NEG_NS_AS"])
        neg_ns_sep  = read_neg(needed["NEG_NS_SEP"])
        neg_all_as  = read_neg(needed["NEG_ALL_AS"])
        rep_ns      = read_rep(needed["REP_NS_AS"])
        rep_all     = read_rep(needed["REP_ALL_AS"])
        disp_ns     = read_disp(needed["DISP_NS_SEP"])
        disp_all    = read_disp(needed["DISP_ALL_SEP"])
        log("Loaded all sources.")

        # 1) Policy comparison (NS)
        ys=[y for y in YEARS if y in neg_ns_as and y in neg_ns_sep]
        ra=[neg_ns_as[y][2] for y in ys]; rs=[neg_ns_sep[y][2] for y in ys]
        fig=plt.figure(figsize=(10,5.5)); ax=fig.add_subplot(111)
        ax.plot(ys,[pct(x) for x in ra],marker="o",label="NS: dismiss counted negative")
        ax.plot(ys,[pct(x) for x in rs],marker="o",label="NS: dismiss separate")
        ax.set_xlabel("Decision year"); ax.set_ylabel("Negative rate (%)")
        ax.set_title("Negative rate: Policy comparison (Non-suspects)")
        ax.grid(True,alpha=0.3); ax.legend(loc="best")
        save(fig,"neg_rate_policy_NS_lines")

        # 2) QC effect (dismiss counted negative)
        ys2=[y for y in YEARS if y in neg_ns_as and y in neg_all_as]
        rns=[neg_ns_as[y][2] for y in ys2]; rall=[neg_all_as[y][2] for y in ys2]
        fig=plt.figure(figsize=(10,5.5)); ax=fig.add_subplot(111)
        ax.plot(ys2,[pct(x) for x in rns],marker="o",label="Non-suspects")
        ax.plot(ys2,[pct(x) for x in rall],marker="o",label="All rows")
        ax.set_xlabel("Decision year"); ax.set_ylabel("Negative rate (%)")
        ax.set_title("Negative rate: QC scope effect (Dismissals counted negative)")
        ax.grid(True,alpha=0.3); ax.legend(loc="best")
        save(fig,"neg_rate_QC_NS_vs_ALL_lines")

        # 3) Rep presence NS vs ALL
        ys3=[y for y in YEARS if y in rep_ns and y in rep_all]
        rpr_ns=[rep_ns[y][2] for y in ys3]; rpr_all=[rep_all[y][2] for y in ys3]
        fig=plt.figure(figsize=(10,5.5)); ax=fig.add_subplot(111)
        ax.plot(ys3,[pct(x) for x in rpr_ns],marker="o",label="Non-suspects")
        ax.plot(ys3,[pct(x) for x in rpr_all],marker="o",label="All rows")
        ax.set_xlabel("Decision year"); ax.set_ylabel("Representation present (%)")
        ax.set_title("Representation presence: NS vs ALL")
        ax.grid(True,alpha=0.3); ax.legend(loc="best")
        save(fig,"rep_presence_NS_vs_ALL_lines")

        # 4) Disposition mix (2024, dismiss separate)
        cats=["grant","deny","dismiss","mixed","remand","vacate","withdraw","unknown"]
        ty=2024
        ns_vals=[disp_ns.get(ty,{}).get(c,0) for c in cats]
        all_vals=[disp_all.get(ty,{}).get(c,0) for c in cats]
        fig=plt.figure(figsize=(10,5.5)); ax=fig.add_subplot(111)
        labels=["Non-suspects","All rows"]; x=[0,1]; bottom=[0,0]
        for i,cat in enumerate(cats):
            h=[ns_vals[i], all_vals[i]]
            ax.bar(x,h,bottom=bottom,label=cat); bottom=[bottom[j]+h[j] for j in (0,1)]
        ax.set_xticks(x,labels); ax.set_ylabel("Count")
        ax.set_title(f"Disposition mix: {ty} (Dismissals separate)")
        ax.legend(loc="upper right",ncol=2,fontsize=8); ax.grid(axis="y",alpha=0.3)
        save(fig,"disp_mix_2024_NS_vs_ALL_stacked")

        # 5) One-page PDF
        pdf_path=os.path.join(OUT,"figure_sheet.rev16.pdf")
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(pdf_path) as pdf:
            fig=plt.figure(figsize=(11,8.5))
            ax1=fig.add_subplot(221); ax1.plot(ys,[pct(x) for x in ra],marker="o",label="NS dismiss neg")
            ax1.plot(ys,[pct(x) for x in rs],marker="o",label="NS dismiss sep"); ax1.set_title("Neg rate: Policy (NS)")
            ax1.grid(True,alpha=0.3); ax1.legend(fontsize=8)

            ax2=fig.add_subplot(222); ax2.plot(ys2,[pct(x) for x in rns],marker="o",label="NS")
            ax2.plot(ys2,[pct(x) for x in rall],marker="o",label="ALL"); ax2.set_title("Neg rate: QC (dismiss neg)")
            ax2.grid(True,alpha=0.3); ax2.legend(fontsize=8)

            ax3=fig.add_subplot(223); ax3.plot(ys3,[pct(x) for x in rpr_ns],marker="o",label="NS")
            ax3.plot(ys3,[pct(x) for x in rpr_all],marker="o",label="ALL")
            ax3.set_title("Rep presence"); ax3.set_xlabel("Year"); ax3.set_ylabel("%")
            ax3.grid(True,alpha=0.3); ax3.legend(fontsize=8)

            ax4=fig.add_subplot(224); bottom=[0,0]; x=[0,1]
            for i,cat in enumerate(["grant","deny","dismiss","mixed","remand","vacate","withdraw","unknown"]):
                h=[ns_vals[i], all_vals[i]]; ax4.bar(x,h,bottom=bottom,label=cat)
                bottom=[bottom[j]+h[j] for j in (0,1)]
            ax4.set_xticks(x,["NS","ALL"]); ax4.set_title(f"Disposition mix {ty} (dismiss sep)")
            ax4.grid(axis="y",alpha=0.3); ax4.legend(fontsize=7,ncol=3,loc="upper right")

            fig.suptitle("BVA Study 2010-2025: Sensitivity Figures (rev16c)", y=0.98, fontsize=12)
            fig.tight_layout(rect=[0,0,1,0.96])
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        log(f"Saved {pdf_path}")
        log("DONE rev16c")
    except Exception:
        err=traceback.format_exc()
        log("FATAL:\n"+err)
        blog("FATAL (see log in OUT).")

if __name__ == "__main__":
    main()
