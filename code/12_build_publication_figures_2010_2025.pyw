# bva_make_charts_2010_2025.rev02.pyw
# Zero-dependency chart maker: writes SVGs (no matplotlib needed).
# Reads the rev03 CSVs and outputs to {BVA_ROOT}/index/figs_2010_2025/*.svg

import os, csv, math, datetime, tkinter as tk
from tkinter.scrolledtext import ScrolledText

ROOT = os.environ.get("BVA_ROOT", ".")
IDX  = os.path.join(ROOT, "index")
OUT  = os.path.join(IDX, "figs_2010_2025")
LOGF = os.path.join(OUT, "chart_log.rev02.txt")
os.makedirs(OUT, exist_ok=True)

F_REP_ANY = os.path.join(IDX, "rep_presence_by_decision_year.clean.2010_2025.rev03.csv")
F_REP_TYP = os.path.join(IDX, "rep_type_share_by_year.clean.2010_2025.rev03.csv")
F_DISP    = os.path.join(IDX, "disposition_by_decision_year.clean.2010_2025.rev03.csv")
F_NEG     = os.path.join(IDX, "negative_rate_by_decision_year.clean.2010_2025.rev03.csv")

# ---------- tiny SVG helpers ----------
def svgend(w,h,content):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n{content}\n</svg>'

def rect(x,y,w,h,fill="#cccccc",stroke="#000",sw=1,opacity=1.0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'

def line(x1,y1,x2,y2,stroke="#000",sw=1,opacity=1.0):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'

def text(x,y,s,anchor="start",size=12,weight="normal"):
    return f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>'

def circle(x,y,r=3,fill="#000"):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>'

def polyline(points, stroke="#000", sw=2, fill="none", opacity=1.0):
    pts=" ".join(f"{x},{y}" for x,y in points)
    return f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'

def polygon(points, fill="#888", opacity=0.6, stroke="none"):
    pts=" ".join(f"{x},{y}" for x,y in points)
    return f'<polygon points="{pts}" fill="{fill}" opacity="{opacity}" stroke="{stroke}"/>'

# ---------- data readers ----------
def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def to_int(x):
    try: return int(x)
    except: return None

def to_float(x):
    try: return float(x)
    except: return None

# ---------- chart 1: Any Representative Rate (line) ----------
def chart_rep_presence():
    rows = read_csv(F_REP_ANY)
    data=[]
    for r in rows:
        y = to_int(r["decision_year"])
        if y is None: continue
        data.append((y, to_float(r["rate_any_rep"])))
    data.sort()
    if not data: return None
    w,h=900,480; padL, padR, padT, padB = 70, 20, 40, 60
    x0,y0 = padL, h-padB; x1,y1 = w-padR, padT
    content=[]
    content.append(rect(0,0,w,h,fill="#ffffff",stroke="#ffffff"))
    # axes
    content.append(line(x0,y0,x1,y0)) # x axis
    content.append(line(x0,y0,x0,y1)) # y axis
    # ticks
    years=[y for y,_ in data]
    xs = list(range(min(years), max(years)+1))
    def sx(year):
        return x0 + (year - xs[0])/(xs[-1]-xs[0]) * (x1-x0) if xs[-1]!=xs[0] else x0
    def sy(val):
        # val in [0,1]
        v = 0 if val is None else max(0,min(1,val))
        return y0 - v*(y0-y1)
    for yr in xs:
        X=sx(yr)
        content.append(line(X,y0,X,y0+5))
        if yr%2==0:  # label every 2nd year to reduce clutter
            content.append(text(X,y0+20,str(yr),anchor="middle",size=11))
    for v in [0.0,0.25,0.5,0.75,1.0]:
        Y=sy(v)
        content.append(line(x0-5,Y,x1,Y,stroke="#ddd"))
        content.append(text(x0-10,Y+4,f"{v:.2f}",anchor="end",size=10))
    content.append(text((x0+x1)//2, 25, "Any Representative Rate by Decision Year (2010-2025)", anchor="middle", size=16, weight="bold"))
    content.append(text((x0+x1)//2, h-10, "Decision Year", anchor="middle", size=12))
    content.append(text(18,(y0+y1)//2,"Rate",anchor="middle",size=12))
    # line
    pts=[(sx(y), sy(v)) for y,v in data if v is not None]
    content.append(polyline(pts, stroke="#1f77b4", sw=2))
    for X,Y in pts:
        content.append(circle(X,Y,3,fill="#1f77b4"))
    out = os.path.join(OUT,"rep_presence_rate.svg")
    with open(out,"w",encoding="utf-8") as f: f.write(svgend(w,h,"\n".join(content)))
    return out

# ---------- chart 2: Negative Rate (deny + dismiss) ----------
def chart_negative_rate():
    rows = read_csv(F_NEG)
    data=[]
    for r in rows:
        y = to_int(r["decision_year"])
        if y is None: continue
        data.append((y, to_float(r["negative_rate"])))
    data.sort()
    if not data: return None
    w,h=900,480; padL, padR, padT, padB = 70, 20, 40, 60
    x0,y0 = padL, h-padB; x1,y1 = w-padR, padT
    content=[rect(0,0,w,h,fill="#ffffff",stroke="#ffffff")]
    content += [line(x0,y0,x1,y0), line(x0,y0,x0,y1)]
    years=[y for y,_ in data]
    xs=list(range(min(years), max(years)+1))
    def sx(year):
        return x0 + (year - xs[0])/(xs[-1]-xs[0]) * (x1-x0) if xs[-1]!=xs[0] else x0
    def sy(val):
        v=0 if val is None else max(0,min(1,val))
        return y0 - v*(y0-y1)
    for yr in xs:
        X=sx(yr); content += [line(X,y0,X,y0+5)]
        if yr%2==0: content.append(text(X,y0+20,str(yr),anchor="middle",size=11))
    for v in [0.0,0.25,0.5,0.75,1.0]:
        Y=sy(v); content += [line(x0-5,Y,x1,Y,stroke="#ddd"), text(x0-10,Y+4,f"{v:.2f}",anchor="end",size=10)]
    content.append(text((x0+x1)//2, 25, "Negative Outcome Rate (Deny + Dismiss) by Year (2010-2025)", anchor="middle", size=16, weight="bold"))
    content.append(text((x0+x1)//2, h-10, "Decision Year", anchor="middle", size=12))
    content.append(text(18,(y0+y1)//2,"Rate",anchor="middle",size=12))
    pts=[(sx(y), sy(v)) for y,v in data if v is not None]
    content.append(polyline(pts, stroke="#d62728", sw=2))
    for X,Y in pts: content.append(circle(X,Y,3,fill="#d62728"))
    out = os.path.join(OUT,"negative_rate_deny_plus_dismiss.svg")
    with open(out,"w",encoding="utf-8") as f: f.write(svgend(w,h,"\n".join(content)))
    return out

# ---------- chart 3: Rep Type Shares (stacked area) ----------
def chart_rep_type_shares():
    rows = read_csv(F_REP_TYP)
    series = {"attorney":{}, "agent":{}, "VSO":{}, "none":{}, "unknown":{}}
    years=set()
    for r in rows:
        y=to_int(r["decision_year"]); 
        if y is None: continue
        years.add(y)
        t=r["rep_type"]; s=to_float(r["share"])
        if t in series: series[t][y]=s or 0.0
    xs=sorted(years)
    if not xs: return None
    w,h=900,480; padL, padR, padT, padB = 70, 20, 40, 60
    x0,y0 = padL, h-padB; x1,y1 = w-padR, padT
    def sx(year): 
        return x0 + (year - xs[0])/(xs[-1]-xs[0]) * (x1-x0) if xs[-1]!=xs[0] else x0
    def sy(val): 
        v=max(0,min(1,val))
        return y0 - v*(y0-y1)
    colors = {
        "attorney":"#1f77b4","agent":"#ff7f0e","VSO":"#2ca02c","none":"#9467bd","unknown":"#8c564b"
    }
    order = ["attorney","agent","VSO","none","unknown"]
    # build stacked polygons from bottom to top
    cum=[0.0]*len(xs)
    content=[rect(0,0,w,h,fill="#ffffff",stroke="#ffffff")]
    # grid and axes
    content += [line(x0,y0,x1,y0), line(x0,y0,x0,y1)]
    for yr in xs:
        X=sx(yr); content += [line(X,y0,X,y0+5)]
        if yr%2==0: content.append(text(X,y0+20,str(yr),anchor="middle",size=11))
    for v in [0.0,0.25,0.5,0.75,1.0]:
        Y=sy(v); content += [line(x0-5,Y,x1,Y,stroke="#ddd"), text(x0-10,Y+4,f"{v:.2f}",anchor="end",size=10)]
    content.append(text((x0+x1)//2, 25, "Representative Type Shares by Year (2010-2025)", anchor="middle", size=16, weight="bold"))
    content.append(text((x0+x1)//2, h-10, "Decision Year", anchor="middle", size=12))
    content.append(text(18,(y0+y1)//2,"Share",anchor="middle",size=12))
    # areas
    for key in order:
        vals=[series[key].get(y,0.0) for y in xs]
        top=[c+v for c,v in zip(cum, vals)]
        # polygon path: left->right along top, then right->left along bottom
        pts=[]
        for i,y in enumerate(xs): pts.append((sx(y), sy(top[i])))
        for i,y in enumerate(reversed(xs)): pts.append((sx(list(reversed(xs))[i]), sy(cum[len(xs)-1-i])))
        content.append(polygon(pts, fill=colors[key], opacity=0.7, stroke="none"))
        cum=top
    # legend
    lx,ly = x1-170, y1+10; step=18
    for i,key in enumerate(order):
        content.append(rect(lx, ly+i*step-10, 12,12, fill=colors[key], stroke="#333", sw=0.5))
        content.append(text(lx+18, ly+i*step, key, size=12))
    out = os.path.join(OUT,"rep_type_shares_stacked.svg")
    with open(out,"w",encoding="utf-8") as f: f.write(svgend(w,h,"\n".join(content)))
    return out

# ---------- chart 4: Dispositions by Year (grouped bars) ----------
def chart_dispositions():
    rows = read_csv(F_DISP)
    wanted = ["grant","deny","dismiss","remand","vacate","withdraw","mixed","unknown"]
    years=set(); counts={w:{} for w in wanted}
    for r in rows:
        y=to_int(r["decision_year"]); 
        if y is None: continue
        years.add(y)
        d=r["primary_disposition"]; n=int(r["n"])
        if d in counts: counts[d][y]=counts[d].get(y,0)+n
    xs=sorted(years)
    if not xs: return None
    w,h=980,520; padL, padR, padT, padB = 70, 20, 40, 80
    x0,y0 = padL, h-padB; x1,y1 = w-padR, padT
    ymax = max(sum(counts[d].get(y,0) for d in wanted) for y in xs) or 1
    # compute bar positions
    groups=len(xs); cats=len(wanted)
    gspan = (x1-x0)/groups
    bw = gspan/(cats+0.5)

    colors = {
        "grant":"#2ca02c","deny":"#d62728","dismiss":"#8c564b","remand":"#1f77b4",
        "vacate":"#9467bd","withdraw":"#17becf","mixed":"#ff7f0e","unknown":"#7f7f7f"
    }
    def sx(i, j): # group i, cat j
        return x0 + i*gspan + j*bw + 0.25*bw
    def sy(v):
        # simple linear scale to max; add 10% headroom
        v = v / (ymax*1.1)
        return y0 - v*(y0-y1)

    content=[rect(0,0,w,h,fill="#ffffff",stroke="#ffffff")]
    content += [line(x0,y0,x1,y0), line(x0,y0,x0,y1)]
    # y grid
    for frac in [0,0.25,0.5,0.75,1.0]:
        Y = y0 - frac*(y0-y1)
        content += [line(x0-5,Y,x1,Y,stroke="#ddd")]
        content += [text(x0-10, Y+4, f"{int(frac*ymax*1.1):,}", anchor="end", size=10)]
    # bars + x labels
    for i,y in enumerate(xs):
        content.append(text(x0 + i*gspan + gspan/2, y0+22, str(y), anchor="middle", size=10))
        for j,cat in enumerate(wanted):
            v = counts[cat].get(y,0)
            X = sx(i,j); Y = sy(v)
            content.append(rect(X, Y, bw*0.9, y0-Y, fill=colors[cat], stroke="#333", sw=0.3))
    # title/labels
    content.append(text((x0+x1)//2, 25, "Dispositions by Year (counts, 2010-2025)", anchor="middle", size=16, weight="bold"))
    content.append(text((x0+x1)//2, h-10, "Decision Year", anchor="middle", size=12))
    content.append(text(18,(y0+y1)//2,"Count",anchor="middle",size=12))
    # legend
    lx,ly = x1-200, y1+10; step=18
    for i,cat in enumerate(wanted):
        content.append(rect(lx, ly+i*step-10, 12,12, fill=colors[cat], stroke="#333", sw=0.5))
        content.append(text(lx+18, ly+i*step, cat, size=12))
    out = os.path.join(OUT,"dispositions_by_year_bars.svg")
    with open(out,"w",encoding="utf-8") as f: f.write(svgend(w,h,"\n".join(content)))
    return out

# ---------- tiny UI + logging ----------
def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    LOG.insert(tk.END, f"{ts} | {msg}\n"); LOG.see(tk.END)
    try:
        with open(LOGF,"a",encoding="utf-8") as f: f.write(f"{ts} | {msg}\n")
    except: pass

def run():
    try:
        made=[]
        p = chart_rep_presence()
        log("rep_presence_rate: " + ("OK" if p else "NO DATA")); 
        if p: log("→ " + p); made.append(p)
        p = chart_negative_rate()
        log("negative_rate_deny_plus_dismiss: " + ("OK" if p else "NO DATA"))
        if p: log("→ " + p); made.append(p)
        p = chart_rep_type_shares()
        log("rep_type_shares_stacked: " + ("OK" if p else "NO DATA"))
        if p: log("→ " + p); made.append(p)
        p = chart_dispositions()
        log("dispositions_by_year_bars: " + ("OK" if p else "NO DATA"))
        if p: log("→ " + p); made.append(p)
        if made:
            log("All charts complete.")
        else:
            log("No charts produced. Check that rev03 CSVs exist in {ROOT}/index.")
    except Exception as e:
        log("Error: " + str(e))

# UI
root = tk.Tk()
root.title("BVA Charts 2010-2025 • rev02 (SVG, no deps)")
root.geometry("800x360")
frm = tk.Frame(root); frm.pack(fill="both", expand=True, padx=10, pady=10)
tk.Label(frm, text=f"CSV folder: {IDX}").pack(anchor="w")
tk.Label(frm, text=f"Output folder: {OUT}").pack(anchor="w", pady=(0,6))
LOG = ScrolledText(frm, wrap="word", height=16); LOG.pack(fill="both", expand=True)

root.after(200, lambda: run())
root.mainloop()
