# BVA_Downloader.pyw  (no external deps)
import sys, os, time, threading, csv, concurrent.futures, xml.etree.ElementTree as ET, ssl, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

APP_TITLE = "BVA Full-Text Downloader"
UA = "bva-bulk-downloader/1.3"

def resolve_out_root():
    env_root = os.environ.get("BVA_ROOT")
    if env_root:
        base = Path(env_root)
    else:
        base = Path(".")
    decisions = base / "decisions"
    indexes = base / "indexes"
    decisions.mkdir(parents=True, exist_ok=True)
    indexes.mkdir(parents=True, exist_ok=True)
    return decisions, indexes

CTX = ssl.create_default_context()

def fetch(url, timeout=30, stream=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            resp = urllib.request.urlopen(req, timeout=timeout, context=CTX)
            if stream:
                return resp  # caller must .read()
            data = resp.read()
            resp.close()
            return data
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}")

def fetch_xml(url):
    return fetch(url, timeout=30, stream=False)

def parse_sitemap(xml_bytes):
    root = ET.fromstring(xml_bytes)
    ns = root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else ""
    T = (lambda n: f"{{{ns}}}{n}") if ns else (lambda n: n)
    rows = []
    for u in root.findall(T("url")):
        loc = (u.findtext(T("loc")) or "").strip()
        lastmod = (u.findtext(T("lastmod")) or "").strip()
        if loc.lower().endswith(".txt"):
            rows.append({"loc": loc, "lastmod": lastmod})
    return rows

def safe_write_stream(url, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60, context=CTX) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 15)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(path)

def needs_download(path: Path):
    return (not path.exists()) or (path.stat().st_size == 0)

def url_to_relpath(year: int, url: str) -> Path:
    parts = url.split("/")
    try:
        idx = next(i for i, p in enumerate(parts) if p.startswith("vetapp"))
        tail = parts[idx+1:]
    except StopIteration:
        tail = [parts[-1]]
    if not tail:
        tail = [parts[-1]]
    return Path(str(year)) / Path(*tail)

def download_one(out_root: Path, entry: dict):
    year = entry["year"]; url = entry["loc"]; lastmod = entry.get("lastmod","")
    rel = url_to_relpath(year, url)
    out_path = out_root / rel
    try:
        if needs_download(out_path):
            safe_write_stream(url, out_path)
        size = out_path.stat().st_size if out_path.exists() else 0
        return {"year": year, "url": url, "lastmod": lastmod, "local_path": str(out_path), "status": "ok", "bytes": size}
    except Exception as e:
        return {"year": year, "url": url, "lastmod": lastmod, "local_path": str(out_path), "status": f"error:{type(e).__name__}", "bytes": 0}

def write_index(index_root: Path, year: int, rows: list[dict]) -> Path:
    p = index_root / f"vetapp{str(year)[-2:]}_index.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year","url","lastmod","local_path","status","bytes"])
        w.writeheader(); w.writerows(rows)
    return p

class App:
    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE); root.geometry("740x520"); root.resizable(False, False)
        self.decisions_root, self.index_root = resolve_out_root()

        frm = ttk.Frame(root, padding=12); frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=f"Decisions: {self.decisions_root}\nIndexes:   {self.index_root}").pack(anchor="w", pady=(0,8))

        c = ttk.Frame(frm); c.pack(fill="x", pady=(0,8))
        self.start_year_var = tk.IntVar(value=2010)
        self.end_year_var   = tk.IntVar(value=datetime.now().year)
        self.workers_var    = tk.IntVar(value=12)

        ttk.Label(c, text="Start year").grid(row=0, column=0, sticky="w")
        ttk.Entry(c, textvariable=self.start_year_var, width=8).grid(row=0, column=1, padx=(6,18))
        ttk.Label(c, text="End year").grid(row=0, column=2, sticky="w")
        ttk.Entry(c, textvariable=self.end_year_var, width=8).grid(row=0, column=3, padx=(6,18))
        ttk.Label(c, text="Workers").grid(row=0, column=4, sticky="w")
        ttk.Entry(c, textvariable=self.workers_var, width=6).grid(row=0, column=5, padx=(6,18))

        self.start_btn = ttk.Button(c, text="Start", command=self.start); self.start_btn.grid(row=0, column=6, padx=(6,6))
        self.stop_btn  = ttk.Button(c, text="Stop", command=self.stop, state="disabled"); self.stop_btn.grid(row=0, column=7, padx=(6,0))

        self.progress = ttk.Progressbar(frm, mode="indeterminate"); self.progress.pack(fill="x")
        self.log = tk.Text(frm, height=22, wrap="word"); self.log.pack(fill="both", expand=True, pady=(8,0))
        self.log.configure(state="disabled")

        self.stop_flag = threading.Event()
        self.worker_thread = None
        self.log_line("Ready.")

    def log_line(self, msg):
        self.log.configure(state="normal"); self.log.insert("end", msg + "\n"); self.log.see("end"); self.log.configure(state="disabled")
        self.root.update_idletasks()

    def start(self):
        if self.worker_thread and self.worker_thread.is_alive(): return
        self.stop_flag.clear()
        s = self.start_year_var.get(); e = self.end_year_var.get(); w = max(2, min(int(self.workers_var.get()), 32))
        self.start_btn.configure(state="disabled"); self.stop_btn.configure(state="normal"); self.progress.start(10)
        self.log_line(f"Starting {s} → {e} with {w} workers.")
        self.worker_thread = threading.Thread(target=self.run_job, args=(s, e, w), daemon=True); self.worker_thread.start()

    def stop(self):
        self.stop_flag.set(); self.log_line("Stop requested…")

    def run_job(self, s, e, w):
        try:
            for y in range(s, e + 1):
                if self.stop_flag.is_set(): self.log_line("Stopped by user."); break
                try:
                    count, ok, err, idx = self.process_year(y, w)
                    self.log_line(f"{y}: urls={count}, ok={ok}, err={err} -> {idx}")
                except Exception as ex:
                    self.log_line(f"{y}: sitemap error: {ex}")
        finally:
            self.progress.stop(); self.start_btn.configure(state="normal"); self.stop_btn.configure(state="disabled"); self.log_line("Done.")

    def process_year(self, year, workers):
        yy = str(year)[-2:]; sm_url = f"https://www.va.gov/vetapp{yy}/sitemap.xml"
        items = parse_sitemap(fetch_xml(sm_url))
        for it in items: it["year"] = year

        results = []
        chunk = 100
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for i in range(0, len(items), chunk):
                if self.stop_flag.is_set(): break
                batch = items[i:i+chunk]
                for res in ex.map(lambda e: download_one(self.decisions_root, e), batch):
                    results.append(res)
                self.log_line(f"{year}: downloaded {len(results)}/{len(items)}")
        idx = write_index(self.index_root, year, results)
        ok = sum(1 for r in results if r["status"] == "ok")
        err = sum(1 for r in results if r["status"].startswith("error"))
        return len(items), ok, err, idx

def main():
    root = tk.Tk()
    style = ttk.Style()
    try: style.theme_use("vista")
    except tk.TclError: pass
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
