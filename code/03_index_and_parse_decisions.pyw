# Builds manifest, pilot parse, and QA sample. Double-click to run. No console.

import os, sys, csv, hashlib, json, re, traceback, random, datetime, threading, queue, pathlib, subprocess, time

# ---------- Config ----------
ROOT_DIR = os.environ.get("BVA_ROOT", ".")
OUTPUT_ROOT = os.environ.get("BVA_ROOT", ".")
INCLUDE_EXT = {".pdf", ".htm", ".html", ".docx", ".doc", ".txt"}
EXCLUDE_DIR_SUBSTR = {"\\~$", "\\temp\\", "\\_temp\\", "\\thumbs", "\\$recycle.bin"}
PILOT_RATE = 0.01
MAX_PILOT = 11915
QA_SAMPLE_SIZE = 300
COMPUTE_SHA256 = True
HASH_CHUNK = 1024 * 1024
LOG_MAX_LINES = 5000
# ----------------------------

# Globals set by ensure_deps
HAS_PYPDF = False
HAS_PYPDF2 = False

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(HASH_CHUNK), b""):
            h.update(b)
    return h.hexdigest()

def mime_guess_from_ext(ext):
    e = ext.lower()
    return {
        ".pdf": "application/pdf",
        ".htm": "text/html", ".html": "text/html",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".txt": "text/plain",
    }.get(e, "application/octet-stream")

def infer_year_from_path(p):
    for m in re.finditer(r"(20[1][0-9]|202[0-5])", p):
        try:
            yr = int(m.group(1))
            if 2010 <= yr <= 2025:
                return yr
        except:
            pass
    return ""

def should_skip_dir(path):
    low = path.lower()
    return any(s in low for s in EXCLUDE_DIR_SUBSTR)

def discover_files(qmsg):
    qmsg.put("Scanning files...")
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        if should_skip_dir(dirpath):
            continue
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext and ext in INCLUDE_EXT:
                yield os.path.join(dirpath, name)

def safe_stat(path):
    try:
        st = os.stat(path)
        return st.st_size, st.st_ctime, st.st_mtime
    except Exception:
        return "", "", ""

def to_iso(ts):
    if not ts: return ""
    try:
        return datetime.datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")
    except Exception:
        return ""

def stable_case_id(path, size):
    base = f"{path}|{size}"
    return hashlib.md5(base.encode("utf-8", "ignore")).hexdigest()

def ensure_dirs():
    for sub in ["index", "parsed", "qa", "logs"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)

def write_codebook():
    codebook = {
        "manifest.rev01.csv": {
            "file_path": "Full absolute path",
            "file_name": "Base file name",
            "file_ext": "Lowercased extension",
            "file_size_bytes": "Integer size",
            "sha256": "SHA256 if enabled",
            "created_dt": "FS created datetime",
            "modified_dt": "FS modified datetime",
            "inferred_year": "Year inferred from path/name",
            "pages_est": "Optional page count",
            "mime_guess": "From extension",
            "case_id": "Stable hash of path+size",
        },
        "parsed.pilot.rev01.csv": {
            "case_id": "Stable ID",
            "source_path": "Original file",
            "type": "pdf|html|docx|doc|txt",
            "file_pages": "Pages if found",
            "header_snippet": "Header text",
            "docket_no": "Regex capture",
            "decision_date": "Regex capture",
            "vlj_name": "Regex capture",
            "rep_line": "Representative line",
            "outcome_primary": "grant|deny|remand|mixed|unknown",
        },
        "pilot_sample.rev01.csv": {
            "case_id": "From manifest",
            "source_path": "From manifest",
            "inferred_year": "From manifest",
            "type": "From pilot parse",
            "parsed_docket_no": "From pilot",
            "parsed_decision_date": "From pilot",
            "parsed_outcome": "From pilot",
            "raw_header_snippet": "From pilot",
        }
    }
    path = os.path.join(OUTPUT_ROOT, "index", "codebook.rev01.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2)

# -------- Robust dependency bootstrap --------
def _pip(args, qmsg):
    env = os.environ.copy()
    env["PIP_USER"] = "1"
    try:
        subprocess.check_call([sys.executable, "-m", "pip"] + args, env=env)
        return True
    except Exception as e:
        qmsg.put(f"pip call failed: {' '.join(args)} | {e}")
        return False

def ensure_deps(qmsg):
    global HAS_PYPDF, HAS_PYPDF2
    # 1) Ensure pip exists for this interpreter
    try:
        import pip  # noqa
    except Exception:
        qmsg.put("pip missing. Bootstrapping with ensurepip...")
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
            qmsg.put("ensurepip complete.")
        except Exception as e:
            qmsg.put(f"ensurepip failed: {e}")

    # Helper to try import then install
    def try_import_or_install(pkg, mod, alt_names=None):
        try:
            __import__(mod)
            return True
        except Exception:
            qmsg.put(f"Missing {pkg}. Installing...")
            ok = _pip(["install", "--upgrade", "--user", pkg], qmsg)
            if ok:
                try:
                    __import__(mod)
                    qmsg.put(f"Installed {pkg}")
                    return True
                except Exception as e:
                    qmsg.put(f"Import after install failed for {pkg}: {e}")
        # try alternates if given
        if alt_names:
            for pkg2, mod2 in alt_names:
                try:
                    __import__(mod2)
                    qmsg.put(f"Using existing {pkg2}")
                    return True
                except Exception:
                    qmsg.put(f"Installing alternate {pkg2}...")
                    ok = _pip(["install", "--upgrade", "--user", pkg2], qmsg)
                    if ok:
                        try:
                            __import__(mod2)
                            qmsg.put(f"Installed {pkg2}")
                            return True
                        except Exception as e:
                            qmsg.put(f"Import after install failed for {pkg2}: {e}")
        return False

    # PDF stack: prefer pypdf, fallback PyPDF2
    HAS_PYPDF = try_import_or_install("pypdf", "pypdf", alt_names=[("PyPDF2", "PyPDF2")])
    if not HAS_PYPDF:
        try:
            import PyPDF2  # noqa
            HAS_PYPDF2 = True
        except Exception:
            HAS_PYPDF2 = False
    else:
        HAS_PYPDF2 = False

    # DOCX, BS4
    docx_ok = try_import_or_install("python-docx", "docx")
    bs4_ok = try_import_or_install("beautifulsoup4", "bs4")

    # If neither PDF lib is available, we still proceed. Pilot will skip PDFs.
    if not (HAS_PYPDF or HAS_PYPDF2):
        qmsg.put("Warning: No PDF library available. PDFs will be logged and skipped in pilot parse.")

    if not docx_ok:
        qmsg.put("Warning: python-docx unavailable. DOCX will be skipped in pilot parse.")
    if not bs4_ok:
        qmsg.put("Warning: beautifulsoup4 unavailable. HTML will be skipped in pilot parse.")

# -------- Parsing helpers --------
RE_DOCKET = re.compile(r"(Docket\s*No\.?|Docket\s*#)\s*[:\-]?\s*([A-Za-z0-9\- ]{3,})", re.I)
RE_DATE = re.compile(r"(Decision\s*Date|Date\s*of\s*Decision)\s*[:\-]?\s*([A-Z][a-z]{2,}\s+\d{1,2},\s+20\d{2}|\d{4}\-\d{2}\-\d{2}|\d{2}/\d{2}/\d{4})", re.I)
RE_VLJ = re.compile(r"(Veterans\s+Law\s+Judge|VLJ)\s*[:\-]?\s*([A-Z][A-Za-z\.\- ]{2,})", re.I)
RE_REP = re.compile(r"(Representative|Appellant\s+represented\s+by)\s*[:\-]?\s*(.+)", re.I)
RE_OUTCOME_GRANT = re.compile(r"\b(granted|grant)\b", re.I)
RE_OUTCOME_DENY = re.compile(r"\b(denied|deny)\b", re.I)
RE_OUTCOME_REMAND = re.compile(r"\b(remanded|remand)\b", re.I)

def get_outcome(text):
    g = bool(RE_OUTCOME_GRANT.search(text))
    d = bool(RE_OUTCOME_DENY.search(text))
    r = bool(RE_OUTCOME_REMAND.search(text))
    s = g + d + r
    if s == 0: return "unknown"
    if s > 1: return "mixed"
    if g: return "grant"
    if d: return "deny"
    if r: return "remand"
    return "unknown"

def extract_text_pdf(path):
    global HAS_PYPDF, HAS_PYPDF2
    try:
        if HAS_PYPDF:
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = len(reader.pages)
            grab = min(3, pages)
            text = "\n".join([(reader.pages[i].extract_text() or "") for i in range(grab)])
            return pages, text
        elif HAS_PYPDF2:
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            pages = len(reader.pages)
            grab = min(3, pages)
            text = "\n".join([(reader.pages[i].extract_text() or "") for i in range(grab)])
            return pages, text
        else:
            return "", ""
    except Exception:
        return "", ""

def extract_text_html(path):
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return "", ""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        soup = BeautifulSoup(raw, "html.parser")
        heads = []
        for tag in ["h1","h2","h3","header","title"]:
            for el in soup.find_all(tag):
                heads.append(el.get_text(" ", strip=True))
        body_txt = soup.get_text(" ", strip=True)
        text = "\n".join(heads[:5]) + "\n" + body_txt[:5000]
        return "", text
    except Exception:
        return "", ""

def extract_text_docx(path):
    try:
        from docx import Document
    except Exception:
        return "", ""
    try:
        d = Document(path)
        paras = [p.text for p in d.paragraphs[:80]]
        text = "\n".join(paras)
        return "", text
    except Exception:
        return "", ""

def extract_text_doc(path):
    try:
        with open(path, "rb") as f:
            b = f.read(200000)
        text = b.decode("cp1252", errors="ignore")
        return "", text
    except Exception:
        return "", ""

def extract_text_txt(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return "", f.read(200000)
    except Exception:
        try:
            with open(path, "r", encoding="cp1252", errors="ignore") as f:
                return "", f.read(200000)
        except Exception:
            return "", ""

def header_fields_from_text(text):
    snippet = (text or "").strip()[:3000]
    docket = ""
    decision_date = ""
    vlj = ""
    rep_line = ""
    try:
        m = RE_DOCKET.search(snippet)
        if m: docket = m.group(2).strip()
        m = RE_DATE.search(snippet)
        if m: decision_date = m.group(2).strip()
        m = RE_VLJ.search(snippet)
        if m: vlj = m.group(2).strip()
        m = RE_REP.search(snippet)
        if m: rep_line = m.group(2).strip()[:500]
    except Exception:
        pass
    outcome = get_outcome(snippet)
    return snippet, docket, decision_date, vlj, rep_line, outcome

def pilot_take(n):
    k = max(1, int(n * PILOT_RATE))
    return min(k, MAX_PILOT)

def stratified_sample(paths_by_ext, total_k):
    exts = list(paths_by_ext.keys())
    if not exts: return []
    buckets = {e: list(paths_by_ext[e]) for e in exts}
    for e in buckets: random.shuffle(buckets[e])
    picked, i = [], 0
    while len(picked) < total_k and any(buckets.values()):
        e = exts[i % len(exts)]
        if buckets[e]: picked.append(buckets[e].pop())
        i += 1
    return picked

def work(qmsg):
    start = time.time()
    ensure_dirs()
    log_path = os.path.join(OUTPUT_ROOT, "logs", "ingest.rev01.log")
    manifest_path = os.path.join(OUTPUT_ROOT, "index", "manifest.rev01.csv")
    parsed_path = os.path.join(OUTPUT_ROOT, "index", "parsed.pilot.rev01.csv")
    qa_path = os.path.join(OUTPUT_ROOT, "qa", "pilot_sample.rev01.csv")

    log_lines = []
    def log(line):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{ts} | {line}"
        log_lines.append(msg)
        if len(log_lines) > LOG_MAX_LINES:
            del log_lines[0:len(log_lines)-LOG_MAX_LINES]
        qmsg.put(line)

    # Deps
    ensure_deps(qmsg)

    # Discover
    files = list(discover_files(qmsg))
    total = len(files)
    if total == 0:
        qmsg.put("No files found. Check INCLUDE_EXT or ROOT_DIR.")
    log(f"Discovered files: {total}")

    # Manifest
    with open(manifest_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["file_path","file_name","file_ext","file_size_bytes","sha256","created_dt","modified_dt","inferred_year","pages_est","mime_guess","case_id"])
        for idx, path in enumerate(files, 1):
            try:
                size, ctime, mtime = safe_stat(path)
                ext = os.path.splitext(path)[1].lower()
                sha = sha256_file(path) if (COMPUTE_SHA256 and size and size < 2*1024*1024*1024) else ""
                year = infer_year_from_path(path)
                mime = mime_guess_from_ext(ext)
                case = stable_case_id(path, size or 0)
                w.writerow([path, os.path.basename(path), ext, size, sha, to_iso(ctime), to_iso(mtime), year, "", mime, case])
                if idx % 5000 == 0:
                    log(f"Manifest rows: {idx}/{total}")
            except Exception as e:
                log(f"Manifest error: {path} | {e}")

    # Pilot selection
    paths_by_ext = {}
    for p in files:
        e = os.path.splitext(p)[1].lower()
        paths_by_ext.setdefault(e, []).append(p)
    k = pilot_take(total)
    pilot_paths = stratified_sample(paths_by_ext, k)
    log(f"Pilot size: {len(pilot_paths)}")

    # Pilot parse
    skipped_counts = {"pdf":0, "docx":0, "html":0}
    with open(parsed_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["case_id","source_path","type","file_pages","header_snippet","docket_no","decision_date","vlj_name","rep_line","outcome_primary"])
        for i, p in enumerate(pilot_paths, 1):
            try:
                ext = os.path.splitext(p)[1].lower()
                size, _, _ = safe_stat(p)
                case = stable_case_id(p, size or 0)
                pages, text, ftype = "", "", ext[1:]

                if ext == ".pdf":
                    if HAS_PYPDF or HAS_PYPDF2:
                        pages, text = extract_text_pdf(p)
                    else:
                        skipped_counts["pdf"] += 1
                        text = ""
                elif ext in {".htm", ".html"}:
                    txt = extract_text_html(p)
                    if txt == ("",""):
                        skipped_counts["html"] += 1
                    pages, text = txt
                elif ext == ".docx":
                    txt = extract_text_docx(p)
                    if txt == ("",""):
                        skipped_counts["docx"] += 1
                    pages, text = txt
                elif ext == ".doc":
                    pages, text = extract_text_doc(p)
                elif ext == ".txt":
                    pages, text = extract_text_txt(p)

                snippet, docket, ddate, vlj, rep, outcome = header_fields_from_text(text or "")
                w.writerow([case, p, ftype, pages, snippet.replace("\r"," ").replace("\n"," ")[:3000], docket, ddate, vlj, rep, outcome])
                if i % 1000 == 0:
                    log(f"Pilot parsed: {i}/{len(pilot_paths)}")
            except Exception as e:
                log(f"Pilot parse error: {p} | {e}")

    if skipped_counts["pdf"] or skipped_counts["docx"] or skipped_counts["html"]:
        log(f"Pilot skips: PDF:{skipped_counts['pdf']} DOCX:{skipped_counts['docx']} HTML:{skipped_counts['html']}")

    # QA sample
    parsed_rows = {}
    try:
        with open(parsed_path, "r", encoding="utf-8") as f:
            rr = csv.DictReader(f)
            for row in rr:
                parsed_rows[row["case_id"]] = row
    except Exception as e:
        log(f"Read parsed pilot error: {e}")

    all_cases = list(parsed_rows.keys())
    if len(all_cases) <= QA_SAMPLE_SIZE:
        sample_ids = all_cases
    else:
        random.shuffle(all_cases)
        sample_ids = all_cases[:QA_SAMPLE_SIZE]

    manifest_year = {}
    manifest_ext = {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as mf:
            mr = csv.DictReader(mf)
            for mrow in mr:
                manifest_year[mrow["case_id"]] = mrow["inferred_year"]
                manifest_ext[mrow["case_id"]] = mrow["file_ext"].lstrip(".")
    except Exception as e:
        log(f"Read manifest for QA error: {e}")

    with open(qa_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["case_id","source_path","inferred_year","type","parsed_docket_no","parsed_decision_date","parsed_outcome","raw_header_snippet"])
        for cid in sample_ids:
            row = parsed_rows[cid]
            w.writerow([
                cid,
                row.get("source_path",""),
                manifest_year.get(cid, ""),
                row.get("type","") or manifest_ext.get(cid,""),
                row.get("docket_no",""),
                row.get("decision_date",""),
                row.get("outcome_primary",""),
                row.get("header_snippet",""),
            ])

    write_codebook()

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    dur = time.time() - start
    qmsg.put(f"Done in {int(dur)}s")

# ---------- Minimal UI ----------
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import queue as _q

def run_in_thread(qmsg):
    try:
        work(qmsg)
    except Exception:
        qmsg.put("Fatal error:")
        qmsg.put(traceback.format_exc())
        qmsg.put("Done.")

def pump_queue():
    try:
        while True:
            msg = Q.get_nowait()
            if msg:
                log_box.insert(tk.END, msg + "\n")
                log_box.see(tk.END)
    except _q.Empty:
        pass
    root.after(200, pump_queue)

root = tk.Tk()
root.title("BVA Indexer • rev02")
root.geometry("900x500")

frame = tk.Frame(root)
frame.pack(fill="both", expand=True, padx=10, pady=10)

lbl = tk.Label(frame, text=f"Root: {ROOT_DIR}   Output: {OUTPUT_ROOT}")
lbl.pack(anchor="w", pady=(0,6))

log_box = ScrolledText(frame, wrap="word", height=25)
log_box.pack(fill="both", expand=True)

Q = _q.Queue()
t = threading.Thread(target=run_in_thread, args=(Q,), daemon=True)
t.start()

root.after(200, pump_queue)
root.mainloop()
