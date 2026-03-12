# bva_qc_recheck_txt.rev07.pyw
# Text-only recheck with broader, softer validity signals + explain logging.

import os, re, sqlite3, traceback, threading, queue, datetime
from tkinter import Tk, Frame, Label
from tkinter.scrolledtext import ScrolledText

ROOT = os.environ.get("BVA_ROOT", ".")
DB = os.path.join(ROOT, "parsed", "parsed.full.rev01.sqlite")
SUMMARY_TXT = os.path.join(ROOT, "index", "qc_recheck_summary.rev07.txt")
EXPLAINS_CSV = os.path.join(ROOT, "index", "qc_recheck_explains.rev07.csv")
SAMPLE_CSV   = os.path.join(ROOT, "index", "qc_recheck_sample_200.rev07.csv")
ENC = "utf-8"

# ---------------- Loaders ----------------
def load_txt(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(2_000_000)
    except Exception:
        try:
            with open(path, "r", encoding="cp1252", errors="ignore") as f:
                return f.read(2_000_000)
        except Exception:
            return ""

# ---------------- Disposition ----------------
RE_SECTION = re.compile(r"\n([A-Z][A-Z \t]{3,})\n")
RE_ORDER   = re.compile(r"\bORDER\b", re.M)
WORD = {
    "grant": re.compile(r"\bgrant(ed|s)?\b", re.I),
    "deny": re.compile(r"\bden(y|ied|ies)\b", re.I),
    "remand": re.compile(r"\bremand(ed|s)?\b", re.I),
    "dismiss": re.compile(r"\bdismiss(ed|al|es)?\b", re.I),
    "vacate": re.compile(r"\bvacate(d|s)?\b", re.I),
    "withdraw": re.compile(r"\bwithdraw(n|al|s|ing)?\b", re.I),
}
def slice_order(t):
    if not t: return ""
    m = RE_ORDER.search(t)
    if not m: return ""
    m2 = RE_SECTION.search(t, m.start()+5)
    return t[m.start(): (m2.start() if m2 else len(t))]
def dispositions(t):
    seg = slice_order(t) or (t or "")[:8000]
    found = set(k for k, rx in WORD.items() if rx.search(seg) or (k in ("dismiss","vacate") and rx.search(t or "")))
    if not found: return "unknown", ""
    return ("mixed" if len(found) > 1 else next(iter(found))), ",".join(sorted(found))

# ---------------- Signals ----------------
CORE_HDRS = [re.compile(p, re.I) for p in [
    r"\bORDER\b", r"\bFINDINGS OF FACT\b", r"\bCONCLUSIONS OF LAW\b",
    r"\bREASONS AND BASES\b", r"\bREMAND\b", r"\bISSUE(S)?\b", r"\bINTRODUCTION\b"
]]
HIGH_MARKERS = [re.compile(p, re.I) for p in [
    r"\bVeterans\s+Law\s+Judge\b", r"\bATTORNEY\s+FOR\s+THE\s+BOARD\b",
    r"\bBoard\s+of\s+Veterans[’']?\s+Appeals\b", r"\bCitation\s+Nr\b",
    r"\bDocket\s+No\.?\b", r"\bDecision\s+Date\b"
]]
VETSIG = [re.compile(p, re.I) for p in [
    r"\bVeteran\b", r"\bAppellant\b"
]]
ISSUE_PHRASE = re.compile(r"\b(Entitlement\s+to|service\s+connection\s+for|increased\s+rating\s+for)\b", re.I)
USC = re.compile(r"38\s*U\.?\s*S\.?\s*C\.?", re.I)
CFR = re.compile(r"38\s*C\.?\s*F\.?\s*R\.?", re.I)

def english_density(t):
    s = (t or "")
    chars = len(s)
    words = len(re.findall(r"[A-Za-z]{2,}", s))
    letters = len(re.findall(r"[A-Za-z]", s))
    total = len(s)
    alpha_ratio = (letters / total) if total else 0.0
    return chars, words, alpha_ratio

def valid_signals(t):
    t = (t or "").strip()
    # single high-confidence marker?
    if any(rx.search(t) for rx in HIGH_MARKERS):
        return True, "high_marker"

    core = any(rx.search(t) for rx in CORE_HDRS)
    cites = (len(USC.findall(t)) + len(CFR.findall(t))) >= 2
    issue = ISSUE_PHRASE.search(t) is not None
    vet = any(rx.search(t) for rx in VETSIG)

    chars, words, alpha = english_density(t)
    narrative = (chars >= 1200 and words >= 150 and alpha >= 0.65 and vet)

    # one-of these is enough now
    if core:   return True, "core_header"
    if cites:  return True, "legal_cites>=2"
    if issue:  return True, "issue_phrase"
    if narrative: return True, "narrative_veteran"

    return False, "none"

# ---------------- Output helpers ----------------
def write_summary(**kw):
    os.makedirs(os.path.dirname(SUMMARY_TXT), exist_ok=True)
    with open(SUMMARY_TXT, "w", encoding=ENC) as f:
        f.write("QC Recheck Summary rev07 (text-only, softer)\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat(sep=' ', timespec='seconds')}\n")
        for k,v in kw.items(): f.write(f"{k}: {v}\n")

def run(q):
    try:
        q.put("Starting text-only recheck (rev07)…")
        if not os.path.exists(DB):
            q.put(f"Missing DB: {DB}"); write_summary(error="DB not found"); return
        conn = sqlite3.connect(DB); cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='qc'")
        if not cur.fetchone():
            q.put("qc table missing."); write_summary(error="qc table missing"); conn.close(); return

        # remaining suspects
        cur.execute("SELECT case_id FROM qc WHERE is_struct_suspect=1")
        suspects = [r[0] for r in cur.fetchall()]
        total_sus = len(suspects)
        q.put(f"Remaining suspects: {total_sus}")
        if total_sus == 0:
            write_summary(message="No remaining suspects."); conn.close(); q.put("Done."); return

        # temp table
        cur.execute("DROP TABLE IF EXISTS temp_suspects")
        cur.execute("CREATE TEMP TABLE temp_suspects (cid TEXT PRIMARY KEY)")
        for i in range(0, total_sus, 5000):
            cur.executemany("INSERT OR IGNORE INTO temp_suspects(cid) VALUES(?)", [(c,) for c in suspects[i:i+5000]])
        conn.commit()

        # fetch paths (txt)
        cur.execute("""
            SELECT p.case_id, p.source_path
              FROM parsed p INNER JOIN temp_suspects s ON s.cid = p.case_id
        """)
        rows = cur.fetchall()
        pathmap = {cid: p for cid, p in rows}

        updates = []
        explains_rows = []
        cleared = 0
        # also write a 200-case sample of suspects with signals for manual spot-check
        sample_rows = []
        sample_cap = 200
        for n, cid in enumerate(suspects, 1):
            p = pathmap.get(cid, "")
            t = load_txt(p)
            is_valid, reason = valid_signals(t)
            primary, dset = dispositions(t)
            deny_equiv = 1 if (primary in ("deny","dismiss") or ("deny" in (dset.split(",") if dset else []))) else 0
            suspect = 0 if is_valid else 1
            if suspect == 0: cleared += 1
            updates.append((primary, dset, deny_equiv, "signals:"+reason, suspect, cid))
            explains_rows.append((cid, "yes" if suspect==0 else "no", reason, primary, dset))
            if len(sample_rows) < sample_cap:
                sample_rows.append((cid, p, reason, primary, dset, len(t)))

            if n % 1000 == 0 or n == total_sus:
                q.put(f"Rechecked {n}/{total_sus} | cleared {cleared}")

        # apply updates
        cur.executemany("""
            UPDATE qc
               SET primary_disposition=?,
                   disposition_set=?,
                   deny_equivalent=?,
                   struct_flags=?,
                   is_struct_suspect=?
             WHERE case_id=?""", updates)
        conn.commit()

        # counts after
        cur.execute("SELECT COUNT(*) FROM qc WHERE is_struct_suspect=1")
        remaining = cur.fetchone()[0]
        conn.close()

        # write explains/sample
        os.makedirs(os.path.dirname(EXPLAINS_CSV), exist_ok=True)
        with open(EXPLAINS_CSV, "w", encoding=ENC, newline="") as f:
            f.write("case_id,cleared,explain_signal,primary_disposition,disposition_set\n")
            for r in explains_rows: f.write(",".join(str(x) for x in r) + "\n")
        with open(SAMPLE_CSV, "w", encoding=ENC, newline="") as f:
            f.write("case_id,source_path,explain_signal,primary_disposition,disposition_set,text_len\n")
            for r in sample_rows: f.write(",".join(str(x) for x in r) + "\n")

        write_summary(suspects_input=total_sus, cleared=cleared, remaining=remaining)
        q.put(f"Done. Cleared {cleared}/{total_sus}. Remaining suspects: {remaining}")
    except Exception:
        err = traceback.format_exc()
        q.put("Fatal error:"); q.put(err)
        write_summary(error=err)

# ---------------- UI ----------------
def pump():
    try:
        while True:
            msg = Q.get_nowait(); LOG.insert("end", msg + "\n"); LOG.see("end")
    except queue.Empty:
        pass
    root.after(200, pump)

root = Tk()
root.title("BVA QC Recheck (Text-Only) • rev07")
root.geometry("900x480")
frm = Frame(root); frm.pack(fill="both", expand=True, padx=10, pady=10)
Label(frm, text=f"DB: {DB}").pack(anchor="w", pady=(0,6))
LOG = ScrolledText(frm, wrap="word", height=22); LOG.pack(fill="both", expand=True)
Q = queue.Queue()
threading.Thread(target=run, args=(Q,), daemon=True).start()
root.after(200, pump)
root.mainloop()
