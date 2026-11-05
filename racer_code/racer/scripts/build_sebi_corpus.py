"""Build the full SEBI investment-adviser clause corpus from the two official PDFs (needs poppler's pdftotext).

  python scripts/build_sebi_corpus.py REGS.pdf CIRCULAR.pdf --out data/regulatory/sebi_full.jsonl

REGS.pdf     SEBI (Investment Advisers) Regulations, 2013, consolidated as amended up to 25 Nov 2025 (sebi.gov.in)
CIRCULAR.pdf Master Circular for Investment Advisers, 6 Feb 2026 (sebi.gov.in)

Page headers, page numbers, footnotes (amendment histories, circular references) and inline footnote markers are
removed. Regulations are chunked at their first-level sub-clauses, e.g. 17(a) or 19(1), each prefixed with the
regulation's lead-in sentence; the master circular is chunked at its paragraphs, never across a section.
Every record carries `reg` and `sub` (regulation number and sub-clause) so evaluations can locate provisions.
"""
import argparse, json, re, subprocess
from pathlib import Path

ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}


def pdf_pages(path):
    txt = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True).stdout
    return txt.split("\f")


def strip_page(page):
    """Drop the footnote block (starts at the left margin with a bare number, no period) and page furniture."""
    out = []
    for line in page.splitlines():
        if re.match(r"^\d{1,3}(\s+\S|\s*$)", line) and not re.match(r"^\d{1,3}[A-Z]?\.", line):
            break
        if re.match(r"^\s*(Page \d+ of \d+|\d{1,3})\s*$", line):
            continue
        out.append(line)
    return out


def clean(text):
    text = re.sub(r"\d{1,3}\[", "", text).replace("[***]", "").replace("]", "")
    text = re.sub(r"\[\s*(?=[\[\(])", "", text)  # stray "[" left by nested amendment markers
    text = re.sub(r"(?<=[a-z\)”])\d{1,2}\b", "", text)  # trailing footnote markers: "Advisers1"
    text = re.sub(r"-\s*\n\s*(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def regulation_chunks(pages):
    lines = [l for p in pages for l in strip_page(p)]
    start = next(i for i, l in enumerate(lines) if re.match(r"^\s*CHAPTER I\b", l))
    regs, cur = [], None
    for l in lines[start:]:
        m = re.match(r"^(\d+[A-Z]?)\.\s+(\S.*)$", l)
        if m and not l.startswith(" "):
            if regs and int(re.sub(r"\D", "", m.group(1))) < int(re.sub(r"\D", "", regs[-1]["reg"])):
                break  # numbering restarts in the schedules after the last regulation
            cur = {"reg": m.group(1), "lines": [m.group(2)]}; regs.append(cur)
        elif cur and not re.match(r"^\s*(CHAPTER|SCHEDULE|FORM)\b", l):
            if l.strip() and not l.startswith(" ") and l.strip().endswith(".") and len(l.strip()) < 60:
                continue  # marginal heading of the next regulation, e.g. "Suitability."
            cur["lines"].append(l)
    out = []
    for r in regs:
        body = re.sub(r"\d{1,3}\[", "", "\n".join(r["lines"]))  # "65[(14)" -> "(14)" before locating sub-clauses
        marks = list(re.finditer(r"(?m)^\s*\((\d+|[a-z]{1,2})\)\s", body))
        first = [m for m in marks if m.group(1) not in ROMAN or m.group(1) == "i" and not any(
            x.group(1) == "h" for x in marks)]
        # keep only the first-level sequence: (1),(2),... or (a),(b),...; numbers may skip omitted sub-regulations
        seq, kind = [], None
        for m in first:
            tag = m.group(1)
            k = "num" if tag.isdigit() else "alpha"
            if k == "num":
                ok = (not seq and tag == "1") or (seq and kind == "num" and 0 < int(tag) - int(seq[-1].group(1)) <= 3)
            else:
                ok = kind in (None, "alpha") and tag == chr(ord("a") + len(seq))
            if ok:
                seq.append(m); kind = k
        lead = clean(body[: seq[0].start()]) if seq else ""
        if not seq:
            out.append({"reg": r["reg"], "sub": "", "text": clean(body)})
            continue
        for j, m in enumerate(seq):
            end = seq[j + 1].start() if j + 1 < len(seq) else len(body)
            part = clean(body[m.start():end])
            out.append({"reg": r["reg"], "sub": m.group(1), "text": (lead + " " + part).strip() if lead else part})
    return split_long([c for c in out if len(c["text"]) >= 40])


def split_long(chunks, max_chars=1500, target=1000):
    """Split oversized chunks at sentence or semicolon boundaries; parts keep reg/sub with a -k suffix."""
    out = []
    for c in chunks:
        if len(c["text"]) <= max_chars:
            out.append(c); continue
        pieces, cur = [], ""
        for s in re.split(r"(?<=[.;])\s+", c["text"]):
            if cur and len(cur) + len(s) > target:
                pieces.append(cur); cur = ""
            cur = (cur + " " + s).strip()
        pieces.append(cur)
        for k, t in enumerate(pieces, 1):
            out.append({**c, "sub": f"{c['sub']}-{k}" if c["sub"] else str(k), "text": t})
    return out


def circular_chunks(pages, max_chars=1000):
    body_pages = pages[5:]  # skip covering letter (pp. 1-2) and table of contents (pp. 3-5)
    lines = [l for p in body_pages for l in strip_page(p)]
    out, sec, buf = [], "0", []

    def flush():
        t = clean("\n".join(buf))
        if len(t) >= 40:
            out.append({"reg": f"MC{sec}", "sub": str(sum(c["reg"] == f"MC{sec}" for c in out) + 1), "text": t})
        buf.clear()

    para = []
    for l in lines + [""]:
        m = re.match(r"^(\d{1,2})\.\s+[A-Z]", l)
        if m:
            if para: buf.append("\n".join(para)); para = []
            flush(); sec = m.group(1)
            para.append(l)
        elif l.strip():
            para.append(l)
        elif para:
            block = "\n".join(para); para = []
            if len(clean("\n".join(buf + [block]))) > max_chars and buf:
                flush()
            buf.append(block)
    flush()
    return split_long(out)


ap = argparse.ArgumentParser()
ap.add_argument("regs_pdf"); ap.add_argument("circular_pdf")
ap.add_argument("--out", default="data/regulatory/sebi_full.jsonl")
a = ap.parse_args()
regs = regulation_chunks(pdf_pages(a.regs_pdf))
circ = circular_chunks(pdf_pages(a.circular_pdf))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w") as f:
    for c in regs:
        ref = f"{c['reg']}({c['sub']})" if c["sub"] else c["reg"]
        f.write(json.dumps({"id": f"SEBI-IA-{ref}", "reg": c["reg"], "sub": c["sub"],
                            "source": f"SEBI (Investment Advisers) Regulations, 2013, Reg. {ref}",
                            "text": c["text"]}) + "\n")
    for c in circ:
        f.write(json.dumps({"id": f"SEBI-{c['reg']}-{c['sub']}", "reg": c["reg"], "sub": c["sub"],
                            "source": f"SEBI Master Circular for Investment Advisers (Feb 2026), section {c['reg'][2:]}",
                            "text": c["text"]}) + "\n")
print(f"wrote {len(regs)} regulation chunks + {len(circ)} circular chunks = {len(regs) + len(circ)} to {a.out}")
