"""Split a regulatory text file (that YOU obtained, e.g. MiFID II / ESMA guidelines / SEBI circular)
into clause-level JSONL for data/regulatory/clauses.jsonl.

  python scripts/split_clauses.py mifid2.txt --source "MiFID II" --out data/regulatory/clauses.jsonl --append
Splits on blank lines / numbered paragraphs; drops fragments shorter than 40 chars.
"""
import argparse, json, re
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("txt"); ap.add_argument("--source", required=True)
ap.add_argument("--out", default="data/regulatory/clauses.jsonl"); ap.add_argument("--append", action="store_true")
a = ap.parse_args()
text = Path(a.txt).read_text(errors="ignore")
parts = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n|(?=\n\s*\(?\d+[.)]\s)", text)]
parts = [p for p in parts if len(p) >= 40]
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
tag = re.sub(r"\W+", "", a.source)[:12]
with open(a.out, "a" if a.append else "w") as f:
    for i, p in enumerate(parts, 1):
        f.write(json.dumps({"id": f"{tag}-{i:04d}", "source": a.source, "text": p}) + "\n")
print(f"wrote {len(parts)} clauses to {a.out}")
