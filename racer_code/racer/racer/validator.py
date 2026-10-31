"""Post-generation validator + programmatic hallucination metrics (no LLM judge needed)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:%)?")
_CITE_P = re.compile(r"\[P(\d+)\]")
_CITE_R = re.compile(r"\[R(\d+)\]")
_FUNDISH = re.compile(r"\b(?:[A-Z][\w&\-]*\s){1,5}(?:Fund|Scheme|Plan|ETF)\b")


@dataclass
class Report:
    ok: bool
    metrics: dict
    issues: list = field(default_factory=list)

    @property
    def feedback(self) -> str:
        return "; ".join(self.issues)


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.rstrip("%")))
        except ValueError:
            pass
    return out


def validate(text: str, ctx: dict, catalogue_names=(), rules=None, products_df=None) -> Report:
    """ctx = {tier, products:[{name,...}], clauses:[{text,...}]}."""
    issues = []
    allowed = sorted({p["name"].lower() for p in ctx["products"]}, key=len, reverse=True)
    work = text

    def mask(names):
        """Remove whole-token occurrences of names (longest first) so substrings are not double counted."""
        nonlocal work
        found = []
        for n in names:
            pat = re.compile(r"(?<![\w])" + re.escape(n) + r"(?![\w])", re.I)
            if pat.search(work):
                found.append(n)
                work = pat.sub(" \x00 ", work)
        return found

    # 1. product grounding: catalogue names used but not retrieved, and invented fund-like names
    mentioned = mask(allowed)
    others = sorted({n.lower() for n in catalogue_names} - set(allowed), key=len, reverse=True)
    outside = mask(others)
    inv = []
    for m in _FUNDISH.findall(work):
        if "\x00" not in m:
            inv.append(m.strip())
    n_prod_mentions = len(mentioned) + len(outside) + len(inv)
    prod_ground = 1.0 if n_prod_mentions == 0 else len(mentioned) / n_prod_mentions
    if outside:
        issues.append(f"products not in the retrieved set: {outside[:3]}")
    if inv:
        issues.append(f"unknown product names: {inv[:3]}")

    # 2. citation validity
    pc = [int(i) for i in _CITE_P.findall(text)]
    rc = [int(i) for i in _CITE_R.findall(text)]
    bad_p = [i for i in pc if not 1 <= i <= len(ctx["products"])]
    bad_r = [i for i in rc if not 1 <= i <= len(ctx["clauses"])]
    n_cites = len(pc) + len(rc)
    cite_valid = 1.0 if n_cites == 0 else 1 - (len(bad_p) + len(bad_r)) / n_cites
    if bad_p or bad_r:
        issues.append(f"invalid citations: P{bad_p} R{bad_r}")

    # 3. numeric grounding: every number must occur in the context (ignoring citation indices)
    body = _CITE_P.sub("", _CITE_R.sub("", work.replace("\x00", " ")))
    ctx_nums = []
    for p in ctx["products"]:
        for v in p.values():
            if isinstance(v, (int, float)):
                ctx_nums.append(float(v))
    for c in ctx["clauses"]:
        ctx_nums += _numbers(c["text"])
    nums = _numbers(body)
    ungrounded = [n for n in nums if not any(abs(n - c) <= 1e-6 + 0.005 * abs(c) for c in ctx_nums)]
    num_ground = 1.0 if not nums else 1 - len(ungrounded) / len(nums)
    if ungrounded:
        issues.append(f"numbers not found in context: {ungrounded[:4]}")

    # 4. suitability: any mentioned (retrieved) product failing the hard checks for the user's tier
    viol = 0
    if rules is not None and products_df is not None:
        for a in mentioned:
            row = products_df[products_df["scheme_name"].str.lower() == a]
            if len(row):
                good, r = rules.passes(row.iloc[0], ctx["tier"])
                if not good:
                    viol += 1
        if viol:
            issues.append(f"{viol} recommended product(s) violate suitability rules for tier {ctx['tier']}")

    metrics = {"product_grounding": prod_ground, "citation_validity": cite_valid,
               "numeric_grounding": num_ground, "suitability_violations": viol,
               "has_hallucination": bool(outside or inv or bad_p or bad_r or ungrounded)}
    return Report(ok=not issues, metrics=metrics, issues=issues)
