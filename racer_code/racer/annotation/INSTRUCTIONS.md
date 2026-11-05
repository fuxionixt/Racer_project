# Annotation instructions

Thank you for helping. There are two independent tasks; you may do either or both. Please work alone and do not
discuss items with other annotators until everyone has finished. Your own professional judgement is what matters.
There are no trick items.

Save your copy of each sheet under your initials, e.g. `cce_AB.csv` and `clauses_AB.csv`, and fill in only the
blank columns.

---

## Task 1: Is this fund suitable for this client? (`cce_expert_sheet.csv`, 100 rows, about 60 to 90 minutes)

Each row pairs one Indian mutual fund with one retail client. Decide whether recommending this fund to this client
would be **suitable** in the sense of the SEBI (Investment Advisers) Regulations, Regulation 17:

> (a) All investments on which investment advice is provided is appropriate to the risk profile of the client;
> (d) [the adviser] has a reasonable basis for believing that a recommendation ... (i) meets the client's investment
> objectives; (ii) is such that the client is able to bear any related investment risks consistent with its
> investment objectives and risk tolerance ...

**The client.** Only the client's risk capacity is known:

| Capacity | Meaning |
|---|---|
| LOW | Weakest third of clients by income, savings and credit history; cannot bear losses |
| MEDIUM | Middle third; can bear moderate fluctuations |
| HIGH | Strongest third; can bear high volatility and drawdowns |

Assume a general long-term savings goal; there is no further information about the client.

**The fund.** Columns are taken from the public catalogue:

- `riskometer_level`: SEBI mutual-fund riskometer, 1 (Low) to 6 (Very High)
- `expense_ratio_pct`: annual cost as % of assets
- `rating_0_5`: star rating; **0 means unrated**
- `std_dev_pct`: volatility of returns
- `sharpe`: risk-adjusted return
- `return_3yr_pct`: annualized 3-year return

For reference, the catalogue medians are: expense ratio 0.63%, rating 3, standard deviation 12.22, Sharpe 1.30.

**Fill in**

- `label`: exactly `Suitable` or `Unsuitable`
- `confidence`: 1 (unsure), 2 (fairly sure), 3 (certain)
- `comment`: optional, but helpful when you are unsure or the case is borderline

---

## Task 2: Which regulatory passages matter for suitability? (`clause_relevance_sheet.csv`, 322 rows, about 2 hours)

Each row is a passage from the SEBI (Investment Advisers) Regulations, 2013, or the SEBI Master Circular for
Investment Advisers (Feb 2026). Imagine an adviser about to **recommend a specific mutual fund to a client** and
asking: *which rules govern whether this recommendation is suitable?* Grade each passage:

| Grade | Meaning | Example |
|---|---|---|
| 2 | Directly governs whether a recommendation is suitable | Reg. 17(a): investments must be appropriate to the client's risk profile |
| 1 | Related: the adviser should consider it, but it does not decide suitability | Risk-profiling procedure; disclosure of a product's key features |
| 0 | Not relevant to this decision | Registration fees, office records, cybersecurity |

Grade the passage on its own text, not on the document it comes from.
