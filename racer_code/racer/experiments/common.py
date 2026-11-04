import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.model_selection import train_test_split
from racer import config
from racer.data import load_funds, load_profiles, synth_funds, synth_profiles, load_regulatory

QUERIES = ["long term wealth creation", "stable regular income", "tax saving investment", "retirement planning",
           "short term parking of surplus cash", "diversified portfolio growth", "low cost index fund", "capital protection"]

# 64 investor queries (the 8 above plus 56 more) for the larger generation and filtering experiments.
EXTENDED_QUERIES = QUERIES + [
    "child education fund in ten years", "monthly SIP for a first-time investor", "emergency fund with easy withdrawal",
    "high growth small cap exposure", "large cap blue chip stability", "balanced fund for moderate risk",
    "debt fund with low volatility", "liquid fund for three months", "gilt fund for interest rate gains",
    "corporate bond fund with good rating", "hybrid fund for a conservative retiree", "aggressive equity for young investor",
    "sector fund in technology", "banking and financial services fund", "infrastructure growth fund",
    "pharma and healthcare fund", "international equity diversification", "dividend yield income fund",
    "value investing fund", "flexi cap fund with low expense ratio", "mid cap fund with strong Sharpe ratio",
    "ELSS with three year lock in", "fund with highest rating and low cost", "low risk fund for senior citizen",
    "inflation beating returns over five years", "short duration debt for one year", "ultra short term fund for parking bonus",
    "overnight fund for treasury cash", "arbitrage fund for low tax", "multi asset allocation fund",
    "dynamic asset allocation fund", "equity savings fund", "credit risk fund with higher yield",
    "money market fund", "floater fund for rising rates", "banking and PSU debt fund",
    "index fund tracking Nifty 50", "fund for a house down payment in three years", "wedding expenses in two years",
    "fund for a car purchase next year", "long term pension planning", "fund with consistent five year returns",
    "low beta defensive fund", "fund with small drawdowns", "focused fund with concentrated portfolio",
    "contra investing fund", "ESG sustainable investing fund", "fund for NRI investor",
    "quarterly income for expenses", "surplus bonus investment for growth", "first investment with small amount",
    "low minimum SIP amount fund", "tax efficient debt fund", "fund manager with long track record",
    "fund with large assets under management", "safe fund for unemployed period"]

def parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--funds", default=str(config.DATA_DIR / "funds" / "funds.csv"))
    ap.add_argument("--profiles", default=str(config.DATA_DIR / "profiles" / "profiles.csv"))
    ap.add_argument("--regulatory", default=str(config.DATA_DIR / "regulatory" / "clauses.jsonl"))
    ap.add_argument("--smoke", action="store_true", help="SYNTHETIC data; plumbing test only; never report these numbers")
    return ap

def out_dir(smoke):
    d = config.RESULTS_DIR / ("smoke" if smoke else ""); d.mkdir(parents=True, exist_ok=True); return d

def get_funds(a):
    if a.smoke: return synth_funds(400)
    if not Path(a.funds).exists(): sys.exit(f"Missing {a.funds}: place the Kaggle 'Mutual Funds India' CSV there (see README).")
    return load_funds(a.funds)

def get_profiles(a):
    if a.smoke: return synth_profiles(800)
    if not Path(a.profiles).exists(): sys.exit(f"Missing {a.profiles}: place the 'User Financial Profiles' CSV there (see README).")
    return load_profiles(a.profiles)

def split(df, strat, seed=config.SEED):
    tr, te = train_test_split(df, test_size=0.25, stratify=strat, random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)

def save(a, name, obj):
    if a.smoke: obj = {"SYNTHETIC_SMOKE_TEST_DO_NOT_REPORT": True, **obj}
    p = out_dir(a.smoke) / name; p.write_text(json.dumps(obj, indent=2, default=float)); print("wrote", p)
