import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.model_selection import train_test_split
from racer import config
from racer.data import load_funds, load_profiles, synth_funds, synth_profiles, load_regulatory

QUERIES = ["long term wealth creation", "stable regular income", "tax saving investment", "retirement planning",
           "short term parking of surplus cash", "diversified portfolio growth", "low cost index fund", "capital protection"]

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
