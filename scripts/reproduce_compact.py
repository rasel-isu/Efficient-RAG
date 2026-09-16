"""
Tier-1 reproduction gate.

Compares our measured Raw-Document numbers against CompAct (EMNLP 2024) Table 2.
If these do not land close, nothing downstream is trustworthy and the fix is to
debug the pipeline rather than to proceed to the contribution runs.

CompAct's protocol: Contriever-MSMARCO over DPR Wikipedia 2018, top-30 documents,
dev split of each dataset (TriviaQA: see note in the paper), no reranking.
Compression rate = raw tokens / compressed tokens.

READER NOTE: CompAct's published numbers use LLaMA3-8B. This project runs
Llama-3.2-3B, so absolute EM/F1 are NOT expected to match and an absolute
tolerance test would fail for reasons unrelated to pipeline correctness. The
gate therefore checks what a smaller reader should still preserve - the relative
ORDERING of dataset difficulty - and reports absolute gaps for documentation.
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import exact_match, token_f1, gold_answers

# CompAct Table 2, "Raw Document" row (reader = LLaMA3-8B, top-30)
PUBLISHED_RAW = {
    "hotpotqa": (29.4, 40.3),
    "musique":  (6.5, 15.6),
    "2wiki":    (25.4, 31.2),
    "nq_open":  (39.0, 51.3),
    "triviaqa": (68.9, 77.1),
}
# CompAct Table 2, "RECOMP (extractive)" row - our second baseline, same setup
PUBLISHED_RECOMP = {
    "hotpotqa": (29.7, 39.9), "musique": (6.7, 15.7), "2wiki": (29.9, 34.9),
    "nq_open":  (34.6, 45.1), "triviaqa": (67.6, 74.1),
}


def score(path):
    d = json.load(open(path))
    if not d:
        return None
    return (np.mean([exact_match(r["rag_answer"], gold_answers(r)) for r in d]) * 100,
            np.mean([token_f1(r["rag_answer"], gold_answers(r)) for r in d]) * 100,
            len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-glob", default="OUTPUT/reproduce/*_raw.json")
    ap.add_argument("--condition", default="baseline_rerank-off",
                    help="suffix used by run_grid.py outputs")
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="EM gap tolerance; only used with --absolute")
    ap.add_argument("--absolute", action="store_true",
                    help="require absolute agreement (valid only with a LLaMA3-8B-class reader)")
    args = ap.parse_args()

    print(f"{'dataset':12s} {'EM (ours)':>10s} {'EM (pub)':>9s} {'ΔEM':>7s} "
          f"{'F1 (ours)':>10s} {'F1 (pub)':>9s} {'ΔF1':>7s} {'n':>6s}  verdict")
    print("-" * 92)
    fails, checked, measured = 0, 0, {}
    for path in sorted(glob.glob(args.results_glob)):
        base = os.path.basename(path)
        # accept both run_reproduce.sh (<ds>_raw.json) and run_grid.py
        # (<ds>_<condition>_rerank-<on|off>.json) naming
        ds = base.replace("_raw.json", "").replace(f"_{args.condition}.json", "")
        ds = ds.replace(".json", "")
        if ds not in PUBLISHED_RAW:
            continue
        got = score(path)
        if not got:
            continue
        em, f1, n = got
        pem, pf1 = PUBLISHED_RAW[ds]
        dem, df1 = em - pem, f1 - pf1
        checked += 1
        measured[ds] = em
        if args.absolute:
            ok = abs(dem) <= args.tolerance
            fails += 0 if ok else 1
            verdict = "OK" if ok else "OFF"
        else:
            verdict = "-"
        print(f"{ds:12s} {em:10.1f} {pem:9.1f} {dem:+7.1f} {f1:10.1f} {pf1:9.1f} "
              f"{df1:+7.1f} {n:6d}  {verdict}")
    print("-" * 92)
    if checked == 0:
        print(f"NO RESULTS - nothing matched {args.results_glob}; run scripts/run_reproduce.sh first")
        return 2

    if args.absolute:
        print(f"PASS - {checked}/{checked} datasets within tolerance" if not fails
              else f"FAIL - {fails}/{checked} outside ±{args.tolerance} EM; debug before proceeding")
        return 1 if fails else 0

    # Rank correlation: a weaker reader should score lower everywhere but keep
    # the same dataset difficulty ordering. A broken pipeline usually does not.
    shared = [d for d in measured if d in PUBLISHED_RAW]
    ours = [measured[d] for d in shared]
    pub = [PUBLISHED_RAW[d][0] for d in shared]
    if len(shared) >= 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr(ours, pub)
        print(f"\nSpearman rank correlation vs CompAct (LLaMA3-8B): rho={rho:.3f} (p={p:.3f}), n={len(shared)}")
        print("dataset difficulty ordering preserved" if rho >= 0.8
              else "ORDERING DIVERGES from published - inspect before proceeding")
        print("NOTE: absolute EM is not comparable (reader is Llama-3.2-3B, not LLaMA3-8B).")
        return 0 if rho >= 0.8 else 1
    print("need >=3 datasets for the ordering check")
    return 2


if __name__ == "__main__":
    sys.exit(main())
