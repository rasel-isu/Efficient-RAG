"""Does retrieval precision moderate the cost of compression?

The compression penalty is d = EM(compressed) - EM(uncompressed), measured on
the same questions. If cleaner retrieval makes compression more damaging, then
d should be more negative with the cross-encoder reranker on than off. The
interaction is (d_on - d_off), tested by a paired bootstrap over questions -
paired, because every cell evaluates the identical question set.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import exact_match, gold_answers

DS_LABEL = {"triviaqa": "TriviaQA", "nq_open": "NQ-Open", "hotpotqa": "HotpotQA",
            "2wiki": "2WikiMultihopQA", "musique": "MuSiQue"}
ORDER = ["triviaqa", "nq_open", "hotpotqa", "2wiki", "musique"]


def em_by_id(path):
    rows = json.load(open(path))
    return {str(r["id"]): exact_match(r["rag_answer"], gold_answers(r)) for r in rows}


def paired(ds, method="filter_summ", d="OUTPUT/full"):
    """Per-question compression penalty under each rerank setting, aligned by id."""
    cells = {(c, rr): em_by_id(f"{d}/{ds}_{c}_rerank-{rr}.json")
             for c in ("baseline", method) for rr in ("off", "on")}
    ids = set.intersection(*(set(v) for v in cells.values()))
    ids = sorted(ids)
    d_off = np.array([cells[(method, "off")][i] - cells[("baseline", "off")][i] for i in ids])
    d_on = np.array([cells[(method, "on")][i] - cells[("baseline", "on")][i] for i in ids])
    return ids, d_off, d_on


def boot(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="OUTPUT/full")
    ap.add_argument("--method", default="filter_summ")
    args = ap.parse_args()

    print("Compression penalty (EM points) and the rerank interaction\n")
    print(f"{'Dataset':<18}{'penalty off':>26}{'penalty on':>26}{'interaction (on-off)':>30}")
    allo, alln = [], []
    for ds in ORDER:
        try:
            ids, d_off, d_on = paired(ds, args.method, args.dir)
        except FileNotFoundError:
            continue
        allo.append(d_off); alln.append(d_on)
        a = boot(d_off * 100); b = boot(d_on * 100); c = boot((d_on - d_off) * 100)
        sig = "  *" if (c[1] > 0) or (c[2] < 0) else ""
        print(f"{DS_LABEL[ds]:<18}{a[0]:>9.1f} [{a[1]:>5.1f},{a[2]:>5.1f}]"
              f"{b[0]:>12.1f} [{b[1]:>5.1f},{b[2]:>5.1f}]"
              f"{c[0]:>16.1f} [{c[1]:>5.1f},{c[2]:>5.1f}]{sig}  (n={len(ids)})")
    if allo:
        d_off, d_on = np.concatenate(allo), np.concatenate(alln)
        a = boot(d_off * 100); b = boot(d_on * 100); c = boot((d_on - d_off) * 100)
        sig = "  *" if (c[1] > 0) or (c[2] < 0) else ""
        print(f"{'POOLED':<18}{a[0]:>9.1f} [{a[1]:>5.1f},{a[2]:>5.1f}]"
              f"{b[0]:>12.1f} [{b[1]:>5.1f},{b[2]:>5.1f}]"
              f"{c[0]:>16.1f} [{c[1]:>5.1f},{c[2]:>5.1f}]{sig}  (n={len(d_off)})")
    print("\n95% paired bootstrap CIs over questions; * = CI excludes zero.")


if __name__ == "__main__":
    main()
