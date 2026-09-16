"""Aggregate a (dataset x condition x rerank) grid into the tables the paper needs."""
import glob, json, os, sys, re
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import exact_match, token_f1, gold_answers, supporting_fact_retention

DS_ORDER = ["triviaqa", "nq_open", "hotpotqa", "2wiki", "musique"]
COND_ORDER = ["baseline", "truncate", "filter_summ"]


def load(path):
    d = json.load(open(path))
    if not d:
        return None
    em = np.mean([exact_match(r["rag_answer"], gold_answers(r)) for r in d]) * 100
    f1 = np.mean([token_f1(r["rag_answer"], gold_answers(r)) for r in d]) * 100
    raw = np.mean([r.get("retrieved_context_tokens", 0) for r in d])
    ctx = np.mean([r.get("compressed_context_tokens", 0) for r in d])
    sfr = [supporting_fact_retention(r.get("supporting_facts"), r.get("compressed_context"))
           for r in d]
    sfr = [x[0] for x in sfr if x]
    n_correct = int(round(em / 100 * len(d)))
    return dict(n=len(d), em=em, f1=f1, raw=raw, ctx=ctx, n_correct=n_correct,
                ratio=raw / ctx if ctx else float("nan"),
                sfr=float(np.mean(sfr)) if sfr else float("nan"))


# A dataset whose EM is at or near the floor cannot separate conditions: the
# differences it shows are quantisation of a couple of questions, not signal.
# MuSiQue does this with a small reader, so it is reported on F1 and
# supporting-fact retention instead, and EM is marked unreliable rather than
# quietly presented alongside datasets where EM means something.
EM_FLOOR_CORRECT = 3     # fewer than this many correct answers -> EM unusable


def em_is_usable(cells, ds):
    """True when at least one condition for `ds` clears the floor."""
    vals = [v["n_correct"] for k, v in cells.items() if k[0] == ds]
    return bool(vals) and max(vals) >= EM_FLOOR_CORRECT


def token_tables(cells, datasets, floored):
    """The contribution tables: token cost first, accuracy as the thing traded away.

    This project's claim is efficiency - fewer tokens at acceptable accuracy loss -
    so the headline is tokens/question and reduction vs the uncompressed baseline,
    with accuracy reported as RETENTION (compressed / baseline) rather than as a
    standalone score.
    """
    print("\n=== TOKEN COST per question " + "=" * 52)
    print(f"{'dataset':10s} {'rerank':>6s} {'condition':>13s} {'prompt':>8s} "
          f"{'context':>8s} {'reduction':>10s} {'EM':>6s} {'retention':>10s}")
    print("-" * 78)
    for ds in datasets:
        for rr in ["off", "on"]:
            base = cells.get((ds, "baseline", rr))
            if not base:
                continue
            for c in COND_ORDER:
                v = cells.get((ds, c, rr))
                if not v:
                    continue
                red = (1 - v["ctx"] / base["ctx"]) * 100 if base["ctx"] else float("nan")
                ret = (v["em"] / base["em"] * 100) if base["em"] else float("nan")
                ret_s = "-" if ds in floored else (f"{ret:9.1f}%" if ret == ret else "-")
                em_s = "-" if ds in floored else f"{v['em']:6.1f}"
                print(f"{ds:10s} {rr:>6s} {c:>13s} {v['prompt']:8.0f} {v['ctx']:8.0f} "
                      f"{red:9.1f}% {em_s} {ret_s}")
    print("\nreduction = context tokens saved vs the uncompressed baseline")
    print("retention = compressed EM / baseline EM (omitted where EM is at the floor)")

    print("\n=== EFFICIENCY: accuracy per 1k prompt tokens " + "=" * 33)
    print(f"{'dataset':10s} {'rerank':>6s} " + "".join(f"{c:>14s}" for c in COND_ORDER))
    print("-" * 62)
    for ds in datasets:
        if ds in floored:
            continue
        for rr in ["off", "on"]:
            row = f"{ds:10s} {rr:>6s} "
            for c in COND_ORDER:
                v = cells.get((ds, c, rr))
                row += (f"{v['em']/(v['prompt']/1000):14.2f}" if v and v["prompt"] else f"{'-':>14s}")
            print(row)
    print("higher = more exact matches per 1k tokens spent")


def main(pattern):
    cells = {}
    for p in glob.glob(pattern):
        m = re.match(r"(.+)_(baseline|truncate|filter_summ)_rerank-(on|off)\.json$",
                     os.path.basename(p))
        if not m:
            continue
        got = load(p)
        if got:
            cells[(m.group(1), m.group(2), m.group(3))] = got
    if not cells:
        print(f"no grid cells matched {pattern}")
        return

    datasets = [d for d in DS_ORDER if any(k[0] == d for k in cells)]
    floored = [d for d in datasets if not em_is_usable(cells, d)]
    if floored:
        print("EM AT FLOOR (report F1 / supporting-fact retention instead): "
              + ", ".join(floored))
        print(f"  fewer than {EM_FLOOR_CORRECT} correct answers in every condition - "
              "EM differences here are noise, not signal.\n")
    for rr in ["off", "on"]:
        if not any(k[2] == rr for k in cells):
            continue
        print(f"\n=== rerank {rr.upper()} " + "=" * 62)
        print(f"{'dataset':10s} " + "".join(f"{c:>24s}" for c in COND_ORDER))
        print(f"{'':10s} " + "".join(f"{'EM    F1   comp  SFr':>24s}" for _ in COND_ORDER))
        print("-" * 84)
        for ds in datasets:
            row = f"{ds:10s}{'*' if ds in floored else ' '}"[:11]
            for c in COND_ORDER:
                v = cells.get((ds, c, rr))
                row += (f"{v['em']:6.1f}{v['f1']:6.1f}{v['ratio']:6.2f}x{v['sfr']:6.2f}"
                        if v else f"{'-':>24s}")
            print(row)

    # the contribution comparison: does reranking change the compression penalty?
    print("\n=== rerank effect on the compression penalty (EM) " + "=" * 32)
    print(f"{'dataset':10s} {'cond':>12s} {'rerank off':>11s} {'rerank on':>10s} {'Δ(on-off)':>10s}")
    print("-" * 58)
    for ds in datasets:
        if ds in floored:
            continue
        for c in COND_ORDER:
            a, b = cells.get((ds, c, "off")), cells.get((ds, c, "on"))
            if a and b:
                print(f"{ds:10s} {c:>12s} {a['em']:11.1f} {b['em']:10.1f} {b['em']-a['em']:+10.1f}")

    # compression penalty relative to uncompressed, per retrieval setting
    print("\n=== compression penalty vs baseline (EM points) " + "=" * 34)
    print(f"{'dataset':10s} {'rerank':>7s} {'truncate':>10s} {'filter_summ':>13s}")
    print("-" * 44)
    for ds in datasets:
        if ds in floored:
            continue
        for rr in ["off", "on"]:
            base = cells.get((ds, "baseline", rr))
            if not base:
                continue
            vals = []
            for c in ["truncate", "filter_summ"]:
                v = cells.get((ds, c, rr))
                vals.append(f"{v['em']-base['em']:+.1f}" if v else "-")
            print(f"{ds:10s} {rr:>7s} {vals[0]:>10s} {vals[1]:>13s}")

    _f1_table(cells, datasets)
    token_tables(cells, datasets, floored)


def _f1_table(cells, datasets):
    print("\n=== compression penalty vs baseline (F1 points) " + "=" * 34)
    print("F1 degrades gracefully where EM floors, so this table covers every dataset.")
    print(f"{'dataset':10s} {'rerank':>7s} {'truncate':>10s} {'filter_summ':>13s}")
    print("-" * 44)
    for ds in datasets:
        for rr in ["off", "on"]:
            base = cells.get((ds, "baseline", rr))
            if not base:
                continue
            vals = []
            for c in ["truncate", "filter_summ"]:
                v = cells.get((ds, c, rr))
                vals.append(f"{v['f1']-base['f1']:+.1f}" if v else "-")
            print(f"{ds:10s} {rr:>7s} {vals[0]:>10s} {vals[1]:>13s}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "OUTPUT/dryrun_grid/*.json")
