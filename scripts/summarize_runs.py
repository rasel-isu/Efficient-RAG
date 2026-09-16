"""Aggregate any set of result JSONs into one comparison table."""
import json, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import exact_match, token_f1, gold_answers, supporting_fact_retention

rows = []
for path in sys.argv[1:]:
    d = json.load(open(path))
    if not d:
        continue
    em = np.mean([exact_match(r["rag_answer"], gold_answers(r)) for r in d])
    f1 = np.mean([token_f1(r["rag_answer"], gold_answers(r)) for r in d])
    ctx = np.mean([r.get("compressed_context_tokens", 0) for r in d])
    raw = np.mean([r.get("retrieved_context_tokens", 0) for r in d])
    gold = [r["n_gold_retrieved"] for r in d if "n_gold_retrieved" in r]
    sfr = [supporting_fact_retention(r.get("supporting_facts"), r.get("compressed_context"))
           for r in d]
    sfr = [x for x in sfr if x]
    rows.append((os.path.basename(path).replace(".json", ""), len(d), em * 100, f1,
                 raw, ctx, raw / ctx if ctx else 0,
                 np.mean(gold) if gold else float("nan"),
                 np.mean([x[0] for x in sfr]) if sfr else float("nan")))

hdr = f"{'run':28s} {'n':>4s} {'EM%':>6s} {'F1':>6s} {'rawTok':>7s} {'ctxTok':>7s} {'ratio':>6s} {'gold@k':>7s} {'SFret':>6s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r[0]:28s} {r[1]:4d} {r[2]:6.2f} {r[3]:6.3f} {r[4]:7.0f} {r[5]:7.0f} {r[6]:6.2f}x {r[7]:7.2f} {r[8]:6.2f}")
