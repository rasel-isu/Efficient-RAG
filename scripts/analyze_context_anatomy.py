"""Decompose the context into bookkeeping headers vs actual passage content.

Every condition renders its context as "[Snippet i | score=s]\\n<text>", and the
headers are counted in `compressed_context_tokens`. At top-30 that bookkeeping
is a large fixed cost, so the reported compression ratio understates how far the
passage TEXT is actually compressed. This script separates the two.
"""
import argparse, glob, json, os, re, sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HDR = re.compile(r"\[Snippet \d+ \| score=[-\d.]+\]")
DS_LABEL = {"triviaqa": "TriviaQA", "nq_open": "NQ-Open", "hotpotqa": "HotpotQA",
            "2wiki": "2WikiMultihopQA", "musique": "MuSiQue"}
ORDER = ["triviaqa", "nq_open", "hotpotqa", "2wiki", "musique"]


def anatomy(path, tok):
    rows = json.load(open(path))
    hdr, body, snips, short, words = [], [], [], 0, []
    for r in rows:
        ctx = r.get("compressed_context", "")
        heads = HDR.findall(ctx)
        bodies = [p.strip() for p in HDR.split(ctx)[1:]]
        hdr.append(sum(len(tok.encode(h, add_special_tokens=False)) for h in heads))
        body.append(sum(len(tok.encode(b, add_special_tokens=False)) for b in bodies))
        snips.append(len(heads))
        short += sum(1 for b in bodies if len(b.split()) <= 3)
        words += [len(b.split()) for b in bodies]
    if not rows:
        return None
    return dict(n=len(rows), snips=float(np.mean(snips)), hdr=float(np.mean(hdr)),
                body=float(np.mean(body)),
                per_snip=float(np.mean(body)) / max(np.mean(snips), 1),
                words=float(np.mean(words)) if words else 0.0,
                short_pct=short / max(sum(snips), 1) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", default="off")
    ap.add_argument("--reader", default="meta-llama/Llama-3.2-3B-Instruct")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.reader, use_fast=True)

    print(f"Context anatomy (rerank {args.rerank}); tokens counted with the reader's tokenizer\n")
    print(f"{'Dataset':<18}{'cond':<13}{'hdr':>6}{'body':>7}{'tok/snip':>9}"
          f"{'words/snip':>11}{'<=3w':>7}{'ratio(all)':>11}{'ratio(body)':>12}")
    for ds in ORDER:
        base = anatomy(f"OUTPUT/full/{ds}_baseline_rerank-{args.rerank}.json", tok)
        if not base:
            continue
        for cond in ["baseline", "truncate", "filter_summ"]:
            p = f"OUTPUT/full/{ds}_{cond}_rerank-{args.rerank}.json"
            if not os.path.exists(p):
                continue
            a = anatomy(p, tok)
            r_all = (base["hdr"] + base["body"]) / max(a["hdr"] + a["body"], 1)
            r_body = base["body"] / max(a["body"], 1)
            print(f"{DS_LABEL[ds]:<18}{cond:<13}{a['hdr']:>6.0f}{a['body']:>7.0f}"
                  f"{a['per_snip']:>9.1f}{a['words']:>11.1f}{a['short_pct']:>6.0f}%"
                  f"{r_all:>10.1f}x{r_body:>11.1f}x")
    print("\nratio(all) counts the snippet headers, as the reported compression rate does;\n"
          "ratio(body) counts only passage text.")


if __name__ == "__main__":
    main()
