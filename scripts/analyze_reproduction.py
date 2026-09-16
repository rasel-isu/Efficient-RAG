"""
Detailed analysis of the Tier-1 reproduction, written to a markdown report.

Covers what the slurm log's one-line gate does not: bootstrap confidence
intervals, comparison against every published baseline (not just CompAct's Raw
Document row), per-question-type breakdowns, retrieval diagnostics and cost.
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import (exact_match, token_f1, gold_answers, question_type,
                        bootstrap_ci, supporting_fact_retention)

# --- published numbers, for comparison -------------------------------------
# Shared with scripts/analyze_grid.py so the two reports can never disagree.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from published import COMPACT, RECOMP_PAPER  # noqa: E402

DS_LABEL = {"triviaqa": "TriviaQA", "nq_open": "NQ-Open", "hotpotqa": "HotpotQA",
            "2wiki": "2WikiMultihopQA", "musique": "MuSiQue"}
ORDER = ["triviaqa", "nq_open", "hotpotqa", "2wiki", "musique"]


def analyse(path):
    d = json.load(open(path))
    ems = [exact_match(r["rag_answer"], gold_answers(r)) for r in d]
    f1s = [token_f1(r["rag_answer"], gold_answers(r)) for r in d]
    em, em_lo, em_hi = bootstrap_ci(ems)
    f1, f1_lo, f1_hi = bootstrap_ci(f1s)

    by_type = defaultdict(list)
    for r, e in zip(d, ems):
        by_type[r.get("question_type") or question_type(r["question"], gold_answers(r))].append(e)

    # `n_gold_retrieved` is meaningless in corpus mode: passages come from the
    # 21M-passage Wikipedia index, not the dataset's annotated pool, so nothing
    # carries an is_gold flag. Measure evidence recall directly instead - does
    # the retrieved context actually contain the gold supporting sentences?
    sf_hits, em_hit, em_miss = [], [], []
    for r, e in zip(d, ems):
        sf = r.get("supporting_facts")
        if not sf:
            continue
        ctx = "\n".join(c["text"] for c in r.get("retrieved_chunks", []))
        got = supporting_fact_retention(sf, ctx)
        if got is None:
            continue
        sf_hits.append(got[0])
        (em_hit if got[0] >= 0.5 else em_miss).append(e)

    return dict(
        n=len(d), em=em * 100, em_ci=(em_lo * 100, em_hi * 100),
        f1=f1 * 100, f1_ci=(f1_lo * 100, f1_hi * 100),
        ctx=np.mean([r["retrieved_context_tokens"] for r in d]),
        prompt=np.mean([r["prompt_token"] for r in d]),
        compl=np.mean([r["completion_token"] for r in d]),
        t_ret=np.mean([r["t_retrieval_ms"] for r in d]),
        t_gen=np.mean([r["t_generate_ms"] for r in d]),
        t_e2e=np.mean([r["t_end_to_end_ms"] for r in d]),
        sf_recall=float(np.mean(sf_hits)) * 100 if sf_hits else float("nan"),
        em_hit=float(np.mean(em_hit)) * 100 if em_hit else float("nan"),
        em_miss=float(np.mean(em_miss)) * 100 if em_miss else float("nan"),
        n_hit=len(em_hit), n_miss=len(em_miss), n_sf=len(sf_hits),
        by_type={k: (len(v), float(np.mean(v)) * 100) for k, v in
                 sorted(by_type.items(), key=lambda x: -len(x[1]))},
        reader=d[0]["generator"], top_k=d[0]["top_k"], n_shot=d[0]["n_shot"],
        encoder=d[0]["encoder"], reranked=d[0]["reranked"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="OUTPUT/reproduce/*_baseline_rerank-off.json")
    ap.add_argument("--out", default="RESULTS_reproduction.md")
    args = ap.parse_args()

    res = {}
    for p in glob.glob(args.glob):
        ds = os.path.basename(p).replace("_baseline_rerank-off.json", "")
        res[ds] = analyse(p)
    if not res:
        raise SystemExit(f"no files matched {args.glob}")
    datasets = [d for d in ORDER if d in res]
    any_cfg = res[datasets[0]]

    L = []
    w = L.append
    w("# Tier-1 Reproduction — Detailed Analysis\n")
    w(f"Uncompressed (\"Raw Document\") retrieval-augmented QA on five benchmarks, "
      f"run to validate the pipeline against published baselines before any "
      f"compression experiments.\n")
    w("## Configuration\n")
    w("| | |")
    w("|---|---|")
    w(f"| Reader | `{any_cfg['reader']}` |")
    w(f"| Retriever | Contriever-MSMARCO (`{any_cfg['encoder']}`), dot-product, no reranking |")
    w(f"| Corpus | DPR Wikipedia 2018, 21,015,324 passages |")
    w(f"| Depth | top-{any_cfg['top_k']} |")
    w(f"| Prompt | {any_cfg['n_shot']}-shot in-context (RECOMP format), greedy decoding |")
    w(f"| Split / size | dev, n={any_cfg['n']} per dataset |\n")

    w("## Token cost — the baseline this project aims to reduce\n")
    w("This project's contribution is token efficiency: cutting the tokens sent to the "
      "generator while keeping accuracy acceptable. These are the **uncompressed** "
      "costs that the compression conditions are measured against.\n")
    w("| Dataset | prompt tokens | of which context | compressible share |")
    w("|---|---|---|---|")
    for ds in datasets:
        r = res[ds]
        w(f"| {DS_LABEL[ds]} | **{r['prompt']:.0f}** | {r['ctx']:.0f} | "
          f"{r['ctx']/r['prompt']*100:.1f}% |")
    w("")
    w("The remainder is the 5-shot in-context block (~100-145 tokens) plus the question. "
      "That part is a fixed cost and cannot be compressed away, but at top-30 it is under "
      "3% of the prompt, so essentially the whole prompt is addressable.\n")
    tot = sum(res[d]["prompt"] * res[d]["n"] for d in datasets)
    w(f"Across all {sum(res[d]['n'] for d in datasets)} questions in this run the "
      f"uncompressed setting consumed **{tot/1e6:.2f}M prompt tokens**.\n")

    w("## Headline accuracy\n")
    w("| Dataset | EM | 95% CI | F1 | 95% CI | ctx tokens |")
    w("|---|---|---|---|---|---|")
    for ds in datasets:
        r = res[ds]
        w(f"| {DS_LABEL[ds]} | **{r['em']:.1f}** | [{r['em_ci'][0]:.1f}, {r['em_ci'][1]:.1f}] "
          f"| **{r['f1']:.1f}** | [{r['f1_ci'][0]:.1f}, {r['f1_ci'][1]:.1f}] | {r['ctx']:.0f} |")
    w("")

    w("## Comparison with published methods\n")
    w("CompAct (EMNLP 2024) Table 2 — reader **LLaMA3-8B**, same retriever/corpus/depth. "
      "Our reader is 3B, so absolute parity is not expected; the comparison shows where "
      "this pipeline sits relative to published systems.\n")
    for ds in datasets:
        if ds not in COMPACT:
            continue
        r = res[ds]
        w(f"**{DS_LABEL[ds]}**\n")
        w("| System | Reader | EM | F1 | ΔEM vs ours |")
        w("|---|---|---|---|---|")
        w(f"| **Ours (uncompressed)** | Llama-3.2-3B | **{r['em']:.1f}** | **{r['f1']:.1f}** | — |")
        for name, (e, f) in COMPACT[ds].items():
            w(f"| {name} | LLaMA3-8B | {e:.1f} | {f:.1f} | {r['em']-e:+.1f} |")
        w("")

    w("### RECOMP (ICLR 2024) — Flan-UL2 20B, Contriever top-5\n")
    w("A different reader and retrieval depth, so this is context rather than a "
      "like-for-like comparison. The `T5 (off-the-shelf)` row is the closest published "
      "analogue to this project's original method.\n")
    for ds in datasets:
        if ds not in RECOMP_PAPER:
            continue
        r = res[ds]
        w(f"**{DS_LABEL[ds]}** (ours uncompressed: EM {r['em']:.1f} / F1 {r['f1']:.1f})\n")
        w("| System | EM | F1 |")
        w("|---|---|---|")
        for name, (e, f) in RECOMP_PAPER[ds].items():
            w(f"| {name} | {e:.1f} | {f:.1f} |")
        w("")

    w("## Retrieval diagnostics — evidence recall\n")
    w("Retrieval is over the full 21M-passage Wikipedia index, so no retrieved passage "
      "carries a dataset `is_gold` flag. Instead we measure whether the gold supporting "
      "sentences actually appear in the retrieved top-30 (token recall >= 0.6 per "
      "sentence), and split accuracy by whether the evidence was found. Only the "
      "multi-hop sets ship sentence-level supporting facts.\n")
    w("| Dataset | evidence recall | EM when evidence found | EM when missed | n (found/missed) |")
    w("|---|---|---|---|---|")
    for ds in datasets:
        r = res[ds]
        if np.isnan(r["sf_recall"]):
            w(f"| {DS_LABEL[ds]} | n/a (no supporting-fact annotations) | — | — | — |")
        else:
            w(f"| {DS_LABEL[ds]} | {r['sf_recall']:.1f}% | {r['em_hit']:.1f} "
              f"| {r['em_miss']:.1f} | {r['n_hit']}/{r['n_miss']} |")
    w("")

    w("## Cost and latency (per question)\n")
    w("Retrieval is negligible (~30-40 ms over 21M passages); generation dominates, and "
      "generation time is driven by encoding the retrieved context. Fewer context tokens "
      "therefore buys latency as well as cost.\n")
    w("| Dataset | prompt tok | completion tok | retrieval ms | generation ms | total ms |")
    w("|---|---|---|---|---|---|")
    for ds in datasets:
        r = res[ds]
        w(f"| {DS_LABEL[ds]} | {r['prompt']:.0f} | {r['compl']:.1f} | {r['t_ret']:.0f} "
          f"| {r['t_gen']:.0f} | {r['t_e2e']:.0f} |")
    w("")

    w("## Accuracy by question type\n")
    for ds in datasets:
        r = res[ds]
        if len(r["by_type"]) <= 1 and list(r["by_type"])[0] in ("None", None):
            continue
        w(f"**{DS_LABEL[ds]}** — " + ", ".join(
            f"{k}: {v[1]:.1f}% (n={v[0]})" for k, v in r["by_type"].items() if k) + "\n")

    open(args.out, "w").write("\n".join(L))
    print(f"wrote {args.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
