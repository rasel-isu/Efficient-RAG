"""
Run a whole (dataset x condition x rerank) grid in ONE process.

run_benchmark.py reloads the 32 GB Contriever index on every invocation, which
costs ~5 minutes per cell and dominates runtime for a 30-cell grid. Here the
index, the reader and the summariser are each loaded once and reused; only the
cheap parts (reranker toggle, dataset swap) vary per cell.
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm

from rag.indexing import Indexing
from rag.datasets import LOADERS, inject_cross_question_distractors
from main import CONDITIONS

DATASETS = ["hotpotqa", "2wiki", "musique", "nq_open", "triviaqa"]


def check_reusable(path, expect, want_ids):
    """Decide whether an existing cell may be reused, or is stale.

    Resume-on-restart is essential for long jobs, but a cell that was produced
    under different settings must NOT be silently skipped: the run then reports
    old numbers as if they were new. This compares the stored config and the
    exact question ids, and returns a reason string when they disagree.
    """
    try:
        prev = json.load(open(path))
    except Exception as e:
        return f"unreadable ({type(e).__name__})"
    if not prev:
        return "empty file"
    got = prev[0]
    for k, want in expect.items():
        have = got.get(k, "<absent>")
        if have != want:
            return f"{k}: file has {have!r}, run wants {want!r}"
    if len(prev) != len(want_ids):
        return f"row count: file has {len(prev)}, run wants {len(want_ids)}"
    if [str(r.get("id")) for r in prev] != [str(i) for i in want_ids]:
        return "question ids differ (different sample)"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--conditions", nargs="+", default=["baseline", "truncate", "filter_summ"])
    ap.add_argument("--rerank", nargs="+", default=["off", "on"])
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42,
                    help="sampling seed; fixed so every cell evaluates the same questions")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--encoder", default="contriever")
    ap.add_argument("--retrieval", default="corpus", choices=["corpus", "pool"])
    ap.add_argument("--n-distractors", type=int, default=None)
    ap.add_argument("--summary-model", default="google/flan-t5-small")
    ap.add_argument("--truncate-to", type=int, default=200)
    ap.add_argument("--reader", default=None)
    ap.add_argument("--out-dir", default="OUTPUT/grid")
    ap.add_argument("--allow-stale", action="store_true",
                    help="reuse existing cells even if their config/sample differs "
                         "(default: abort, so stale results are never reported as new)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    indexing = Indexing([], reader=args.reader)

    needs_summ = any(CONDITIONS[c]["needs_model"] for c in args.conditions)
    summ = args.summary_model if needs_summ else None

    # Build the retriever ONCE with the reranker enabled, then toggle the
    # reranker attribute per cell - the 32 GB index stays resident.
    if args.retrieval == "corpus":
        from rag.global_retrieval import GlobalRAG
        index_dir = f"DATASET/wiki_dpr/index_{args.encoder}"
        if not os.path.exists(f"{index_dir}/embeddings.fp16.npy"):
            raise SystemExit(f"missing index {index_dir}")
        print(f"loading {index_dir} (one time) ...", flush=True)
        rag = GlobalRAG(indexing.token_counter, summ, index_dir=index_dir,
                        encoder=args.encoder, tokenizer=indexing.tokenizer,
                        top_k=args.top_k, use_reranker=True)
    else:
        from rag.local_retrieval import LocalRAG
        rag = LocalRAG(indexing.token_counter, summ, tokenizer=indexing.tokenizer,
                       top_k=args.top_k, use_reranker=True)
    reranker = rag.reranker

    for ds in args.datasets:
        examples = LOADERS[ds](n=args.n, seed=args.seed)
        if ds == "triviaqa" and args.retrieval == "pool":
            examples = inject_cross_question_distractors(examples, n=8)

        # Order matters: run the LM compressor first so `truncate` can be given a
        # budget matched to what that compressor actually produced on THIS
        # dataset. A fixed budget is not comparable across datasets - flan-t5
        # compresses NQ far less than the others (817 vs ~510 tokens).
        conds = sorted(args.conditions,
                       key=lambda c: {"filter_summ": 0, "baseline": 1, "truncate": 2}.get(c, 1))
        matched_budget = None

        for cond in conds:
            cfg = CONDITIONS[cond]
            for rr in args.rerank:
                tag = f"{ds}_{cond}_rerank-{rr}"
                path = f"{args.out_dir}/{tag}.json"
                if os.path.exists(path) and os.path.getsize(path) > 2:
                    expect = {
                        "sample_seed": args.seed, "top_k": args.top_k,
                        "n_shot": args.n_shot, "generator": indexing.model_name,
                        "encoder": args.encoder, "method": cond,
                        "reranked": rr == "on",
                        "retrieval": args.retrieval,
                    }
                    stale = check_reusable(path, expect, [e.id for e in examples])
                    if stale and not args.allow_stale:
                        raise SystemExit(
                            f"\nSTALE CELL: {path}\n"
                            f"  reason: {stale}\n"
                            f"  This cell was produced under different settings. Reusing it\n"
                            f"  would report old numbers as new results.\n"
                            f"  Fix: delete the stale cells (rm {args.out_dir}/*.json)\n"
                            f"       or pass --allow-stale to reuse them deliberately.\n")
                    if stale:
                        print(f"WARN reusing stale {tag} ({stale})", flush=True)
                    # recover the matched budget, else `truncate` silently falls
                    # back to the default instead of the compressor's actual size
                    if cond == "filter_summ" and matched_budget is None:
                        prev = json.load(open(path))
                        if prev:
                            matched_budget = int(np.mean(
                                [r["compressed_context_tokens"] for r in prev]))
                    print(f"skip {tag} (verified same config + sample)", flush=True)
                    continue
                rag.reranker = reranker if rr == "on" else None
                budget = args.truncate_to
                if cond == "truncate" and matched_budget:
                    budget = matched_budget
                    print(f"  [{tag}] matched budget = {budget} tokens", flush=True)

                out, t0 = [], time.perf_counter()
                for ex in tqdm(examples, desc=tag, leave=False):
                    passages = ex.passages(n_distractors=args.n_distractors)
                    if not passages and args.retrieval != "corpus":
                        continue
                    ans, meta = rag.answer(
                        ex.question, passages, dataset=ds, n_shot=args.n_shot,
                        use_summarizer=cfg["summarizer"],
                        use_keyword_filtering=cfg["filtering"],
                        truncate_to=budget if cfg.get("truncate") else 0)
                    rec = {"id": ex.id, "dataset": ds, "method": cond,
                           "reranked": rr == "on", "encoder": args.encoder,
                           "retrieval": args.retrieval, "top_k": args.top_k,
                           "n_shot": args.n_shot, "generator": indexing.model_name,
                           "sample_seed": args.seed,
                           "summary_model": summ if cfg["needs_model"] else None,
                           "question": ex.question, "answers": ex.answers,
                           "question_type": ex.question_type,
                           "supporting_facts": ex.supporting_facts,
                           "rag_answer": ans}
                    rec.update(meta)
                    out.append(rec)
                # recreate the dir defensively: a cell represents up to an hour
                # of compute and must not be lost to a missing directory
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                tmp = path + ".part"
                with open(tmp, "w") as fh:
                    json.dump(out, fh, indent=1)
                os.replace(tmp, path)          # atomic: no half-written cells
                if cond == "filter_summ" and out and matched_budget is None:
                    matched_budget = int(np.mean([r["compressed_context_tokens"] for r in out]))
                print(f"{tag}: {len(out)} recs in {time.perf_counter()-t0:.0f}s", flush=True)
    print("GRID DONE")


if __name__ == "__main__":
    main()
