"""
Run one (dataset x compression x retrieval-noise) cell of the experiment grid.

The retrieval-noise axis is the point of this script: `--n-distractors` controls
how much irrelevant evidence reaches the compressor, from 0 (oracle retrieval)
to the dataset's full distractor pool (the noisy regime prior compression work
evaluates in).
"""

import argparse, json, os
from tqdm import tqdm

from rag.indexing import Indexing
from rag.datasets import LOADERS, inject_cross_question_distractors
from rag.local_retrieval import LocalRAG
from main import CONDITIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--condition", default="baseline", choices=list(CONDITIONS))
    ap.add_argument("--summary-model", default=None)
    ap.add_argument("--n", type=int, default=500, help="number of questions")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed")
    ap.add_argument("--n-distractors", type=int, default=None,
                    help="distractors per question; None = all available")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-shot", type=int, default=5,
                    help="in-context examples (RECOMP uses 5); 0 disables")
    ap.add_argument("--no-rerank", action="store_true",
                    help="disable the cross-encoder (baselines do NOT rerank)")
    ap.add_argument("--encoder", default="contriever", choices=["contriever", "bge"])
    ap.add_argument("--retrieval", default="pool", choices=["pool", "corpus"],
                    help="'pool' = per-question candidates; 'corpus' = DPR Wikipedia")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--reader", default=None,
                    help="generator model id; default $RAG_READER or Llama-3.2-3B-Instruct")
    ap.add_argument("--truncate-to", type=int, default=0,
                    help="total context token budget for the 'truncate' condition")
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    cfg = CONDITIONS[args.condition]
    if cfg["needs_model"] and not args.summary_model:
        raise SystemExit(f"condition '{args.condition}' needs --summary-model")

    examples = LOADERS[args.dataset](n=args.n, seed=args.seed)
    if args.dataset == "triviaqa":
        examples = inject_cross_question_distractors(examples, n=8)
    # configures Settings.llm / embed model / token counter without building a
    # Weaviate index (that only happens in get_index())
    indexing = Indexing([], reader=args.reader)
    rag_kw = dict(tokenizer=indexing.tokenizer, top_k=args.top_k,
                  use_reranker=not args.no_rerank)
    summ = args.summary_model if cfg["needs_model"] else None
    # NQ has no per-question candidate pool, so it always retrieves from the corpus
    use_corpus = args.retrieval == "corpus" or args.dataset == "nq_open"
    if use_corpus:
        from rag.global_retrieval import GlobalRAG
        index_dir = args.index_dir or f"DATASET/wiki_dpr/index_{args.encoder}"
        if not os.path.exists(f"{index_dir}/embeddings.fp16.npy"):
            raise SystemExit(f"missing index {index_dir} - run scripts/build_wiki_index.py")
        rag = GlobalRAG(indexing.token_counter, summ,
                        index_dir=index_dir, encoder=args.encoder, **rag_kw)
    else:
        rag = LocalRAG(indexing.token_counter, summ, **rag_kw)

    out = []
    desc = f"{args.dataset}/{args.condition}/d={args.n_distractors}"
    for ex in tqdm(examples, desc=desc):
        passages = ex.passages(n_distractors=args.n_distractors)
        if not passages and not use_corpus:
            continue
        ans, meta = rag.answer(
            ex.question, passages,
            dataset=args.dataset, n_shot=args.n_shot,
            use_summarizer=cfg["summarizer"],
            use_keyword_filtering=cfg["filtering"],
            truncate_to=args.truncate_to if cfg.get("truncate") else 0)
        rec = {
            "id": ex.id, "dataset": ex.dataset, "method": args.condition,
            "summary_model": args.summary_model if cfg["needs_model"] else None,
            "generator": indexing.model_name, "question": ex.question,
            "answers": ex.answers, "question_type": ex.question_type,
            "supporting_facts": ex.supporting_facts,
            "n_distractors": args.n_distractors, "top_k": args.top_k,
            "n_shot": args.n_shot, "encoder": args.encoder,
            "retrieval": "corpus" if use_corpus else "pool",
            "reranked": not args.no_rerank,
            "truncate_to": args.truncate_to if cfg.get("truncate") else 0,
            "n_gold_available": len(ex.gold),
            "rag_answer": ans,
        }
        rec.update(meta)
        out.append(rec)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.outfile} ({len(out)} records)")


if __name__ == "__main__":
    main()
