import argparse
import json
import os
from uuid import uuid1

from tqdm import tqdm
import pandas as pd

from rag.indexing import Indexing
from rag.retrieval import Retriever


# Ablation conditions. `summarizer` / `filtering` are what actually get passed
# down to the retriever - previously these were hardcoded at the call site, so
# every run produced the same condition regardless of its filename.
CONDITIONS = {
    "baseline":        dict(filtering=False, summarizer=False, needs_model=False),
    "keyword_only":    dict(filtering=True,  summarizer=False, needs_model=False),
    "summarizer_only": dict(filtering=False, summarizer=True,  needs_model=True),
    "filter_summ":     dict(filtering=True,  summarizer=True,  needs_model=True),
    # Matched-budget control: keep the first N tokens of each retrieved passage
    # so the generator sees the same *number* of tokens as a compressed run but
    # none of the query-aware selection. If a compressor cannot beat this, the
    # compression step is not what is buying the accuracy.
    "truncate":        dict(filtering=False, summarizer=False, needs_model=False,
                            truncate=True),
}

HUMAN_PROMPT = ("If the correct answer is yes or no, reply ONLY 'yes' or 'no'. "
                "Otherwise, reply with the exact answer and no additional text")


def get_ans_from_rag_for_rag_mini_wikipedia(outfile, summary_model=None,
                                            condition="filter_summ", n=None,
                                            prompt_style="qa_default", use_chat=False, reader=None):
    cfg = CONDITIONS[condition]
    if cfg["needs_model"] and summary_model is None:
        raise ValueError(f"condition '{condition}' needs --summary-model")

    data = pd.read_csv('DATASET/rag-mini-wikipedia/test.csv')
    if n:
        data = data.iloc[:n]
    data = data.reset_index(drop=True)

    texts = pd.read_csv('DATASET/rag-mini-wikipedia/passages.csv')['passage'].to_list()
    indexing = Indexing(texts, reader=reader)
    index, nodes = indexing.get_index()
    retriver = Retriever(index, nodes, indexing.token_counter,
                         summary_model if cfg["needs_model"] else None,
                         tokenizer=indexing.tokenizer)

    qan = []
    for i in tqdm(range(len(data)), desc=condition):
        question = data.loc[i]['question']
        answer = data.loc[i]['answer']
        d = {
            'id': str(uuid1()),
            'dataset': 'rag_mini',
            'method': condition,
            'prompt_style': prompt_style, 'use_chat': use_chat,
            'summary_model': summary_model if cfg["needs_model"] else None,
            'generator': indexing.model_name,
            'question': question,
            'answers': [answer],                     # LIST, even for single answer
            'question_type': None,                   # native label if dataset has one
            'supporting_facts': [],                  # populated for multi-hop datasets
        }
        rag_answer, token_count = retriver.get_token_effi_response(
            question, human_pormpt=HUMAN_PROMPT,
            use_summarizer=cfg["summarizer"],
            use_keyword_filtering=cfg["filtering"],
            prompt_style=prompt_style, use_chat=use_chat,
        )
        d['rag_answer'] = rag_answer
        d.update(token_count)
        qan.append(d)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as f:
        json.dump(qan, f, indent=1)
    print(f"wrote {outfile}  ({len(qan)} records)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="filter_summ", choices=list(CONDITIONS))
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--summary-model", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--reader", default=None,
                    help="generator model id; default $RAG_READER or Llama-3.2-3B-Instruct")
    ap.add_argument("--prompt-style", default="qa_default",
                    choices=["instruction_first", "instruction_last", "qa_default"])
    ap.add_argument("--chat", action="store_true", help="use the chat template (default: raw completion, which scores higher on small readers)")
    ap.add_argument("--no-chat", action="store_true",
                    help="send the raw prompt instead of the chat template")
    args = ap.parse_args()
    get_ans_from_rag_for_rag_mini_wikipedia(
        outfile=args.outfile, summary_model=args.summary_model,
        condition=args.condition, n=args.n,
        prompt_style=args.prompt_style, use_chat=args.chat, reader=args.reader,
    )
