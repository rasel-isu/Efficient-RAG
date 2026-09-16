"""
evaluate_rag.py
Metric suite for RAG context-compression experiments.

Runs on the CURRENT output schema:
    {"id", "question", "answer", "rag_answer",
     "prompt_token", "completion_token", "total_token"}

and automatically enables extra metrics when these optional fields are present:
    "retrieved_context_tokens"   -> compression ratio
    "compressed_context_tokens"  -> compression ratio (clean numerator)
    "compressed_context"         -> faithfulness (LLM judge)
    "t_*_ms"                     -> latency aggregates
    "question_type"              -> uses stored type instead of heuristic
    "answer" as list             -> max EM/F1 over answer aliases

Core deps:   numpy, scipy, bert_score
Optional:    openai   (only if you enable --llm-judge / faithfulness)

Usage:
    python evaluate_rag.py results.json
    python evaluate_rag.py results.json --no-bertscore           # faster smoke run
    python evaluate_rag.py results.json --llm-judge              # adds LLM correctness
    python evaluate_rag.py results.json --cost-model gpt-4o-mini
    python evaluate_rag.py --compare baseline.json t5small.json  # significance tests
"""

import argparse
import json
import re
import string
from collections import Counter, defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# 1. Core QA metrics (SQuAD-style normalization)
# ---------------------------------------------------------------------------
def normalize_answer(s):
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD convention)."""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _as_list(gold):
    """Allow `answer` to be a single string or a list of acceptable aliases."""
    return gold if isinstance(gold, (list, tuple)) else [gold]


def gold_answers(rec):
    """Read gold answers from either `answers` (list, new schema) or `answer`."""
    return _as_list(rec.get("answers", rec.get("answer")))


def exact_match(pred, gold):
    """1.0 if the prediction exactly matches ANY accepted gold answer, else 0.0."""
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in _as_list(gold)))


def token_f1(pred, gold):
    """Max token-level F1 over accepted gold answers (SQuAD convention)."""
    def f1(a, b):
        a_toks, b_toks = normalize_answer(a).split(), normalize_answer(b).split()
        common = Counter(a_toks) & Counter(b_toks)
        n_same = sum(common.values())
        if n_same == 0:
            return 0.0
        # handle yes/no / empty edge cases
        if not a_toks or not b_toks:
            return float(a_toks == b_toks)
        prec = n_same / len(a_toks)
        rec = n_same / len(b_toks)
        return 2 * prec * rec / (prec + rec)

    return max(f1(pred, g) for g in _as_list(gold))


# ---------------------------------------------------------------------------
# 2. Question typing (matches your Yes/No, When, Which, Who, Where, What, How,
#    Why, Other categories). Yes/No is decided by the gold answer first.
# ---------------------------------------------------------------------------
_WH = ["when", "which", "who", "where", "what", "how", "why"]


def question_type(question, gold):
    golds = [normalize_answer(g) for g in _as_list(gold)]
    if any(g in ("yes", "no") for g in golds):
        return "Yes/No"
    first = normalize_answer(question).split()
    first = first[0] if first else ""
    # leading aux/be verb with no wh-word also implies a yes/no question
    if first in _WH:
        return first.capitalize() if first != "yes/no" else "Yes/No"
    if first in ("was", "were", "is", "are", "did", "do", "does", "has",
                 "have", "had", "can", "could", "will", "would", "should",
                 "am", "was", "be", "been"):
        return "Yes/No"
    return "Other"


# ---------------------------------------------------------------------------
# 3. Cost.  *** VERIFY CURRENT PRICING BEFORE REPORTING ***  (USD per 1M tokens)
#    These are placeholders so the script runs; set the live rates yourself.
# ---------------------------------------------------------------------------
PRICING = {
    # model            : (input_per_1M, output_per_1M)
    "gpt-3.5-turbo":  (0.50, 1.50),
    "gpt-4o-mini":    (0.15, 0.60),
    "custom":         (0.00, 0.00),
}


def record_cost(rec, model):
    pin, pout = PRICING.get(model, PRICING["custom"])
    pt = rec.get("prompt_token", 0)
    ct = rec.get("completion_token", 0)
    return (pt / 1e6) * pin + (ct / 1e6) * pout


# ---------------------------------------------------------------------------
# 4. Compression ratio (only if context token fields are stored)
# ---------------------------------------------------------------------------
def compression_ratio(rec):
    orig = rec.get("retrieved_context_tokens")
    comp = rec.get("compressed_context_tokens")
    if orig and comp and comp > 0:
        return orig / comp
    return None


def supporting_fact_retention(supporting_facts, compressed_context, threshold=0.6):
    """
    For multi-hop datasets (HotpotQA / 2WikiMultihopQA): does the compressor keep
    the gold supporting-fact sentences it needs?

    Each gold sentence gets a token-recall score against the compressed context;
    it counts as "retained" if recall >= threshold. Returns
    (retained_rate, mean_token_recall) or None when inputs are missing.

    This is a dataset-native faithfulness signal and explains multi-hop accuracy
    drops better than a generic LLM-judge: a dropped supporting fact = an answer
    the compressed context can no longer support.
    """
    if not supporting_facts or not compressed_context:
        return None
    ctx = set(normalize_answer(compressed_context).split())
    if not ctx:
        return None
    recalls = []
    for sf in supporting_facts:
        toks = normalize_answer(sf).split()
        if not toks:
            continue
        recalls.append(sum(1 for t in toks if t in ctx) / len(toks))
    if not recalls:
        return None
    retained = sum(1 for r in recalls if r >= threshold) / len(recalls)
    return retained, sum(recalls) / len(recalls)


# ---------------------------------------------------------------------------
# 5. BERTScore (batched over the whole set — do NOT call per example)
# ---------------------------------------------------------------------------
def compute_bertscore(preds, refs):
    """Returns list of per-example F1, or None if bert_score is unavailable."""
    try:
        from bert_score import score
    except ImportError:
        print("[warn] bert_score not installed; skipping BERTScore "
              "(pip install bert_score)")
        return None
    # refs here are the *first* gold alias per example; for multi-alias datasets
    # you can score against each alias and take the max if you want stricter eval.
    _, _, f1 = score(preds, refs, lang="en", verbose=False)
    return f1.tolist()


# ---------------------------------------------------------------------------
# 6. Optional LLM judge (correctness + faithfulness). Uses OpenAI; you already
#    have OPENAI_API_KEY. Falls back silently if unavailable.
# ---------------------------------------------------------------------------
class LocalJudge:
    """Offline correctness judge.

    The scaling paper grades with an LLM (DeepSeek V3) rather than exact match,
    because EM systematically penalises abstractive compression for phrasing
    rather than for content. Reporting EM alone therefore understates any
    abstractive method. This runs a local instruct model so judging needs no API
    key; pass --judge-model to swap it.
    """

    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto").eval()
        self.torch = torch

    def __call__(self, question, gold, pred):
        golds = "; ".join(_as_list(gold))
        msg = [{"role": "user", "content": (
            "You are grading a QA system. Reply with exactly one word: "
            "CORRECT or INCORRECT.\n"
            f"Question: {question}\nReference answer(s): {golds}\n"
            f"System answer: {pred}\n"
            "Is the system answer semantically equivalent to a reference?")}]
        ids = self.tok.apply_chat_template(msg, add_generation_prompt=True,
                                           return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=5, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        txt = self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).upper()
        return float("CORRECT" in txt and "INCORRECT" not in txt)


def _openai_client():
    try:
        from openai import OpenAI
        return OpenAI()
    except Exception as e:
        print(f"[warn] OpenAI client unavailable ({e}); skipping LLM judge")
        return None


def llm_judge_correct(client, question, gold, pred, model="gpt-4o-mini"):
    golds = "; ".join(_as_list(gold))
    prompt = (
        "You are grading a QA system. Reply with exactly one word: "
        "CORRECT or INCORRECT.\n"
        f"Question: {question}\n"
        f"Reference answer(s): {golds}\n"
        f"System answer: {pred}\n"
        "Is the system answer correct (semantically equivalent to a reference)?"
    )
    r = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return float("CORRECT" in r.choices[0].message.content.upper())


def llm_judge_faithful(client, question, context, pred, model="gpt-4o-mini"):
    prompt = (
        "Reply with exactly one word: SUPPORTED or UNSUPPORTED.\n"
        "Given ONLY the context below, is the answer fully supported by it "
        "(no facts beyond the context)?\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer: {pred}\n"
    )
    r = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return float("SUPPORTED" in r.choices[0].message.content.upper()
                 and "UNSUPPORTED" not in r.choices[0].message.content.upper())


# ---------------------------------------------------------------------------
# 7. Bootstrap confidence interval
# ---------------------------------------------------------------------------
def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# 8. Main single-file evaluation
# ---------------------------------------------------------------------------
def evaluate(records, cost_model="gpt-3.5-turbo", do_bertscore=True,
             llm_judge=False, faithfulness=False,
             judge_backend="local", judge_model=None):
    preds = [r["rag_answer"] for r in records]
    refs = [gold_answers(r)[0] for r in records]

    ems, f1s, costs, ratios = [], [], [], []
    retained_rates, recall_means = [], []
    by_type = defaultdict(lambda: {"em": [], "f1": [], "n": 0})

    for r in records:
        gold = gold_answers(r)
        em = exact_match(r["rag_answer"], gold)
        f1 = token_f1(r["rag_answer"], gold)
        qt = r.get("question_type") or question_type(r["question"], gold)
        ems.append(em)
        f1s.append(f1)
        costs.append(record_cost(r, cost_model))
        cr = compression_ratio(r)
        if cr is not None:
            ratios.append(cr)
        sfr = supporting_fact_retention(r.get("supporting_facts"),
                                        r.get("compressed_context"))
        if sfr is not None:
            retained_rates.append(sfr[0])
            recall_means.append(sfr[1])
        by_type[qt]["em"].append(em)
        by_type[qt]["f1"].append(f1)
        by_type[qt]["n"] += 1
        r["_em"], r["_f1"], r["_type"] = em, f1, qt  # keep for compare()

    report = {"n": len(records)}

    em_mean, em_lo, em_hi = bootstrap_ci(ems)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(f1s)
    report["exact_match"] = {"mean": em_mean, "ci95": [em_lo, em_hi]}
    report["token_f1"] = {"mean": f1_mean, "ci95": [f1_lo, f1_hi]}

    # tokens & cost
    report["avg_prompt_tokens"] = float(np.mean([r.get("prompt_token", 0) for r in records]))
    report["avg_completion_tokens"] = float(np.mean([r.get("completion_token", 0) for r in records]))
    report["total_tokens"] = int(np.sum([r.get("total_token", 0) for r in records]))
    report["total_cost_usd"] = float(np.sum(costs))
    report["cost_model"] = cost_model

    # compression ratio (only if stored)
    if ratios:
        cr_mean, cr_lo, cr_hi = bootstrap_ci(ratios)
        report["compression_ratio"] = {"mean": cr_mean, "ci95": [cr_lo, cr_hi]}
    else:
        report["compression_ratio"] = "N/A — store retrieved_context_tokens & compressed_context_tokens"

    # supporting-fact retention (multi-hop only; needs supporting_facts + compressed_context)
    if retained_rates:
        report["supporting_fact_retention"] = {
            "retained_rate": float(np.mean(retained_rates)),
            "mean_token_recall": float(np.mean(recall_means)),
            "n_questions": len(retained_rates),
        }
    else:
        report["supporting_fact_retention"] = "N/A — multi-hop only; store supporting_facts & compressed_context"

    # latency (only if stored)
    lat_keys = [k for k in ("t_retrieval_ms", "t_filter_ms", "t_summarize_ms",
                            "t_generate_ms", "t_end_to_end_ms")
                if any(k in r for r in records)]
    if lat_keys:
        report["latency_ms"] = {k: float(np.mean([r[k] for r in records if k in r]))
                                for k in lat_keys}
    else:
        report["latency_ms"] = "N/A — store per-stage t_*_ms fields (Task 1)"

    # BERTScore
    if do_bertscore:
        bsc = compute_bertscore(preds, refs)
        if bsc is not None:
            b_mean, b_lo, b_hi = bootstrap_ci(bsc)
            report["bertscore_f1"] = {"mean": b_mean, "ci95": [b_lo, b_hi]}
            for r, b in zip(records, bsc):
                r["_bertscore"] = b

    # LLM judge correctness
    if llm_judge:
        if judge_backend == "local":
            judge = LocalJudge(judge_model) if judge_model else LocalJudge()
            jc = [judge(r["question"], gold_answers(r), r["rag_answer"]) for r in records]
            report["llm_judge_correct"] = {"mean": float(np.mean(jc)),
                                           "backend": "local", "model": judge_model}
        else:
            client = _openai_client()
            if client:
                jc = [llm_judge_correct(client, r["question"], gold_answers(r),
                                        r["rag_answer"], model=judge_model or "gpt-4o-mini")
                      for r in records]
                report["llm_judge_correct"] = {"mean": float(np.mean(jc)),
                                               "backend": "openai", "model": judge_model}

    # Faithfulness (needs compressed_context)
    if faithfulness and any("compressed_context" in r for r in records):
        client = _openai_client()
        if client:
            fa = [llm_judge_faithful(client, r["question"],
                                     r.get("compressed_context", ""), r["rag_answer"])
                  for r in records if "compressed_context" in r]
            report["faithfulness"] = {"mean": float(np.mean(fa)), "n_judged": len(fa)}
    elif faithfulness:
        report["faithfulness"] = "N/A — store compressed_context to enable"

    # per-question-type breakdown
    report["by_question_type"] = {
        qt: {"n": d["n"], "em": float(np.mean(d["em"])), "f1": float(np.mean(d["f1"]))}
        for qt, d in sorted(by_type.items(), key=lambda x: -x[1]["n"])
    }
    return report


# ---------------------------------------------------------------------------
# 9. Significance tests between two systems (run on the SAME examples)
# ---------------------------------------------------------------------------
def mcnemar_em(records_a, records_b):
    """McNemar's test on exact-match correctness. Records aligned by id."""
    from scipy.stats import chi2
    a = {r["id"]: exact_match(r["rag_answer"], gold_answers(r)) for r in records_a}
    b = {r["id"]: exact_match(r["rag_answer"], gold_answers(r)) for r in records_b}
    ids = set(a) & set(b)
    b01 = sum(1 for i in ids if a[i] == 1 and b[i] == 0)  # A right, B wrong
    b10 = sum(1 for i in ids if a[i] == 0 and b[i] == 1)  # A wrong, B right
    if b01 + b10 == 0:
        return {"b_a_right": b01, "b_b_right": b10, "p_value": 1.0}
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)        # continuity-corrected
    return {"a_only_right": b01, "b_only_right": b10,
            "statistic": float(stat), "p_value": float(chi2.sf(stat, df=1))}


def paired_bootstrap_f1(records_a, records_b, n_boot=10000, seed=0):
    """Paired bootstrap on the F1 difference (A - B). Records aligned by id."""
    rng = np.random.default_rng(seed)
    a = {r["id"]: token_f1(r["rag_answer"], gold_answers(r)) for r in records_a}
    b = {r["id"]: token_f1(r["rag_answer"], gold_answers(r)) for r in records_b}
    ids = sorted(set(a) & set(b))
    diffs = np.array([a[i] - b[i] for i in ids])
    n = len(diffs)
    boot = diffs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided p: fraction of bootstrap means on the opposite side of 0
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"mean_f1_diff": float(diffs.mean()),
            "ci95": [float(lo), float(hi)], "p_value": float(min(p, 1.0))}


# ---------------------------------------------------------------------------
def _load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", help="path to a results JSON file")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="two result files to run significance tests on")
    ap.add_argument("--cost-model", default="gpt-3.5-turbo")
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--llm-judge", action="store_true")
    ap.add_argument("--faithfulness", action="store_true")
    ap.add_argument("--judge-backend", default="local", choices=["local", "openai"],
                    help="local needs no API key; openai needs OPENAI_API_KEY")
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()

    if args.compare:
        a, b = _load(args.compare[0]), _load(args.compare[1])
        print("== Significance (A vs B) ==")
        print("EM (McNemar):     ", json.dumps(mcnemar_em(a, b), indent=2))
        print("F1 (paired boot): ", json.dumps(paired_bootstrap_f1(a, b), indent=2))
        return

    records = _load(args.results)
    report = evaluate(records, cost_model=args.cost_model,
                      do_bertscore=not args.no_bertscore,
                      llm_judge=args.llm_judge, faithfulness=args.faithfulness,
                      judge_backend=args.judge_backend, judge_model=args.judge_model)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()