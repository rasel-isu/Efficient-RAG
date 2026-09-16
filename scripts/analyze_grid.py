"""
Markdown report for a full (dataset x condition x rerank) grid.

Leads with token cost, because this project's contribution is efficiency: fewer
tokens at acceptable accuracy loss. Accuracy appears as RETENTION against the
uncompressed baseline rather than as a standalone score, and compression ratios
are put beside the published numbers this work competes with.
"""
import argparse, glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import (exact_match, token_f1, gold_answers, bootstrap_ci,
                        supporting_fact_retention)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from published import (COMPACT, COMPACT_COMP, COMPACT_ORDER, COMPACT_READER,  # noqa: E402
                       COMPACT_SOURCE, COMPACT_UNCOMPRESSED, RECOMP_PAPER,
                       RECOMP_READER, RECOMP_SOURCE, RECOMP_UNCOMPRESSED)

OUR_READER = "Llama-3.2-3B"
# CompAct's grid has no reranking, so the rerank-off cell is the matched setting
# and is what the published rows are compared against.
PUB_MATCH_RERANK = "off"

DS_LABEL = {"triviaqa": "TriviaQA", "nq_open": "NQ-Open", "hotpotqa": "HotpotQA",
            "2wiki": "2WikiMultihopQA", "musique": "MuSiQue"}
ORDER = ["triviaqa", "nq_open", "hotpotqa", "2wiki", "musique"]
# Every condition this report knows how to render. Only the ones present in the
# grid become columns (see `COND` in main), so unrun ablations do not widen the
# per-condition tables - but Table 2 always lists them, as explicit gaps.
COND_ALL = ["baseline", "truncate", "keyword_only", "summarizer_only", "filter_summ"]
COND_LABEL = {"baseline": "Uncompressed", "truncate": "Truncate (matched budget)",
              "keyword_only": "Keyword filter only", "summarizer_only": "FLAN-T5 summarizer only",
              "filter_summ": "Keyword filter + FLAN-T5"}
# published compression rates, for context on where this method sits. Derived
# from the shared table rather than restated, so this section and Table 1 can
# never drift apart.
PUBLISHED_COMP = {k: v for k, v in COMPACT_COMP["hotpotqa"].items()
                  if k not in (COMPACT_UNCOMPRESSED, "Oracle")}
EM_FLOOR_CORRECT = 3
MAIN_METHOD = "filter_summ"     # the proposed method; variants are compared to it
BERT_CACHE = "OUTPUT/.bertscore_cache.json"

# Table 1: what the proposed method is measured against, in evaluation order.
BASELINE_ROWS = ["baseline", "truncate"]
# Table 2: every configuration of the proposed pipeline. Rows whose cell is
# absent render as gaps rather than disappearing, so an ablation that has not
# been run yet is visible as missing rather than silently omitted.
VARIANT_ROWS = [("filter_summ", "on"), ("filter_summ", "off"),
                ("keyword_only", "on"), ("keyword_only", "off"),
                ("summarizer_only", "on"), ("summarizer_only", "off")]
REF_VARIANT = ("filter_summ", "on")   # the full proposed system

# columns shared by both comparison tables
PUB_KEYS = ["em", "f1", "ret"]   # Table 1: BERTScore is not published, so it is dropped
CMP_KEYS = ["em", "f1", "bert", "red", "ret"]
CMP_HEAD = {"em": "EM", "f1": "F1", "bert": "BERTScore",
            "red": "Token redn", "ret": "Retention"}
CMP_DHEAD = {"em": "ΔEM", "f1": "ΔF1", "bert": "ΔBERT",
             "red": "ΔRedn", "ret": "ΔRet"}
CMP_FMT = {"em": "{:.1f}", "f1": "{:.1f}", "bert": "{:.2f}",
           "red": "{:.1f}%", "ret": "{:.0f}%"}
CMP_DFMT = {"em": "{:+.1f}", "f1": "{:+.1f}", "bert": "{:+.2f}",
            "red": "{:+.1f}", "ret": "{:+.0f}"}


def cmp_metrics(v, base, ds, floored):
    """The five comparison-table metrics for one cell, or None if it is absent.

    Reduction and retention are both defined against the uncompressed cell of
    the SAME rerank setting, so a variant is never credited with a saving that
    came from a different retrieval configuration.
    """
    if not v:
        return None
    nan = float("nan")
    red = (1 - v["ctx"] / base["ctx"]) * 100 if base and base["ctx"] else nan
    ret = v["em"] / base["em"] * 100 if base and base["em"] else nan
    return {"em": nan if ds in floored else v["em"], "f1": v["f1"],
            "bert": v["bert"], "red": red, "ret": nan if ds in floored else ret}


def cmp_cells(m, ref, keys=None):
    """Value/Δ pairs for one row. Δ is ref minus this row, so positive always
    means the proposed method is ahead - including on reduction, where more
    compression is better."""
    out = []
    for k in keys or CMP_KEYS:
        if m is None or m.get(k) is None or m[k] != m[k]:
            out += ["—", "—"]
            continue
        out.append(CMP_FMT[k].format(m[k]))
        if ref is None or ref is m or ref.get(k) is None or ref[k] != ref[k]:
            out.append("—")
            continue
        d = CMP_DFMT[k].format(ref[k] - m[k])
        out.append("+" + d[1:] if float(d) == 0 else d)  # never print "-0.0"
    return out


def cmp_header(w, lead, keys=None):
    """`lead` is the fixed leading column header(s), pipe-separated."""
    keys = keys or CMP_KEYS
    w(f"| {lead} | " + " | ".join(f"{CMP_HEAD[k]} | {CMP_DHEAD[k]}" for k in keys) + " |")
    w("|" + "---|" * (lead.count("|") + 1 + 2 * len(keys)))


def _bert_cache():
    try:
        return json.load(open(BERT_CACHE))
    except Exception:
        return {}


def bertscore_for(path, preds, refs, cache, enabled):
    """Cached mean BERTScore-F1 for one cell. Keyed by file path + row count so a
    re-run with different results recomputes instead of reusing a stale value."""
    key = f"{os.path.basename(path)}:{len(preds)}"
    if key in cache:
        return cache[key]
    if not enabled:
        return float("nan")
    try:
        from bert_score import score
    except ImportError:
        return float("nan")
    _, _, f1 = score(preds, refs, lang="en", verbose=False, batch_size=64)
    val = float(f1.mean()) * 100
    cache[key] = val
    return val


def mcnemar_p(a_scores, b_scores):
    """McNemar on paired exact-match correctness (continuity corrected)."""
    from scipy.stats import chi2
    b01 = sum(1 for x, y in zip(a_scores, b_scores) if x == 1 and y == 0)
    b10 = sum(1 for x, y in zip(a_scores, b_scores) if x == 0 and y == 1)
    if b01 + b10 == 0:
        return 1.0
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    return float(chi2.sf(stat, df=1))


def load(path, cache=None, bert=False):
    d = json.load(open(path))
    if not d:
        return None
    ems = [exact_match(r["rag_answer"], gold_answers(r)) for r in d]
    f1s = [token_f1(r["rag_answer"], gold_answers(r)) for r in d]
    em, lo, hi = bootstrap_ci(ems)
    sfr = [supporting_fact_retention(r.get("supporting_facts"), r.get("compressed_context"))
           for r in d]
    sfr = [x[0] for x in sfr if x]
    f1m, f1lo, f1hi = bootstrap_ci(f1s)
    bs = bertscore_for(path, [r["rag_answer"] for r in d],
                       [gold_answers(r)[0] for r in d], cache if cache is not None else {}, bert)
    return dict(n=len(d), em=em * 100, em_ci=(lo * 100, hi * 100),
                f1=f1m * 100, f1_ci=(f1lo * 100, f1hi * 100), bert=bs,
                ems=ems, f1s=f1s,
                n_correct=int(round(em * len(d))),
                prompt=float(np.mean([r["prompt_token"] for r in d])),
                compl=float(np.mean([r["completion_token"] for r in d])),
                ctx=float(np.mean([r.get("compressed_context_tokens", 0) for r in d])),
                raw=float(np.mean([r.get("retrieved_context_tokens", 0) for r in d])),
                t_e2e=float(np.mean([r.get("t_end_to_end_ms", 0) for r in d])),
                sfr=float(np.mean(sfr)) * 100 if sfr else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="OUTPUT/full/*.json")
    ap.add_argument("--out", default="RESULTS_full_grid.md")
    ap.add_argument("--bertscore", action="store_true",
                    help="compute BERTScore (slow: ~15k pairs; cached afterwards)")
    args = ap.parse_args()
    cache = _bert_cache()

    cells = {}
    for p in glob.glob(args.glob):
        m = re.match(r"(.+)_(baseline|truncate|filter_summ)_rerank-(on|off)\.json$",
                     os.path.basename(p))
        if not m:
            continue
        got = load(p, cache=cache, bert=args.bertscore)
        if got:
            cells[(m.group(1), m.group(2), m.group(3))] = got
    if not cells:
        raise SystemExit(f"no grid cells matched {args.glob}")

    COND = [c for c in COND_ALL if any(k[1] == c for k in cells)]
    datasets = [d for d in ORDER if any(k[0] == d for k in cells)]
    reranks = [r for r in ["off", "on"] if any(k[2] == r for k in cells)]
    floored = [d for d in datasets
               if max((v["n_correct"] for k, v in cells.items() if k[0] == d), default=0)
               < EM_FLOOR_CORRECT]

    L = []
    w = L.append
    w("# Full Grid — Token Efficiency Analysis\n")
    seeds = set()
    for pth in glob.glob(args.glob):
        try:
            seeds.add(json.load(open(pth))[0].get("sample_seed", "unset"))
        except Exception:
            pass
    w(f"Sampling: **{sorted(str(x) for x in seeds)[0] if len(seeds)==1 else 'MIXED - ' + str(sorted(map(str,seeds)))}**"
      " random seed, identical question set in every cell.\n"
      if seeds else "")
    w("Contribution metric is **token cost**: how far the prompt can be compressed "
      "before accuracy becomes unacceptable. Accuracy is reported as retention "
      "against the uncompressed baseline in the same retrieval setting.\n")
    if floored:
        w(f"> **Note:** EM is at the floor for {', '.join(DS_LABEL[d] for d in floored)} "
          f"(<{EM_FLOOR_CORRECT} correct in every condition); read F1 and "
          f"supporting-fact retention there instead.\n")

    # --- how to read -------------------------------------------------------
    w("## How to read this report\n")
    w("| Term | Meaning |")
    w("|---|---|")
    w(f"| **Ours** | Keyword filter + FLAN-T5 compression of the retrieved context, "
      f"{OUR_READER} reader |")
    w("| Comp. | Compression ratio: context tokens before ÷ after |")
    w("| Token reduction | Share of context tokens removed; higher is better |")
    w("| EM retention | Exact match as a share of the **same system's own** "
      "uncompressed run. In Table 1 that is each paper's own no-compression row, "
      "which is what makes the numbers readable across readers of different sizes; "
      "elsewhere it is our uncompressed cell in the same rerank setting |")
    w("| Δ | **Ours minus the row**, so a positive Δ means the proposed method wins. "
      "Δ on reduction and retention is in percentage points |")
    w("| rerank | bge cross-encoder reranking of the retrieved passages, on or off |\n")
    w("**Table 1** compares the method against published compressors, **Table 2** "
      "against its own ablations, **Table 3** against in-pipeline controls "
      "(no compression, and truncation at a matched token budget). Everything after "
      "them is supporting detail: confidence intervals, significance tests, "
      "supporting-fact retention and latency.\n")

    # --- Table 1: vs published compressors ----------------------------------
    w("## Table 1 — Ours vs published compressors\n")
    w(f"Published rows are transcribed from {COMPACT_SOURCE}. That grid shares our "
      f"retriever, corpus and depth, but uses a larger reader ({COMPACT_READER} vs "
      f"{OUR_READER}) and no reranking - so our rerank-{PUB_MATCH_RERANK} cell is "
      "the matched setting and is the reference row. Δ is *ours minus the row*, so "
      "a positive Δ means our method is ahead.\n")
    w(f"Because the readers differ, raw EM is not a fair column. **EM retention** "
      f"is: each system's EM as a share of its own uncompressed row "
      f"(`{COMPACT_UNCOMPRESSED}` for the published systems, our uncompressed cell "
      "for ours). It measures what the compressor costs its own reader, which is "
      "what a compression paper is claiming.\n")

    def _pub_row(em, f1, unc_em):
        """One Table-1 row, with retention against that system's own ceiling."""
        return {"em": em, "f1": f1,
                "ret": em / unc_em * 100 if unc_em else float("nan")}

    rr = PUB_MATCH_RERANK if PUB_MATCH_RERANK in reranks else reranks[0]
    other = "on" if rr == "off" else "off"
    detail, summary = [], {}

    def _acc(name, reader, comp, m):
        d = summary.setdefault(name, {"reader": reader, "comp": [], "em": [], "ret": []})
        if comp == comp:
            d["comp"].append(comp)
        d["em"].append(m["em"])
        d["ret"].append(m["ret"])

    for ds in datasets:
        if ds not in COMPACT:
            continue
        base, main = cells.get((ds, "baseline", rr)), cells.get((ds, MAIN_METHOD, rr))
        if not (base and main):
            continue
        detail.append(f"**{DS_LABEL[ds]}**\n")
        hdr = []
        cmp_header(hdr.append, "System | Reader | Comp.", PUB_KEYS)
        detail += hdr

        ref = _pub_row(main["em"], main["f1"], base["em"])
        ratio = base["ctx"] / main["ctx"] if main["ctx"] else float("nan")
        ours_label = f"Ours: {COND_LABEL[MAIN_METHOD]}"
        detail.append(f"| **{ours_label}** | {OUR_READER} | {ratio:.1f}× | "
                      + " | ".join(cmp_cells(ref, ref, PUB_KEYS)) + " |")
        _acc(ours_label, OUR_READER, ratio, ref)

        v, b2 = cells.get((ds, MAIN_METHOD, other)), cells.get((ds, "baseline", other))
        if v and b2 and v["ctx"]:
            m = _pub_row(v["em"], v["f1"], b2["em"])
            lbl = f"Ours (rerank {other})"
            detail.append(f"| {lbl} | {OUR_READER} | {b2['ctx']/v['ctx']:.1f}× | "
                          + " | ".join(cmp_cells(m, ref, PUB_KEYS)) + " |")
            _acc(lbl, OUR_READER, b2["ctx"] / v["ctx"], m)

        m = _pub_row(base["em"], base["f1"], base["em"])
        detail.append(f"| Ours, uncompressed | {OUR_READER} | 1.0× | "
                      + " | ".join(cmp_cells(m, ref, PUB_KEYS)) + " |")
        _acc("Ours, uncompressed", OUR_READER, 1.0, m)

        unc = COMPACT[ds].get(COMPACT_UNCOMPRESSED, (float("nan"),))[0]
        for name in COMPACT_ORDER:
            if name not in COMPACT[ds]:
                continue
            e, f = COMPACT[ds][name]
            c = COMPACT_COMP.get(ds, {}).get(name, float("nan"))
            m = _pub_row(e, f, unc)
            detail.append(f"| {name} | {COMPACT_READER} | "
                          + (f"{c:.1f}× | " if c == c else "— | ")
                          + " | ".join(cmp_cells(m, ref, PUB_KEYS)) + " |")
            _acc(name, COMPACT_READER, c, m)
        detail.append("")

    if summary:
        w("**Summary — mean over the datasets each system reports**\n")
        w("| System | Reader | Mean comp. | Mean EM | Mean EM retention | n datasets |")
        w("|---|---|---|---|---|---|")
        for name, d in summary.items():
            w(f"| {'**' + name + '**' if name.startswith('Ours:') else name} "
              f"| {d['reader']} | {np.mean(d['comp']):.1f}× | {np.mean(d['em']):.1f} "
              f"| {np.mean(d['ret']):.0f}% | {len(d['em'])} |")
        w("")
        w("Oracle is the gold-supporting-documents upper bound, not a deployable "
          "system. Mean EM mixes datasets of very different difficulty and is only "
          "comparable within a reader; mean retention is comparable across readers.\n")
    for line in detail:
        w(line)

    w(f"### Closest published analogue — off-the-shelf summarisation\n")
    w(f"{RECOMP_SOURCE}. A different reader *and* depth, so this is positioning "
      "rather than a like-for-like comparison. The `T5 (off-the-shelf)` row is an "
      "untrained summariser applied to retrieved passages - the same recipe as "
      "ours - and is the number this method has to be read against.\n")
    w("| Dataset | System | Reader | EM | F1 | EM retention |")
    w("|---|---|---|---|---|---|")
    for ds in datasets:
        if ds not in RECOMP_PAPER:
            continue
        base, main = cells.get((ds, "baseline", rr)), cells.get((ds, MAIN_METHOD, rr))
        unc = RECOMP_PAPER[ds].get(RECOMP_UNCOMPRESSED, (float("nan"),))[0]
        if base and main:
            r = main["em"] / base["em"] * 100 if base["em"] else float("nan")
            w(f"| {DS_LABEL[ds]} | **Ours: {COND_LABEL[MAIN_METHOD]}** | {OUR_READER} "
              f"| {main['em']:.1f} | {main['f1']:.1f} | {r:.0f}% |")
        for name, (e, f) in RECOMP_PAPER[ds].items():
            r = e / unc * 100 if unc == unc and unc else float("nan")
            w(f"| {DS_LABEL[ds]} | {name} | {RECOMP_READER} | {e:.1f} | {f:.1f} "
              f"| {r:.0f}% |")
    w("")

    # --- Table 2: vs our own variants ---------------------------------------
    w("## Table 2 — Ours vs our variants\n")
    w(f"Ablations of the proposed pipeline. The reference row is the full system "
      f"({COND_LABEL[REF_VARIANT[0]]}, rerank {REF_VARIANT[1]}); every Δ is that row "
      f"minus the variant.\n")
    missing = sorted({COND_LABEL[c] for c, _ in VARIANT_ROWS
                      if not any(k[1] == c for k in cells)})
    cmp_header(w, "Dataset | System")
    for ds in datasets:
        for c, rr in VARIANT_ROWS:
            base = cells.get((ds, "baseline", rr))
            ref = cmp_metrics(cells.get((ds,) + REF_VARIANT),
                              cells.get((ds, "baseline", REF_VARIANT[1])), ds, floored)
            m = cmp_metrics(cells.get((ds, c, rr)), base, ds, floored)
            label = f"{COND_LABEL[c]} (rerank {rr})"
            if (c, rr) == REF_VARIANT:
                label = f"**Ours: {COND_LABEL[c]}** (rerank {rr}, reference)"
                m = ref
            elif m is None:
                label += " _(not run)_"
            w(f"| {DS_LABEL[ds]} | {label} | " + " | ".join(cmp_cells(m, ref)) + " |")
    w("")
    if missing:
        w("> **Gap:** no cells in this grid for " + " or ".join(missing) + ". Run them with "
          "`scripts/run_grid.py --conditions " +
          " ".join(sorted({c for c, _ in VARIANT_ROWS if not any(k[1] == c for k in cells)})) +
          "` and regenerate; the rows above fill in automatically.\n")

    # --- Table 3: in-pipeline controls ---------------------------------------
    w("## Table 3 — Ours vs in-pipeline controls\n")
    w("Paired on identical questions. The uncompressed cell is the accuracy ceiling; "
      "truncation is the matched-budget control that spends the same tokens without "
      "query-aware selection.\n")
    for rr in reranks:
        w(f"**Retrieval: rerank {rr}**\n")
        cmp_header(w, "Dataset | System")
        for ds in datasets:
            base = cells.get((ds, "baseline", rr))
            ref = cmp_metrics(cells.get((ds, MAIN_METHOD, rr)), base, ds, floored)
            if ref is None:
                continue
            w(f"| {DS_LABEL[ds]} | **Ours: {COND_LABEL[MAIN_METHOD]}** | "
              + " | ".join(cmp_cells(ref, ref)) + " |")
            for c in BASELINE_ROWS:
                m = cmp_metrics(cells.get((ds, c, rr)), base, ds, floored)
                w(f"| {DS_LABEL[ds]} | {COND_LABEL[c]} | "
                  + " | ".join(cmp_cells(m, ref)) + " |")
        w("")
    if floored:
        w(f"EM and retention are blank for {', '.join(DS_LABEL[d] for d in floored)} "
          f"(EM at the floor); read F1 and BERTScore there.\n")

    # --- headline ----------------------------------------------------------
    w("## Token reduction and accuracy retention\n")
    for rr in reranks:
        w(f"**Retrieval: rerank {rr}**\n")
        w("| Dataset | Condition | Prompt tok | Context tok | Reduction | Ratio | EM | Retention | F1 |")
        w("|---|---|---|---|---|---|---|---|---|")
        for ds in datasets:
            base = cells.get((ds, "baseline", rr))
            if not base:
                continue
            for c in COND:
                v = cells.get((ds, c, rr))
                if not v:
                    continue
                red = (1 - v["ctx"] / base["ctx"]) * 100 if base["ctx"] else float("nan")
                ratio = base["ctx"] / v["ctx"] if v["ctx"] else float("nan")
                ret = v["em"] / base["em"] * 100 if base["em"] else float("nan")
                em_s = "—" if ds in floored else f"{v['em']:.1f}"
                ret_s = "—" if ds in floored or ret != ret else f"{ret:.0f}%"
                w(f"| {DS_LABEL[ds]} | {COND_LABEL[c]} | {v['prompt']:.0f} | {v['ctx']:.0f} "
                  f"| {red:.1f}% | {ratio:.1f}× | {em_s} | {ret_s} | {v['f1']:.1f} |")
        w("")

    # --- sample counts -----------------------------------------------------
    w("## Sample counts\n")
    w("Every cell evaluates the same question set, so all comparisons are paired.\n")
    w("| Dataset | n per cell | Conditions | Rerank settings | Cells | Total answers |")
    w("|---|---|---|---|---|---|")
    grand = 0
    for ds in datasets:
        got = [(c, rr) for c in COND for rr in reranks if (ds, c, rr) in cells]
        ns = {cells[(ds, c, rr)]["n"] for c, rr in got}
        n_s = str(ns.pop()) if len(ns) == 1 else "/".join(str(x) for x in sorted(ns))
        tot = sum(cells[(ds, c, rr)]["n"] for c, rr in got)
        grand += tot
        w(f"| {DS_LABEL[ds]} | {n_s} | {len({c for c, _ in got})} | "
          f"{len({r for _, r in got})} | {len(got)} | {tot} |")
    w(f"| **Total** | | | | **{len(cells)}** | **{grand}** |\n")

    # --- F1 ----------------------------------------------------------------
    w("## Token F1 — all conditions\n")
    w("F1 degrades more gracefully than EM and stays informative where EM floors.\n")
    for rr in reranks:
        w(f"**Retrieval: rerank {rr}**\n")
        w("| Dataset | " + " | ".join(COND_LABEL[c] for c in COND) +
          " | Δ main vs uncompressed | Retention |")
        w("|---|" + "---|" * (len(COND) + 2))
        for ds in datasets:
            base = cells.get((ds, "baseline", rr))
            main = cells.get((ds, MAIN_METHOD, rr))
            vals = []
            for c in COND:
                v = cells.get((ds, c, rr))
                vals.append(f"{v['f1']:.1f} [{v['f1_ci'][0]:.1f}, {v['f1_ci'][1]:.1f}]" if v else "—")
            d_s = f"{main['f1']-base['f1']:+.1f}" if base and main else "—"
            r_s = f"{main['f1']/base['f1']*100:.0f}%" if base and main and base["f1"] else "—"
            w(f"| {DS_LABEL[ds]} | " + " | ".join(vals) + f" | {d_s} | {r_s} |")
        w("")
    w("Values are mean with 95% bootstrap CI.\n")

    # --- BERTScore ---------------------------------------------------------
    if any(v["bert"] == v["bert"] for v in cells.values()):
        w("## BERTScore F1 — all conditions\n")
        w("Semantic similarity to the reference answer. Because it does not require "
          "surface-form agreement, it separates *phrasing* loss from *content* loss - "
          "a compressor that paraphrases is penalised by EM but not by BERTScore.\n")
        for rr in reranks:
            w(f"**Retrieval: rerank {rr}**\n")
            w("| Dataset | " + " | ".join(COND_LABEL[c] for c in COND) + " | Δ main vs uncompressed |")
            w("|---|" + "---|" * (len(COND) + 1))
            for ds in datasets:
                base = cells.get((ds, "baseline", rr))
                main = cells.get((ds, MAIN_METHOD, rr))
                vals = []
                for c in COND:
                    v = cells.get((ds, c, rr))
                    vals.append(f"{v['bert']:.2f}" if v and v["bert"] == v["bert"] else "—")
                d_s = (f"{main['bert']-base['bert']:+.2f}"
                       if base and main and base["bert"] == base["bert"] else "—")
                w(f"| {DS_LABEL[ds]} | " + " | ".join(vals) + f" | {d_s} |")
            w("")
    else:
        w("## BERTScore F1\n")
        w("_Not computed. Re-run with `--bertscore` (slow on first pass, cached after)._\n")

    # --- head to head vs the proposed method -------------------------------
    w(f"## Variants vs the proposed method ({COND_LABEL[MAIN_METHOD]})\n")
    w("Paired comparison on identical questions. **ΔEM** and **ΔF1** are *other minus "
      "main*, so negative means the proposed method wins. `p` is McNemar's test on "
      "exact-match correctness.\n")
    w("| Dataset | rerank | Variant | ΔEM | ΔF1 | p (EM) | Δ context tok |")
    w("|---|---|---|---|---|---|---|")
    for ds in datasets:
        for rr in reranks:
            main = cells.get((ds, MAIN_METHOD, rr))
            if not main:
                continue
            for c in COND:
                if c == MAIN_METHOD:
                    continue
                v = cells.get((ds, c, rr))
                if not v:
                    continue
                p_em = mcnemar_p(v["ems"], main["ems"])
                star = "**" if p_em < 0.05 else ""
                w(f"| {DS_LABEL[ds]} | {rr} | {COND_LABEL[c]} | {v['em']-main['em']:+.1f} "
                  f"| {v['f1']-main['f1']:+.1f} | {star}{p_em:.4f}{star} "
                  f"| {v['ctx']-main['ctx']:+.0f} |")
    w("")
    w("Bold p-values are significant at 0.05. A negative Δ with a positive Δ context "
      "means the variant is worse *and* not cheaper.\n")

    # --- totals ------------------------------------------------------------
    w("## Total token spend across all datasets\n")
    w("| Condition | rerank | Total prompt tokens | Saved vs uncompressed |")
    w("|---|---|---|---|")
    for rr in reranks:
        basetot = sum(cells[(d, "baseline", rr)]["prompt"] * cells[(d, "baseline", rr)]["n"]
                      for d in datasets if (d, "baseline", rr) in cells)
        for c in COND:
            tot = sum(cells[(d, c, rr)]["prompt"] * cells[(d, c, rr)]["n"]
                      for d in datasets if (d, c, rr) in cells)
            if not tot:
                continue
            saved = (1 - tot / basetot) * 100 if basetot else float("nan")
            w(f"| {COND_LABEL[c]} | {rr} | {tot/1e6:.2f}M | "
              f"{'—' if c == 'baseline' else f'{saved:.1f}%'} |")
    w("")

    # --- efficiency --------------------------------------------------------
    w("## Efficiency — exact matches per 1k prompt tokens\n")
    w("| Dataset | rerank | " + " | ".join(COND_LABEL[c] for c in COND) + " |")
    w("|---|---|" + "---|" * len(COND))
    for ds in datasets:
        if ds in floored:
            continue
        for rr in reranks:
            vals = []
            for c in COND:
                v = cells.get((ds, c, rr))
                vals.append(f"{v['em']/(v['prompt']/1000):.2f}" if v and v["prompt"] else "—")
            w(f"| {DS_LABEL[ds]} | {rr} | " + " | ".join(vals) + " |")
    w("")

    # --- vs published ------------------------------------------------------
    w("## Compression ratio vs published compressors\n")
    w("Published ratios are from CompAct (EMNLP 2024) Table 2 on HotpotQA with a "
      "LLaMA3-8B reader. They are context for where a training-free compressor sits, "
      "not a like-for-like comparison.\n")
    w("| System | Compression | Trained? |")
    w("|---|---|---|")
    for rr in reranks:
        v = cells.get(("hotpotqa", "filter_summ", rr))
        b = cells.get(("hotpotqa", "baseline", rr))
        if v and b and v["ctx"]:
            w(f"| **Ours: keyword + FLAN-T5 (rerank {rr})** | **{b['ctx']/v['ctx']:.1f}×** | no |")
    for name, r in sorted(PUBLISHED_COMP.items(), key=lambda x: -x[1]):
        w(f"| {name} | {r:.1f}× | yes |")
    w("")

    # --- faithfulness ------------------------------------------------------
    w("## Supporting-fact retention (multi-hop)\n")
    w("Share of gold supporting sentences surviving compression. This is the "
      "mechanism behind multi-hop accuracy loss and is measured over more items "
      "than EM, so it is the more stable signal.\n")
    w("| Dataset | rerank | " + " | ".join(COND_LABEL[c] for c in COND) + " |")
    w("|---|---|" + "---|" * len(COND))
    for ds in datasets:
        for rr in reranks:
            vals = []
            for c in COND:
                v = cells.get((ds, c, rr))
                vals.append(f"{v['sfr']:.0f}%" if v and v["sfr"] == v["sfr"] else "n/a")
            if any(x != "n/a" for x in vals):
                w(f"| {DS_LABEL[ds]} | {rr} | " + " | ".join(vals) + " |")
    w("")

    w("## Latency (ms/question, end to end)\n")
    w("| Dataset | rerank | " + " | ".join(COND_LABEL[c] for c in COND) + " |")
    w("|---|---|" + "---|" * len(COND))
    for ds in datasets:
        for rr in reranks:
            vals = []
            for c in COND:
                v = cells.get((ds, c, rr))
                vals.append(f"{v['t_e2e']:.0f}" if v else "—")
            w(f"| {DS_LABEL[ds]} | {rr} | " + " | ".join(vals) + " |")
    w("")

    try:
        os.makedirs(os.path.dirname(BERT_CACHE) or ".", exist_ok=True)
        json.dump(cache, open(BERT_CACHE, "w"))
    except Exception:
        pass

    open(args.out, "w").write("\n".join(L))
    print(f"wrote {args.out} ({len(L)} lines, {len(cells)} cells)")


if __name__ == "__main__":
    main()
