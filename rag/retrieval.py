from copy import deepcopy
import re
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import Settings

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Settings.debug = True


def _simple_tokenize(text: str) -> List[str]:
    """Very simple tokenizer: lowercase, split on non-letters."""
    return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]

def _score_sentence(sent: str, query_tokens: set) -> int:
    """Score sentence by keyword overlap with query."""
    sent_tokens = set(_simple_tokenize(sent))
    return len(sent_tokens & query_tokens)

def extract_relevant_sentences(text: str, query: str, max_sentences: int = 3) -> str:
    """
    Pick the top-k sentences from `text` that are most relevant to `query`
    based on simple word overlap.
    """
    # naïve sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    query_tokens = set(_simple_tokenize(query))

    scored = [(s, _score_sentence(s, query_tokens)) for s in sentences if s.strip()]
    scored.sort(key=lambda x: x[1], reverse=True)

    top = [s for s, score in scored[:max_sentences] if score > 0]
    # fallback: if no overlap, take the first few sentences
    if not top:
        top = sentences[:max_sentences]

    return " ".join(top)

def _chunk_text_by_tokens(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> List[str]:
    """
    Split `text` into chunks where each chunk's tokenized length
    is <= max_tokens. We never drop content.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current_sentences = []
    current_tokens = 0

    for sent in sentences:
        if not sent.strip():
            continue
        n_tokens = len(
            tokenizer(
                sent,
                add_special_tokens=False,
            ).input_ids
        )
        # If adding this sentence would overflow the window, start new chunk
        if current_sentences and current_tokens + n_tokens > max_tokens:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sent]
            current_tokens = n_tokens
        else:
            current_sentences.append(sent)
            current_tokens += n_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def summarize_for_query(passage: str, query: str, summ_tokenizer, summ_model, max_new_tokens: int = 1000) -> str:
    """
    Use a *local* Hugging Face model (no OpenAI) to create a short,
    query-focused summary of a passage.
    """
    # prompt tailored for FLAN / T5-style models
    prompt = (
        "Summarize the following passage so that it only contains "
        "information useful to answer the question.\n\n"
        f"Question: {query}\n\n"
        f"Passage:\n{passage}\n\n"
        "Summary:"
    )

    inputs = summ_tokenizer(
        prompt,
        return_tensors="pt",
        # truncation=True,
        # max_length=1024,      # truncate long passages for the summarizer
    ).to(_device)

    with torch.no_grad():
        output_ids = summ_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,   # ~120–180 token summary
            do_sample=False,
        )

    summary = summ_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary.strip()

def summarize_for_query_with_chunks(
    passage: str,
    query: str,
    summ_tokenizer, summ_model,
    max_new_tokens: int = 120,
    max_prompt_tokens: int = 512,
) -> str:
    """
    Query-aware summarization using a local HF model *without discarding
    any part of the original passage*.

    1. Split passage into token-based chunks that fit the model context.
    2. Summarize each chunk conditioned on the query.
    3. If there are multiple chunk summaries, summarize them again
       into a single final summary.
    """
    # Leave some room in the prompt for instructions + question
    chunk_token_budget = max_prompt_tokens - 128
    if chunk_token_budget <= 0:
        chunk_token_budget = max_prompt_tokens

    # 1) Chunk the original passage -> we see ALL text at least once
    chunks = _chunk_text_by_tokens(passage, summ_tokenizer, chunk_token_budget)

    partial_summaries: List[str] = []

    # 2) Summarize each chunk
    for ch in chunks:
        prompt = (
            "Summarize the following passage so that it only contains\n"
            "information useful to answer the question.\n\n"
            f"Question: {query}\n\n"
            f"Passage:\n{ch}\n\n"
            "Summary:"
        )

        inputs = summ_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,         # truncation now only affects the *prompt*
            max_length=max_prompt_tokens,
        ).to(_device)

        with torch.no_grad():
            output_ids = summ_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        summary = summ_tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()
        partial_summaries.append(summary)

    # 3) If only one chunk -> done
    if len(partial_summaries) == 1:
        return partial_summaries[0]

    # 4) Otherwise, summarize the summaries once more
    combined = " ".join(partial_summaries)
    final_prompt = (
        "You are given several partial summaries of a longer passage.\n"
        "Combine them into a single concise summary that only includes\n"
        "information relevant to the question.\n\n"
        f"Question: {query}\n\n"
        f"Partial summaries:\n{combined}\n\n"
        "Final summary:"
    )

    final_inputs = summ_tokenizer(
        final_prompt,
        return_tensors="pt",
        truncation=True,         # this only truncates summaries-of-summaries
        max_length=max_prompt_tokens,
    ).to(_device)

    with torch.no_grad():
        final_ids = summ_model.generate(
            **final_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    final_summary = summ_tokenizer.decode(
        final_ids[0],
        skip_special_tokens=True,
    ).strip()
    return final_summary


class Retriever:

    def __init__(self, index, nodes, token_counter, summ_model_name=None) -> None:
        self.index, self.nodes = index, nodes
        self.postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        # BAAI/bge-reranker-base
        # link: https://huggingface.co/BAAI/bge-reranker-base
        self.model_reranker = "BAAI/bge-reranker-base"
        # model_reranker = model
        self.rerank = SentenceTransformerRerank(top_n = 5, model = self.model_reranker)
        self.token_counter = token_counter

        if summ_model_name:
            self.summ_tokenizer = AutoTokenizer.from_pretrained(summ_model_name)
            self.summ_model = AutoModelForSeq2SeqLM.from_pretrained(summ_model_name).to(_device)
            self.summ_model.eval()
            

    def get_response(self, query, human_pormpt=None):


        query_engine = self.index.as_query_engine(
            similarity_top_k = 5,
            vector_store_query_mode="hybrid",
            alpha=0.5,
            node_postprocessors = [self.postproc, self.rerank],
        )

        if human_pormpt:
            query = query + human_pormpt

        response = query_engine.query(query)

        last_event = self.token_counter.llm_token_counts[-1]
        return str(response), {
            'prompt_token':last_event.prompt_token_count,
            'completion_token':last_event.completion_token_count,
            'total_token':last_event.total_token_count,
        }
    
    def get_token_effi_response(self, query, human_pormpt: str = None, 
                                use_summarizer: bool = True, use_keyword_filtering: bool = True):

        # Retrieval + rerank (no generation yet) (contribution to final answer)
        retriever = self.index.as_retriever(
            similarity_top_k=5,
            vector_store_query_mode="hybrid",
            alpha=0.5,
            node_postprocessors=[self.postproc, self.rerank],
        )
        nodes = retriever.retrieve(query)  

        # Build compressed, query-aware snippets(compressed representation)contribution to final answer
        compressed_snippets = []
        for i, node_with_score in enumerate(nodes):
            node = node_with_score.node
            raw_text = node.get_content(metadata_mode="all")

            # keyword-based sentence filtering (contribution to final answer)
            if use_keyword_filtering:
                relevant = extract_relevant_sentences(raw_text, query, max_sentences=4)
            else:
                relevant = raw_text

            if use_summarizer:
                # LM-based query-aware summarization (contribution to final answer)
                mini_summary = summarize_for_query_with_chunks(relevant, query,
                                                   self.summ_tokenizer,self.summ_model, max_new_tokens=1000)
            else:
                mini_summary = relevant

            header = f"[Snippet {i+1} | score={node_with_score.score:.3f}]"
            compressed_snippets.append(f"{header}\n{mini_summary}")

        context = "\n\n".join(compressed_snippets)

        # Final prompt for answer generation(contribution to final answer)
        instruction = (
            "If the correct answer is yes or no, reply ONLY 'yes' or 'no'. "
            "Otherwise, reply with the exact answer and no additional text.\n"
        )
        if human_pormpt:
            instruction += human_pormpt + "\n"

        final_prompt = f"""{instruction}
You are given several short, query-aware snippets that are already 
highly relevant to the question. Answer using ONLY this information.

Question: {query}

Snippets:
{context}

Answer:"""

        llm = Settings.llm
        answer_resp = llm.complete(final_prompt)
        answer_text = answer_resp.text.strip()

        last_event = self.token_counter.llm_token_counts[-1]
        return answer_text, {
            'prompt_token':last_event.prompt_token_count,
            'completion_token':last_event.completion_token_count,
            'total_token':last_event.total_token_count,
        }