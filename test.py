def get_token_effi_response(self, query, human_pormpt: str = None,
                                use_summarizer: bool = True, use_keyword_filtering: bool = True):

        t0 = time.perf_counter()
        retriever = self.index.as_retriever(
            similarity_top_k=5,
            vector_store_query_mode="hybrid",
            alpha=0.5,
            node_postprocessors=[self.postproc, self.rerank],
        )
        nodes = retriever.retrieve(query)
        t_retrieval = (time.perf_counter() - t0) * 1000

        # capture the RAW retrieved context (before compression) for recall@k + ratio
        retrieved_chunks = []
        raw_texts = []
        for node_with_score in nodes:
            node = node_with_score.node
            raw_text = node.get_content(metadata_mode="all")
            raw_texts.append(raw_text)
            retrieved_chunks.append({
                "title": node.metadata.get("title", node.node_id),
                "text": raw_text,
                "score": float(node_with_score.score) if node_with_score.score is not None else None,
            })
        retrieved_context_tokens = _count_tokens("\n\n".join(raw_texts))

        compressed_snippets = []
        t_filter = 0.0
        t_summarize = 0.0
        for i, node_with_score in enumerate(nodes):
            node = node_with_score.node
            raw_text = node.get_content(metadata_mode="all")

            if use_keyword_filtering:
                tf = time.perf_counter()
                relevant = extract_relevant_sentences(raw_text, query, max_sentences=4)
                t_filter += (time.perf_counter() - tf) * 1000
            else:
                relevant = raw_text

            if use_summarizer:
                ts = time.perf_counter()
                mini_summary = summarize_for_query_with_chunks(
                    relevant, query, self.summ_tokenizer, self.summ_model, max_new_tokens=1000)
                t_summarize += (time.perf_counter() - ts) * 1000
            else:
                mini_summary = relevant

            header = f"[Snippet {i+1} | score={node_with_score.score:.3f}]"
            compressed_snippets.append(f"{header}\n{mini_summary}")

        context = "\n\n".join(compressed_snippets)
        compressed_context_tokens = _count_tokens(context)

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

        tg = time.perf_counter()
        llm = Settings.llm
        answer_resp = llm.complete(final_prompt)
        t_generate = (time.perf_counter() - tg) * 1000

        answer_raw = answer_resp.text          # BEFORE cleaning
        answer_text = answer_raw.strip()

        last_event = self.token_counter.llm_token_counts[-1]
        return answer_text, {
            'prompt_token': last_event.prompt_token_count,
            'completion_token': last_event.completion_token_count,
            'total_token': last_event.total_token_count,
            # ---- new fields the metrics need ----
            'rag_answer_raw': answer_raw,
            'compressed_context': context,
            'compressed_context_tokens': compressed_context_tokens,
            'retrieved_context_tokens': retrieved_context_tokens,
            'retrieved_chunks': retrieved_chunks,
            't_retrieval_ms': t_retrieval,
            't_filter_ms': t_filter,
            't_summarize_ms': t_summarize,
            't_generate_ms': t_generate,
            't_end_to_end_ms': t_retrieval + t_filter + t_summarize + t_generate,
        }