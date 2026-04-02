from fileinput import filename
import json
from uuid import uuid1
from tqdm import tqdm
import pandas as pd
from rag.indexing import Indexing
from rag.retrieval import Retriever


def get_ans_from_rag_for_rag_mini_wikipedia(outfile, summary_model=None):
    data = pd.read_csv('DATASET/rag-mini-wikipedia/test.csv')
    # data = data.iloc[:2]
    texts = pd.read_csv('DATASET/rag-mini-wikipedia/passages.csv')['passage'].to_list()
    indexing = Indexing(texts)
    index, nodes = indexing.get_index()
    retriver = Retriever(index, nodes, indexing.token_counter, summary_model)
    human_pormpt = "If the correct answer is yes or no, reply ONLY 'yes' or 'no'. Otherwise, reply with the exact answer and no additional text"
    qan = []
    for i in tqdm(range(len(data))):
        question = data.loc[i]['question']
        answer = data.loc[i]['answer']
        d = {'id':str(uuid1()), "question":question, "answer":answer}
        if summary_model is None:
            rag_answer, token_count = retriver.get_response(question, human_pormpt=human_pormpt)
        else:
            rag_answer, token_count = retriver.get_token_effi_response(question, human_pormpt=human_pormpt, 
                                                                       use_summarizer=True, use_keyword_filtering=False)
        d['rag_answer'] = rag_answer
        d['prompt_token'] = token_count['prompt_token']
        d['completion_token'] = token_count['completion_token']
        d['total_token'] = token_count['total_token']
        qan.append(d)

    with open(outfile, 'w') as f:
        json.dump(qan, f, indent=1)

# get_ans_from_rag_for_rag_mini_wikipedia(outfile=f'OUTPUT/baseline_rag.json')
# get_ans_from_rag_for_rag_mini_wikipedia(f'OUTPUT/t5_small_sumry.json', "google/flan-t5-small")
# get_ans_from_rag_for_rag_mini_wikipedia(f'OUTPUT/t5_base_sumry.json', "google/flan-t5-base")
# get_ans_from_rag_for_rag_mini_wikipedia(f'OUTPUT/t5_large_sumry.json', "google/flan-t5-large")
# get_ans_from_rag_for_rag_mini_wikipedia(f'OUTPUT/no_sumry_only_keyword_based_filtering.json', "google/flan-t5-small")
get_ans_from_rag_for_rag_mini_wikipedia(f'OUTPUT/no_keyword_based_filtering_only_sumry.json', "google/flan-t5-small")

