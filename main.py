import json
from uuid import uuid1
from tqdm import tqdm
import pandas as pd
from rag.indexing import Indexing
from rag.retrieval import Retriever


def get_ans_from_rag(data):
    texts = pd.read_csv('DATASET/rag-mini-wikipedia/passages.csv')['passage'].to_list()
    indexing = Indexing(texts)
    index, nodes = indexing.get_index()
    retriver = Retriever(index, nodes, indexing.token_counter)
    human_pormpt = "If the correct answer is yes or no, reply ONLY 'yes' or 'no'. Otherwise, reply with the exact answer and no additional text"
    qan = []
    for i in tqdm(range(len(data))):
        question = data.loc[i]['question']
        answer = data.loc[i]['answer']
        d = {'id':str(uuid1()), "question":question, "answer":answer}
        # rag_answer, token_count = retriver.get_response(question, human_pormpt=human_pormpt)
        # ("get_token_effi_response" contribution to final answer)
        rag_answer, token_count = retriver.get_token_effi_response(question, human_pormpt=human_pormpt)
        d['rag_answer'] = rag_answer
        d['prompt_token'] = token_count['prompt_token']
        d['completion_token'] = token_count['completion_token']
        d['total_token'] = token_count['total_token']
        qan.append(d)

    with open(f'OUTPUT/baseline_rag.json', 'w') as f:
        json.dump(qan, f, indent=1)


data = pd.read_csv('DATASET/rag-mini-wikipedia/test.csv')
# data = data.iloc[:2]
get_ans_from_rag(data)






