import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
FETCH_K=30
FINAL_K=5
MMR_LAMBDA=0.7
N_QUERIES=3

train_df=pd.read_csv("train.csv")
train_titles=train_df['title'].astype(str).to_list()
train_doc_ids=train_df['doc_id'].tolist()
train_labels=train_df['label'].tolist()
train_sources = train_df["source"].tolist()

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

tokenized_corpus = [
    title.lower().split()
    for title in train_titles
]

bm25 = BM25Okapi(tokenized_corpus)

print("Caching train embeddings...")

train_embeddings = embedding_model.embed_documents(train_titles)

embedding_cache = {}

for doc_id, emb in zip(train_doc_ids, train_embeddings):

    embedding_cache[doc_id] = np.array(emb)

print("Embedding cache ready.")



llm = ChatOllama(
    model="finance-bot-3b:latest",
    temperature=0
)

query_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial search expert generating retrieval queries.

Generate {n_queries} search queries that retrieve documents relevant to the input, preserving its original meaning.

Vary the queries across different angles, e.g.:
- the core entity/company/ticker involved
- the financial event or action described
- relevant metrics, numbers, or timeframes
- broader market/sector context

Do not simply reword the same query. Each query should surface a different useful angle.

Return exactly {n_queries} queries, one per line, no numbering, no bullets, no extra text."""
        ),
        (
            "human",
            "Generate {n_queries} search queries for:\n\n{query}"
        ),
    ]
)

query_chain = query_prompt | llm

def generate_search_queries(query):

    response = query_chain.invoke(
        {
            "query": query,
            "n_queries": N_QUERIES
        }
    )

    queries = []

    for line in response.content.split("\n"):

        line = line.strip()

        if not line:
            continue

        line = re.sub(r"^[0-9\.\-\)\s]+", "", line)

        if line:
            queries.append(line)

    queries = queries[:N_QUERIES]

    return list(dict.fromkeys([query]+queries))

def mmr_select(query_embedding,
               candidate_embeddings,
               candidate_scores,
               k=FINAL_K,
               lambda_mult=MMR_LAMBDA):

    query_embedding=np.array(query_embedding)
    candidate_embeddings=np.array(candidate_embeddings)

    def cosine(a,b):

        return np.dot(a,b)/(
            np.linalg.norm(a)*
            np.linalg.norm(b)+1e-9
        )

    selected=[]
    remaining=list(range(len(candidate_embeddings)))

    while remaining and len(selected)<k:

        if not selected:

            best=remaining[
                np.argmax(candidate_scores[remaining])
            ]

        else:

            scores=[]

            for idx in remaining:

                relevance=cosine(
                    query_embedding,
                    candidate_embeddings[idx]
                )

                diversity=max(
                    cosine(
                        candidate_embeddings[idx],
                        candidate_embeddings[s]
                    )
                    for s in selected
                )

                score=lambda_mult*relevance-(1-lambda_mult)*diversity

                scores.append(score)

            best=remaining[np.argmax(scores)]

        selected.append(best)

        remaining.remove(best)

    return selected

def rag_fusion_retrieve(query):

    queries=generate_search_queries(query)

    pool={}

    for q in queries:

        dense_results=vectorstore.similarity_search_with_score(q,k=FETCH_K)

        for doc,score in dense_results:

            doc_id=doc.metadata["doc_id"]

            if doc_id not in pool:

                pool[doc_id]={
                    "text":doc.page_content,
                    "meta":doc.metadata
                }

        bm25_scores=bm25.get_scores(q.lower().split())

        top_idx=np.argsort(bm25_scores)[::-1][:FETCH_K]

        for idx in top_idx:

            doc_id=train_doc_ids[idx]

            if doc_id not in pool:

                pool[doc_id]={
                    "text":train_titles[idx],
                    "meta":{
                        "doc_id":doc_id,
                        "label":train_labels[idx],
                        "source":train_sources[idx]
                    }
                }

    texts=[]
    candidate_embeddings=[]

    for doc in pool.values():

        texts.append(doc["text"])

        candidate_embeddings.append(
            embedding_cache[doc["meta"]["doc_id"]]
        )

    candidate_embeddings=np.array(candidate_embeddings)

    query_embedding=np.array(
        embedding_model.embed_query(query)
    )

    similarity=np.dot(candidate_embeddings,query_embedding)

    selected=mmr_select(query_embedding,candidate_embeddings,similarity,k=FINAL_K,lambda_mult=MMR_LAMBDA)

    return [texts[idx] for idx in selected]