import os
import time
import pandas as pd
from tqdm import tqdm

from retriever import rag_fusion_retrieve

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

TEST_FILE="data_raw.csv"
OUTPUT_FILE="predictions_harixn.csv"
CHECKPOINT_EVERY=20
SLEEP_TIME=0
MAX_RETRIES=3

test_df=pd.read_csv(TEST_FILE)

llm=ChatOllama(
    model="finance-bot-3b:latest",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert financial sentiment classifier.

Classify the headline into EXACTLY ONE label using the provided context.

Labels:
- positive
- neutral
- negative

Rules:
- positive: clearly indicates favourable financial performance, growth, gains, upgrades, rallies, or positive developments.
- negative: clearly indicates losses, declines, downgrades, lawsuits, weak earnings, risks, or negative developments.
- neutral: reports factual information (appointments, filings, approvals, listings, mergers, launches, board changes, routine announcements) without expressing positive or negative market sentiment.

Use the context only to resolve ambiguity in the headline — do not invent sentiment that isn't supported.

Respond with only the label, lowercase, no punctuation, no explanation."""
        ),
        (
            "human",
            """Context:
{context}

Headline:
{headline}

Answer:"""
        ),
    ]
)
chain=prompt|llm


def parse_prediction(text):

    text=text.lower()

    if "positive" in text:
        return "positive"

    if "negative" in text:
        return "negative"

    if "neutral" in text:
        return "neutral"

    return "parse_error"


def predict(headline,context):

    for _ in range(MAX_RETRIES):

        try:

            response=chain.invoke(
            {
                "headline":headline,
                "context":context
            })

            label=parse_prediction(
                response.content
            )

            if label!="parse_error":

                return label

        except Exception:

            pass

    return "parse_error"

if os.path.exists(OUTPUT_FILE):

    done_df=pd.read_csv(OUTPUT_FILE)

    results=done_df.to_dict("records")

    done_ids=set(done_df["doc_id"])

else:

    results=[]

    done_ids=set()

rows=test_df.iloc[3100:].to_dict('records')
for i,row in enumerate(tqdm(rows)):
    if row['doc_id'] in done_ids:
        continue

    docs=rag_fusion_retrieve(row['title'])

    context="\n\n".join(docs)

    prediction=predict(row['title'],context)

    results.append({'doc_id':row['doc_id'],'title':row['title'],'true_label':str(row['label']).lower(),'predicted_label':prediction})

    if(i+1)%CHECKPOINT_EVERY==0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE,index=False)

    time.sleep(SLEEP_TIME)


pd.DataFrame(results).to_csv(
    OUTPUT_FILE,
    index=False
)

print("Done.")



