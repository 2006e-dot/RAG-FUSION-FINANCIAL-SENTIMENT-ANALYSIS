import os
import shutil
import pandas as pd
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

TRAIN_FILE = "train.csv"
CHROMA_DIR = "./chroma_db"
BATCH_SIZE = 500

if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)

train_df = pd.read_csv(TRAIN_FILE)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}, 
    encode_kwargs={"normalize_embeddings": True}
)

documents = []

for _, row in train_df.iterrows():

    documents.append(
        Document(
            page_content=str(row["title"]),
            metadata={
                "doc_id": row["doc_id"],
                "label": row["label"],
                "source": row["source"]
            }
        )
    )

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model
)

for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Creating ChromaDB"):

    batch = documents[i:i+BATCH_SIZE]

    vectorstore.add_documents(batch)

print("ChromaDB created successfully.")
print("Total Documents :", len(documents))
print("Saved at :", CHROMA_DIR)
