# RAG-FUSION-FINANCIAL-SENTIMENT-ANALYSIS

## Overview

This project presents a Retrieval-Augmented Generation (RAG) pipeline for financial sentiment classification. Instead of relying solely on a fine-tuned language model, the system retrieves semantically relevant financial news headlines from a knowledge base and uses them as contextual evidence for sentiment prediction.

The objective is to improve classification performance while maintaining better generalization across unseen financial domains.

---

## Features

- Retrieval-Augmented Generation (RAG)
- Hybrid Dense + Sparse Retrieval
- ChromaDB Vector Database
- BGE-M3 Sentence Embeddings
- Maximum Marginal Relevance (MMR) Reranking
- Fine-tuned Qwen 2.5 Model
- Cross-Domain Evaluation
- Modular Retrieval Pipeline

---

## Retrieval Pipeline

```
Query
   │
   ▼
BGE-M3 Embedding
   │
   ▼
Dense Retrieval (ChromaDB)
        +
Sparse Retrieval (BM25)
   │
   ▼
Candidate Fusion
   │
   ▼
MMR Reranking
   │
   ▼
Fine-Tuned Qwen 2.5
   │
   ▼
Sentiment Prediction
```

---

## Repository Structure

```
.
├── retriever.py
├── evaluate.py
├── chroma_db.py
├── res.py
├── .gitignore
└── README.md
```

---

## Model

- Base Model: Qwen 2.5
- Fine-Tuning: LoRA
- Embedding Model: BAAI/bge-m3
- Vector Database: ChromaDB

---

## Evaluation Datasets

The model is evaluated on publicly available financial sentiment benchmarks.

| Dataset | Purpose |
|----------|---------|
| SEntFiN | Hold-out Evaluation |
| Financial PhraseBank | Hold-out Evaluation |
| Twitter Financial News Sentiment (TFNS) | Cross-Domain Evaluation |

Datasets are **not included** in this repository and should be obtained from their respective official sources.

---

## Current Results

| Dataset | Samples | Accuracy | Macro F1 |
|----------|---------|---------:|---------:|
| SEntFiN | 2000 | **79.35%** | **79.60%** |
| Financial PhraseBank | 2000 | **84.70%** | **81.88%** |
| Twitter Financial News | 11872 | **78.09%** | **71.84%** |

These metrics correspond to the current hybrid retrieval pipeline.

---

## Installation

Clone the repository

```bash
git clone https://github.com/2006e-dot/RAG-FUSION-FINANCIAL-SENTIMENT-ANALYSIS.git
cd RAG-FUSION-FINANCIAL-SENTIMENT-ANALYSIS
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running

Create the vector database

```bash
python chroma_db.py
```

Run evaluation

```bash
python evaluate.py
```

---

## Ongoing Work

The current implementation serves as the baseline system.

Future work includes:

- Adaptive Retrieval Controller
- Dynamic Retrieval Depth Selection
- Dynamic Dense/Sparse Allocation
- Adaptive MMR Parameter Optimization
- Comprehensive Ablation Studies

---

