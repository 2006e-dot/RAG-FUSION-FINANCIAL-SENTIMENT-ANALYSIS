
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
- Fine-tuned Qwen2.5-3B-Instruct
- Cross-Domain Evaluation
- Modular Retrieval Pipeline

---

## Retrieval Pipeline

```text
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
Fine-Tuned Qwen2.5-3B-Instruct
   │
   ▼
Sentiment Prediction
```

---

## Repository Structure

```text
.
├── retriever.py
├── evaluate.py
├── chroma_db.py
├── res.py
├── .gitignore
└── README.md
```

---

## Models

| Component | Configuration |
|---|---|
| Base Language Model | Qwen2.5-3B-Instruct |
| Fine-Tuning Method | LoRA |
| Embedding Model | BAAI/bge-m3 |
| Vector Database | ChromaDB |
| Sparse Retriever | BM25 |
| Reranking | Maximum Marginal Relevance (MMR) |

---

## Fine-Tuning Results

The Qwen2.5-3B-Instruct model was fine-tuned for financial sentiment classification using Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique.

The training process was completed successfully over three epochs, comprising 384 optimization steps.

### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | Qwen2.5-3B-Instruct |
| Fine-Tuning Method | LoRA |
| Training Epochs | 3 |
| Optimization Steps | 384 |
| Training Completion | 100% |

### Numerical Summary

| Metric | Value |
|---|---:|
| Final Training Loss | 0.0674 |
| Final Evaluation Loss | 0.2165 |
| Approximate Minimum Evaluation Loss | ~0.20 |
| Approximate Step of Minimum Evaluation Loss | ~234 |
| Training-Evaluation Loss Difference | 0.1491 |

### Training Analysis

The training loss decreased substantially during the initial optimization stages and subsequently stabilized at a relatively low value.

The evaluation loss also decreased during the early stages of training, reaching an approximate minimum of 0.20 around optimization step 234. During the later stages, the evaluation loss exhibited a slight increase and eventually reached 0.2165.

The difference between the final training and evaluation losses, along with the slight increase in evaluation loss during the later stages, suggests the possibility of mild overfitting. However, the observed behavior does not provide sufficient evidence of severe overfitting.

The results indicate that the model successfully learned the financial sentiment classification task. Further assessment using independent test data and classification metrics is required to establish its generalization performance conclusively.

---

## Evaluation Datasets

The model is evaluated on publicly available financial sentiment benchmarks.

| Dataset | Purpose |
|---|---|
| SEntFiN | Hold-out Evaluation |
| Financial PhraseBank | Hold-out Evaluation |
| Twitter Financial News Sentiment (TFNS) | Cross-Domain Evaluation |

Datasets are **not included** in this repository and should be obtained from their respective official sources.

---

## Current Classification Results

The following metrics correspond to the current hybrid retrieval pipeline.

| Dataset | Samples | Accuracy | Macro F1 |
|---|---:|---:|---:|
| SEntFiN | 2000 | **79.35%** | **79.60%** |
| Financial PhraseBank | 2000 | **84.70%** | **81.88%** |
| Twitter Financial News | 11872 | **78.09%** | **71.84%** |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/2006e-dot/RAG-FUSION-FINANCIAL-SENTIMENT-ANALYSIS.git
cd RAG-FUSION-FINANCIAL-SENTIMENT-ANALYSIS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running

Create the vector database:

```bash
python chroma_db.py
```

Run evaluation:

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
