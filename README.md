# README

## Overview

This project implements a top-K session-based recommendation pipeline using Large Language Models (LLMs) and a graph database (Neo4j). The system infers user intent based on previously interacted items in a session and retrieves semantically similar items. Graph-based reranking is also explored to enhance recommendation quality.

---

## File Structure

### **data/**

Contains all datasets used in the project.

* `products_train_UK.csv`: Metadata for products (title, description, brand, price, etc.). Due to size, not included. Download from:https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge
* `sessions_train.csv`: Training sessions for potential future model fine-tuning. Due to size, not included. Download from the same link above.
* `transactions.csv`: Processed session data in a (prev_item, next_item) format, used for constructing co-occurrence relationships in the graph. Due to size, not included.
* `td_embedding.npy`: Pre-computed embeddings of product title+description. Due to size, not included. Can be generated using `embedding.py`.
* `sessions_test_task1_phase1_UK_100.csv`: 100 test sessions used in evaluation.
* `gt_task1_UK_100.csv`: Ground truth next-item labels for the 100 test sessions.

---

### **src/**

Contains source code modules.

* `db_operation.py`: Handles Neo4j operations including graph construction, node insertion, attribute embedding, relationship creation, and vector index building.
* `embedding.py`: Encodes product title + description into vector embeddings and saves them as `.npy` files.
* `evaluate.py`: Runs the complete recommendation pipeline — from intent generation to item retrieval — and computes evaluation metrics (MRR, Recall\@K).
* `rerank.py`: Defines the reranking class that reorders candidate items using graph-based features (e.g., Personalized PageRank or path-based scoring).

---

### **result/**

Contains experimental outputs.

* `result_100_prompt1.csv`: Recommendation results using the original prompt with Top-20 retrieval.
* `result_100_prompt2.csv`: Recommendation results using a revised, more abstract intent prompt.
* `result_100_k50.csv`: Top-50 retrieval results using the original prompt (without reranking).
* `result_100_k50_rerank.csv`: Final reranked results using graph-based methods on Top-50 candidates.

---

## How to Run

1. **Graph Construction**
   Run `db_operation.py` to build the product graph in Neo4j and insert metadata, embeddings, and relationships.

2. **Embedding**
   Run `embedding.py` to generate and save product text embeddings.

3. **Evaluation**
   Run `evaluate.py` to evaluate different experimental setups and output recommendation results.

---

## Dependencies

* OpenAI GPT API (GPT-3.5-Turbo)
* Neo4j
* numpy
* pandas
* openai
* tqdm