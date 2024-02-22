# Duke Brodhead Center Food Guide - RAG Enhanced

## Overview

This project aims to enhance the domain knowledge of food options at the Brodhead Center at Duke University by implementing a Retrieval-Augmented Generation (RAG) system. The RAG system enriches the prompts provided to a Large Language Model (LLM) with relevant information retrieved from a preprocessed database of food items, resulting in more informed and contextually relevant responses.

## Data

The data for this project is stored in the `data` directory:

- `raw`: Contains raw JSON files with details of food items from various restaurants at Brodhead Center, each item having keys for name, description, price, and type.
- `processed`: Contains the SQLite3 database `vectors.sqlite3` which stores the embeddings of food item summaries.

## Pipeline

The pipeline consists of the following steps:

1. **Data Preprocessing**: JSON files in the `raw` folder are processed to clean and organize the data.
2. **Summary Generation**: Summaries for each food item are generated, which include name, description, price, type, and restaurant.
3. **Embedding Storage**: These summaries are then chunked, and their embeddings are generated using the SentenceTransformer model `all-mpnet-base-v2` and stored in the SQLite3 database.
4. **Inference**: The inference pipeline is triggered through a Streamlit application that takes a user query, retrieves relevant chunks using cosine similarity, and appends them to the LLM prompt for enhanced response generation.

## Model

The SentenceTransformer `all-mpnet-base-v2` model is used to generate embeddings. The GPT-3.5-turbo model from OpenAI is used to generate responses from prompts, both with and without RAG content.

## Performance

The system's performance is evaluated based on the quality of responses generated with and without RAG content. The BERTScore metric is used to measure the similarity between the generated response and a reference text.

## Files

- `util.py`: Contains utility functions including database operations, embedding generation, and similarity computation.
- `store.py`: Main script to preprocess data, generate summaries, generate embeddings, and store them in the database.
- `constants.py`: Stores constants like database name.
- `inference.py`: Contains the Streamlit application for user interaction and inference.

## Usage

Install the package and run the Streamlit application:

```bash
pip install -r requirements.txt
streamlit run inference.py
```

## Screenshots

![Response without RAG](https://github.com/rootsec1/RAG_from_scratch/assets/20264867/93565ea0-738c-4ae0-a184-c6c1e95d0faf)
*Response without RAG*

![Response with RAG](https://github.com/rootsec1/RAG_from_scratch/assets/20264867/59f5ed9b-404a-4066-a5b6-58f489ababc2)
*Response with RAG*

![Similarity score](https://github.com/rootsec1/RAG_from_scratch/assets/20264867/94220439-2ccc-482e-a9ac-fbd2c543288f)
*Similarity score*
