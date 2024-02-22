import sqlite3
import numpy as np
import ast
import re

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from bert_score import score

# Local
from constants import *

similarity_model = SentenceTransformer("all-mpnet-base-v2").to(ML_DEVICE)
openai_client = OpenAI()


def insert_embeddings_to_db(id_list, chunk_list, embedding_list):
    # Store embeddings and associated chunks in SQLite database with columns: hash (unique) bigint, vector text
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS vectors (hash BIGINT UNIQUE, chunk TEXT, embeddings TEXT)"
    )
    for id, chunk, embedding in tqdm(zip(id_list, chunk_list, embedding_list)):
        cursor.execute(
            "INSERT INTO vectors (hash, chunk, embeddings) VALUES (?, ?, ?)",
            (id, chunk, str(embedding))
        )
    conn.commit()


def fetch_all_embeddings_from_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vectors")
    result = cursor.fetchall()
    return result


def generate_embedding_for_string(text: str) -> list:
    # Generate embeddings for the input string
    embedding_list = similarity_model.encode(
        text,
        convert_to_tensor=True
    ).to(ML_DEVICE)
    embedding_list = embedding_list.tolist()
    return embedding_list


def cosine_similarity_custom(v1: list, v2: list) -> float:
    # Compute the cosine similarity between two numeric vectors."""
    # Convert input to numpy arrays of type float
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    # Compute the cosine similarity
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)

    return similarity


def convert_string_to_list(string) -> list:
    """
    Convert a string representation of a list into a list object.

    Parameters:
    - string (str): The string representation of the list.

    Returns:
    - list: The actual list object.
    """
    try:
        # Use ast.literal_eval to safely evaluate the string
        result = ast.literal_eval(string)
        if isinstance(result, list):
            return result
        else:
            raise ValueError("Provided string does not represent a list.")
    except ValueError as e:
        print(f"Error: {e}")
    except SyntaxError as e:
        print(f"Syntax error in string: {e}")


def generate_response_for_prompt_from_llm(prompt: str, RAG_content: str) -> str:
    # Generate a response for the given prompt using the OpenAI Language Model
    prompt = USER_SYSTEM_MESSAGE.format(RAG_content, prompt)
    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": LLM_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


def preprocess_prompt(prompt: str) -> str:
    if prompt.find(" ") == -1:
        return prompt

    # Remove the first word from prompt since it's most likely an instruction, we don't need that for searching in the DB
    prompt = prompt[prompt.index(" ") + 1:]

    # Remove all stopwords from prompt
    prompt = prompt.lower().strip()
    prompt = " ".join(
        [word for word in prompt.split() if word not in STOPWORDS]
    )

    # Get all non-stopwords from LLM_SYSTEM_MESSAGE and remove them from prompt
    system_message = LLM_SYSTEM_MESSAGE.lower().strip()
    system_message = " ".join(
        [word for word in system_message.split() if word not in STOPWORDS]
    )
    unique_words_in_system_message = set(system_message.split(" "))

    for word in prompt.split(" "):
        if word in unique_words_in_system_message:
            prompt = prompt.replace(word, "")

    prompt = prompt.lower().strip()

    # Replace all multiple spaces with single spaces using regex
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.strip()

    return prompt


def craft_RAG_using_top_K_chunks(top_k_chunks: list) -> str:
    # Craft a response using the top K chunks
    RAG_content = "\n\n".join(top_k_chunks)
    return RAG_content


def compute_bert_score_for_response(llm_response: str) -> float:
    # Compute the BERTScore between the prompt and response
    _, _, similarity = score(
        [llm_response],
        [REFERENCE_TEXT],
        lang="en",
        model_type="roberta-large",
        device=ML_DEVICE
    )
    return similarity.item()
