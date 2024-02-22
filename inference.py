import numpy as np
import pandas as pd
import streamlit as st

# Local
from util import *


def load_db_into_dataframe(rows: list) -> pd.DataFrame:
    # Convert list of tuples to DataFrame
    df = pd.DataFrame(rows, columns=["hash", "chunk", "embeddings"])
    return df


def generate_similarity_score(user_query, db_query_embedding: list) -> str:
    user_query_embedding = generate_embedding_for_string(user_query)
    db_query_embedding = ast.literal_eval(db_query_embedding)

    similarity = cosine_similarity_custom(
        db_query_embedding,
        user_query_embedding
    )

    return similarity


def vectorized_similarity_score(row, prompt) -> str:
    db_query_embedding = row["embeddings"]
    similarity = generate_similarity_score(prompt, db_query_embedding)
    return similarity


def get_response_for_prompt_with_RAG(cleaned_prompt: str, df: pd.DataFrame) -> str:
    # Compute similarity scores
    df["similarity"] = df.apply(
        lambda row: vectorized_similarity_score(row, cleaned_prompt),
        axis=1
    )
    print(df.head())

    # Get chunks for the top K similarity scores
    top_k_chunks = df.nlargest(TOP_K, "similarity")["chunk"].values

    # Craft RAG
    RAG_content = craft_RAG_using_top_K_chunks(top_k_chunks)

    # Generate response from LLM
    response = generate_response_for_prompt_from_llm(
        cleaned_prompt,
        RAG_content
    )
    return response


def main():
    rows = fetch_all_embeddings_from_db()
    df = load_db_into_dataframe(rows)

    st.title("Duke Brodhead Center guide")
    should_use_rag = st.radio(
        "Use RAG?:",
        ["No", "Yes"],
        horizontal=True,
    )

    st.write("DataFrame of chunks and their embeddings:")
    st.dataframe(df.head())

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "What can I help you with?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        cleaned_prompt = preprocess_prompt(prompt)
        response = None
        if should_use_rag == "Yes":
            response = get_response_for_prompt_with_RAG(cleaned_prompt, df)
        else:
            response = generate_response_for_prompt_from_llm(
                cleaned_prompt,
                ""
            )

        bert_score = compute_bert_score_for_response(response)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            },

        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"BERTScore for LLM response: {bert_score}"
            }
        )
        st.chat_message("assistant").markdown(response)
        st.chat_message("assistant").markdown(
            f"BERTScore for LLM response: {bert_score}"
        )


if __name__ == '__main__':
    main()

# your objective, describe the data you used
# describe your pipeline
# explain what model you used
# describe the performance of your system both 1) out-of-the-box (no RAG) and 2) with RAG
