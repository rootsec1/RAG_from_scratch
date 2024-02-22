import numpy as np
import pandas as pd
import streamlit as st

# Local
from util import *


def load_db_into_dataframe(rows: list) -> pd.DataFrame:
    """
    Convert list of tuples to DataFrame.

    Parameters:
    - rows (list): The list of tuples to convert.

    Returns:
    - pd.DataFrame: The converted DataFrame.
    """
    df = pd.DataFrame(rows, columns=["hash", "chunk", "embeddings"])
    return df


def generate_similarity_score(user_query, db_query_embedding: list) -> str:
    """
    Generate the similarity score between the user query and the database query embedding.

    Parameters:
    - user_query (str): The user query.
    - db_query_embedding (list): The database query embedding.

    Returns:
    - str: The similarity score.
    """
    # Generate the embedding for the user query
    user_query_embedding = generate_embedding_for_string(user_query)
    db_query_embedding = ast.literal_eval(db_query_embedding)

    # Compute the cosine similarity between the user query embedding and the database query embedding
    similarity = cosine_similarity_custom(
        db_query_embedding,
        user_query_embedding
    )

    return similarity


def vectorized_similarity_score(row, prompt) -> str:
    """
    Compute the similarity score for a row in the DataFrame.

    Parameters:
    - row (pd.Series): The row in the DataFrame.
    - prompt (str): The prompt.

    Returns:
    - str: The similarity score.
    """
    db_query_embedding = row["embeddings"]
    similarity = generate_similarity_score(prompt, db_query_embedding)
    return similarity


def get_response_for_prompt_with_RAG(cleaned_prompt: str, df: pd.DataFrame) -> str:
    """
    Get the response for the prompt using RAG.

    Parameters:
    - cleaned_prompt (str): The cleaned prompt.
    - df (pd.DataFrame): The DataFrame.

    Returns:
    - str: The response.
    """
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
    """
    Main function to run the script.
    """
    # Fetch all embeddings from the database
    rows = fetch_all_embeddings_from_db()

    # Load the database into a DataFrame
    df = load_db_into_dataframe(rows)

    # Set up the Streamlit interface
    st.title("Duke Brodhead Center guide")
    should_use_rag = st.radio(
        "Use RAG?:",
        ["No", "Yes"],
        horizontal=True,
    )

    st.write("DataFrame of chunks and their embeddings:")
    st.dataframe(df.head())

    # Initialize the session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "What can I help you with?"}
        ]

    # Display the chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle the user input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Preprocess the prompt
        cleaned_prompt = preprocess_prompt(prompt)
        response = None
        if should_use_rag == "Yes":
            # If RAG should be used, get the response for the prompt with RAG
            response = get_response_for_prompt_with_RAG(cleaned_prompt, df)
        else:
            # If RAG should not be used, generate the response for the prompt from LLM
            response = generate_response_for_prompt_from_llm(
                cleaned_prompt,
                ""
            )

        # Compute the BERT score for the response
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
