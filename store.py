from tqdm import tqdm

# Local
from setup import *
from constants import *
from util import insert_embeddings_to_db, generate_embedding_for_string


def clean_dirty_entries(dataset: list) -> list:
    """
    Clean the dirty entries in the dataset.

    Parameters:
    - dataset (list): The dataset to clean.

    Returns:
    - list: The cleaned dataset.
    """
    clean_dataset = []
    for food_item_dict in dataset:
        # Check if the food item has options or category
        if "options" in food_item_dict or "category" in food_item_dict:
            # Process the dirty entry
            clean_dataset.extend(
                process_dirty_entry(food_item_dict)
            )
        else:
            # If the food item doesn't have options or category, add it to the clean dataset as is
            clean_dataset.append(food_item_dict)
    return clean_dataset


def generate_summaries_for_food_items(clean_dataset):
    """
    Generate summaries for the food items in the clean dataset.

    Parameters:
    - clean_dataset (list): The clean dataset.

    Returns:
    - list: The list of food items with summaries.
    - list: The list of summaries.
    """
    result_list = []
    summary_list = []
    for food_item_dict in clean_dataset:
        # Extract the necessary information from the food item
        name = food_item_dict.get("name", "").lower()
        description = food_item_dict.get("description", "").lower()

        # Check if the price is a dictionary
        if isinstance(food_item_dict["price"], dict):
            # If the price is a dictionary, convert it to a string
            price = " ".join([
                str(x)
                for x in food_item_dict["price"].values()
            ]).lower()
        else:
            # If the price is not a dictionary, get it as is
            price = food_item_dict.get("price", "").lower()

        restaurant = food_item_dict.get("restaurant", "").lower()
        type = food_item_dict.get("type", "")

        # If the type is None or an empty string, set it to "Vegetarian"
        if type is None or type == "":
            type = "Vegetarian"

        # Generate tags for the food item
        tags = set()
        for word in name.split(" "):
            if word not in STOPWORDS:
                tags.add(word)

        for word in description.split(" "):
            if word not in STOPWORDS:
                tags.add(word)

        for word in price.split(" "):
            if word not in STOPWORDS:
                tags.add(word)

        for word in type.split(" "):
            if word not in STOPWORDS:
                tags.add(word)

        for word in restaurant.split(" "):
            if word not in STOPWORDS:
                tags.add(word)

        # Generate the summary for the food item
        summary = f"Title: {name} - {restaurant}\n"
        summary += f"Description: {description}\n"
        summary += f"Price: ${price}\n"
        summary += f"Type: {type}\n"
        summary += f"Restaurant: {restaurant}\n"
        summary += f"Tags: {', '.join(tags)}\n"

        # Add the summary to the food item
        food_item_dict["summary"] = summary
        result_list.append(food_item_dict)
        summary_list.append(summary)

    return result_list, summary_list


def chunkify_list(summary_list) -> list:
    """
    Chunkify the list of summaries.

    Parameters:
    - summary_list (list): The list of summaries.

    Returns:
    - list: The list of chunks.
    """
    chunk_list = []
    for idx, current_item_summary in enumerate(summary_list):
        # Get the previous and next summaries
        if idx == 0:
            previous_item = ""
        else:
            previous_item = summary_list[idx - 1]

        if idx == len(summary_list) - 1:
            next_item = ""
        else:
            next_item = summary_list[idx + 1]

        # Create the chunk
        chunk = f"{previous_item}\n\n{current_item_summary}\n\n{next_item}"
        chunk = chunk.strip()
        chunk_list.append(chunk)

    return chunk_list


def store_word_embeddings(chunk_list: list[str]):
    """
    Store the word embeddings for the chunks in the database.

    Parameters:
    - chunk_list (list[str]): The list of chunks.
    """
    id_list = []
    embedding_list = []
    print("Generating embeddings for all chunks")
    for chunk in tqdm(chunk_list):
        # Generate the embedding for the chunk
        embedding = generate_embedding_for_string(chunk)
        embedding_list.append(embedding)

        # Generate the hash for the chunk
        chunk_hash = str(hash(chunk))
        id_list.append(chunk_hash)
    print("Generated embeddings for all chunks, storing in DB")

    # Store the embeddings and associated chunks in the database
    insert_embeddings_to_db(id_list, chunk_list, embedding_list)


def main():
    """
    Main function to run the script.
    """
    # Get the filepaths for the data
    filepath_list = get_data_filepaths("data/raw")

    # Get the dataset
    dataset = get_dataset(filepath_list)

    # Clean the dirty entries in the dataset
    clean_dataset = clean_dirty_entries(dataset)

    # Generate summaries for the food items in the clean dataset
    food_items, summary_list = generate_summaries_for_food_items(clean_dataset)

    # Chunkify the list of summaries
    chunks = summary_list + chunkify_list(summary_list)

    # Store the word embeddings for the chunks in the database
    store_word_embeddings(chunks)


if __name__ == "__main__":
    main()
