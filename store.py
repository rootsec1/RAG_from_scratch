from tqdm import tqdm

# Local
from setup import *
from constants import *
from util import insert_embeddings_to_db, generate_embedding_for_string


def clean_dirty_entries(dataset: list) -> list:
    clean_dataset = []
    for food_item_dict in dataset:
        if "options" in food_item_dict:
            clean_dataset.extend(
                process_dirty_entry(food_item_dict)
            )
        elif "category" in food_item_dict:
            clean_dataset.extend(
                process_dirty_entry(food_item_dict)
            )
        else:
            clean_dataset.append(food_item_dict)
    return clean_dataset


def generate_summaries_for_food_items(clean_dataset):
    result_list = []
    summary_list = []
    for food_item_dict in clean_dataset:
        name = food_item_dict.get("name", "").lower()
        description = food_item_dict.get("description", "").lower()

        if isinstance(food_item_dict["price"], dict):
            price = " ".join([
                str(x)
                for x in food_item_dict["price"].values()
            ]).lower()
        else:
            price = food_item_dict.get("price", "").lower()

        restaurant = food_item_dict.get("restaurant", "").lower()
        type = food_item_dict.get("type", "")

        if type is None or type == "":
            type = "Vegetarian"

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

        summary = f"Title: {name} - {restaurant}\n"
        summary += f"Description: {description}\n"
        summary += f"Price: ${price}\n"
        summary += f"Type: {type}\n"
        summary += f"Restaurant: {restaurant}\n"
        summary += f"Tags: {', '.join(tags)}\n"

        food_item_dict["summary"] = summary
        result_list.append(food_item_dict)
        summary_list.append(summary)

    return result_list, summary_list


def chunkify_list(summary_list) -> list:
    # Add next summary and previous summary to each summary to create the chunk

    chunk_list = []
    for idx, current_item_summary in enumerate(summary_list):
        if idx == 0:
            previous_item = ""
        else:
            previous_item = summary_list[idx - 1]

        if idx == len(summary_list) - 1:
            next_item = ""
        else:
            next_item = summary_list[idx + 1]

        chunk = f"{previous_item}\n\n{current_item_summary}\n\n{next_item}"
        chunk = chunk.strip()
        chunk_list.append(chunk)

    return chunk_list


def store_word_embeddings(chunk_list: list[str]):
    id_list = []
    embedding_list = []
    print("Generating embeddings for all chunks")
    for chunk in tqdm(chunk_list):
        embedding = generate_embedding_for_string(chunk)
        embedding_list.append(embedding)

        chunk_hash = str(hash(chunk))
        id_list.append(chunk_hash)
    print("Generated embeddings for all chunks, storing in DB")

    # Store embeddings and associated chunks in SQLite database with columns: hash (unique) bigint, vector text
    insert_embeddings_to_db(id_list, chunk_list, embedding_list)


def main():
    filepath_list = get_data_filepaths("data/raw")
    dataset = get_dataset(filepath_list)
    clean_dataset = clean_dirty_entries(dataset)
    food_items, summary_list = generate_summaries_for_food_items(clean_dataset)
    chunks = summary_list + chunkify_list(summary_list)
    store_word_embeddings(chunks)


if __name__ == "__main__":
    main()
