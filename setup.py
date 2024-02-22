import os
import json


def get_data_filepaths(food_menu_data_path: str) -> list[str]:
    filepaths = []
    file_list = os.listdir(food_menu_data_path)
    for file in file_list:
        if file.endswith(".json"):
            filepaths.append(os.path.join(food_menu_data_path, file))
    return filepaths


def get_dataset(filepaths: list[str]) -> list[dict]:
    dataset = []
    for file in filepaths:
        with open(file, "r") as f:
            data = json.load(f)
            for item in data:
                item["restaurant"] = file.replace(
                    ".json",
                    ""
                ).split("/")[-1]
                dataset.append(item)
    return dataset


def process_dirty_entry(food_item_dict: dict) -> list[dict]:
    resultant_list = []
    if "options" in food_item_dict:
        for option in food_item_dict["options"]:
            if "item" in option and "price" in option:
                new_item_dict = {
                    "name": option.get("item"),
                    "price": option.get("price"),
                    "restaurant": food_item_dict.get("restaurant"),
                    "type": food_item_dict.get("type")
                }
                resultant_list.append(new_item_dict)

    elif "category" in food_item_dict:
        if "items" in food_item_dict:
            for item in food_item_dict["items"]:
                if "name" in item and "price" in item and "type" in item:
                    new_item_dict = {
                        "name": item.get("name"),
                        "price": item.get("price"),
                        "type": item.get("category"),
                        "restaurant": food_item_dict.get("restaurant")
                    }
                    resultant_list.append(new_item_dict)

    elif "price" in food_item_dict and isinstance(food_item_dict["price"], dict):
        price_dict = food_item_dict["price"]
        for key, value in price_dict.items():
            new_item_dict = {
                "name": f"{food_item_dict.get('name')} {key}",
                "price": value,
                "type": food_item_dict.get("type", "Vegetarian"),
                "restaurant": food_item_dict.get("restaurant")
            }
            resultant_list.append(new_item_dict)

    else:
        resultant_list.append(food_item_dict)

    for idx, val in enumerate(resultant_list):
        if "description" not in val:
            resultant_list[idx]["description"] = ""
    return resultant_list
