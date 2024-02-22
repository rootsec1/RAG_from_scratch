import torch
import nltk

from nltk.corpus import stopwords

# Local
nltk.download("stopwords")

RESTAURANT_LIST = [
    "Sazon"
    "Sprout"
    "Ginger + soy"
    "The farmstead"
    "Tandoor"
    "Gyotaku"
    "JB roasts &. Chops"
    "Il forno"
    "Skillet"
    "CaFe"
    "Panera Bread"
]
DB_PATH = "data/processed/vectors.sqlite3"

ML_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LLM_SYSTEM_MESSAGE = """
You are a food expert and a tour guide who knows all the restaurant and their food menu inside Brodhead center at Duke University, Durham, NC.
You are responsible for providing information about the restaurants and their food menu to the customers.
You are also responsible for providing the best restaurant and food menu recommendations to the customers based on their preferences.
You are using the context I provide as a RAG prompt to generate the response.
"""

USER_SYSTEM_MESSAGE = """
Use the following content as context to answer my question regarding the food menus at Brodhead center at Duke University, Durham, NC:
{}

Question: {}
"""

STOPWORDS = set(stopwords.words("english"))
TOP_K = 50

REFERENCE_TEXT = """
Tandoor:
Vegetarian Combo: Includes three vegetable servings, rice, and two pieces of naan for $10.89.
Sada Dosai: A Vegan option priced at $8.59.
Vegetable Dosai: A Vegetarian dish for $10.29.
Garlic Naan: A Vegan option available for $3.79.
Vegetable Samosa: Another Vegan option priced at $4.99.
Mango Lassi: A refreshing Vegetarian drink for $3.99.

Cafe:
Strawberry Smoothie: A Vegetarian option with strawberries, banana, orange juice, and mixed berries sorbet.
Gelato: Various Vegetarian gelato flavors available.

Il_Forno:
Salads: Options like the Smoke House Chopped Salad, The Wedge, Classic Caesar, and Fennel and Apple salads are available for Vegetarians.
Sauces: Various sauce options suitable for Vegetarians.
Sandwiches: Non-Vegetarian options served with one side.

Ginger_and_Soy:
Edamame: A Vegetarian option.
Vegetable Dumplings: Another Vegetarian choice.
Shrimp Dumplings: Vegetarian option available.
BBQ Bun: Vegetarian dish priced at $19.
Fried Spring Roll: Vegetarian option for $15.
Vegetable Spring Roll: Included in catering set, a Vegetarian option.

Sazon:
Cilantro White Rice and Brown Rice: Vegan options.
Black Beans: Another Vegan option.
Grilled Portobello (VH): A marinated Vegetarian dish.
Guacamole and Queso: Vegetarian sides.

Sprouts:
Caesar Salads: A popular Vegetarian choice.
Avocado Toast: A Vegetarian dish priced at $6.75.
Cauliflower Wrap: Vegan option available for $8.50.
Veggie Hand Roll, Shrimp Tempura Hand Roll, Spicy Tuna Hand Roll: Vegetarian rolls available for $3.50.
"""
