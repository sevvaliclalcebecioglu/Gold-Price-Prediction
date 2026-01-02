import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned_gold_price_data.csv"
)
# Path to the cleaned dataset

TARGET = "USD (PM)" 
# Target variable for prediction

TEST_SIZE = 0.2
# Proportion of the dataset to include in the test split

RANDOM_STATE = 42
# Random seed for reproducibility



