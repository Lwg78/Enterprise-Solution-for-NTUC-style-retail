import pandas as pd
import os

# Define path (same logic as before)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'merged_sales.csv')

def inspect_data():
    if not os.path.exists(DATA_PATH):
        print("❌ Error: Processed data file not found. Run data_cleaner.py first.")
        return

    print(f"🕵️‍♂️ Inspecting: {DATA_PATH}\n")
    df = pd.read_csv(DATA_PATH)

    # 1. Show the first 3 rows
    print("--- HEAD (First 3 rows) ---")
    print(df.head(3))
    print("\n")

    # 2. Show Column Names & Types
    print("--- INFO (Columns & Types) ---")
    print(df.info())
    print("\n")

    # 3. Check for Missing Values
    print("--- MISSING VALUES ---")
    print(df.isnull().sum())
    print("\n")

if __name__ == "__main__":
    inspect_data()