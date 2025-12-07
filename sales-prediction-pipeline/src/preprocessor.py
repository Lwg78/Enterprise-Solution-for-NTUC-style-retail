import pandas as pd
import os

def preprocess_data(df=None, filepath=None):
    """
    Loads the cleaned CSV, handles final missing values, and ensures 
    dates are sorted for time-series analysis.
    """
    # 1. Load Data (if not provided directly)
    if df is None:
        if filepath and os.path.exists(filepath):
            df = pd.read_csv(filepath)
        else:
            raise ValueError("❌ No data provided to preprocessor!")

    print(f"⚙️ Preprocessing {len(df)} rows...")

    # 2. Drop that single missing row (sup_id, description, etc.)
    original_len = len(df)
    df.dropna(subset=['sup_id', 'description'], inplace=True)
    if len(df) < original_len:
        print(f"   -> Dropped {original_len - len(df)} rows with missing IDs/Descriptions.")

    # 3. Date Conversion & Sorting (CRITICAL for Armstrong Cycle)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 4. Feature Selection
    # We only need Date and Sales for the Armstrong Model
    # We aggregate sales by date (in case multiple items sold on same day)
    # This creates a daily sales signal.
    daily_sales = df.groupby('date')['sales_amt'].sum().reset_index()
    
    print(f"✅ Data Ready: {len(daily_sales)} daily records from {daily_sales['date'].min().date()} to {daily_sales['date'].max().date()}")
    
    return daily_sales

if __name__ == "__main__":
    # Test run
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'merged_sales.csv')
    preprocess_data(filepath=path)