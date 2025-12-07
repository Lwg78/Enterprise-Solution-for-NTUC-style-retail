import pandas as pd
import numpy as np
import os

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'synthetic_sales.csv')

def generate_synthetic_data():
    print("🧪 Generating Synthetic Data (5 Years)...")
    
    # 1. Create a date range (2015 to 2020)
    dates = pd.date_range(start='2015-01-01', end='2020-01-01', freq='D')
    df = pd.DataFrame({'date': dates})
    
    # 2. Inject "Seasonality" (Sales go up in Dec, down in Jan)
    # Sine wave to simulate yearly trends
    df['seasonality'] = np.sin(2 * np.pi * df.index / 365) * 5000
    
    # 3. Inject "Trend" (Sales slowly grow over years)
    df['trend'] = df.index * 10
    
    # 4. Inject "Noise" (Random days are good/bad)
    df['noise'] = np.random.normal(0, 2000, len(df))
    
    # 5. Calculate Final Sales
    base_sales = 20000
    df['sales_amt'] = base_sales + df['seasonality'] + df['trend'] + df['noise']
    
    # Ensure no negative sales
    df['sales_amt'] = df['sales_amt'].clip(lower=1000)
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Generated {len(df)} rows of synthetic data at:")
    print(f"   {OUTPUT_FILE}")
    print(f"   (Range: {df['date'].min()} to {df['date'].max()})")

if __name__ == "__main__":
    generate_synthetic_data()