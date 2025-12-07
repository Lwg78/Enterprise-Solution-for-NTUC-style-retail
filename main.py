import pandas as pd
import sys
import os

# Ensure python can find your src folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_cleaner import clean_and_merge_data
from src.preprocessor import preprocess_data
from src.ArmstrongCycleTransformer import ArmstrongCycleTransformer # Make sure this file is in src/

# Scikit-Learn Imports
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def main():
    print("🚀 STARTING SALES PREDICTION PIPELINE")
    print("=======================================")

    # 1. CLEAN
    print("\n[Step 1] Cleaning Data...")
    raw_df = clean_and_merge_data()
    if raw_df is None: return

    # 2. PREPROCESS (Aggregate to Daily Sales)
    print("\n[Step 2] Preprocessing...")
    daily_df = preprocess_data(df=raw_df)
    
    # ... inside main() ...
    
    # TOGGLE THIS: True = Simulation, False = Real Data
    USE_SYNTHETIC = True 

    if USE_SYNTHETIC:
        print("\n[Step 1] Loading Synthetic Data (Simulation Mode)...")
        from src.datagen import generate_synthetic_data
        generate_synthetic_data()
        
        syn_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'synthetic_sales.csv')
        daily_df = pd.read_csv(syn_path)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
    else: # <--- MAKE SURE THIS 'ELSE' IS HERE
        print("\n[Step 1] Cleaning Real Data...")
        raw_df = clean_and_merge_data()
        if raw_df is None: return
        
        print("\n[Step 2] Preprocessing Real Data...")
        daily_df = preprocess_data(df=raw_df)

    # ... Proceed to Step 3 ...

    # 3. DEFINE PIPELINE
    print("\n[Step 3] Building Armstrong Pipeline...")
    
    # Define X (Date) and y (Sales Amount)
    # Note: Models need numbers, not dates. We pass the DataFrame to ArmstrongTransformer,
    # which will extract the date features.
    X = daily_df[['date', 'sales_amt']] 
    y = daily_df['sales_amt']

    # Split Data (Shuffle=False is MANDATORY for Time Series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model_pipeline = Pipeline([
        # Generate Cycle Features (Pi Wave, Quarter Wave)
        # Note: We use 'sales_amt' as the target column now!
        ('armstrong', ArmstrongCycleTransformer(target_col='sales_amt')),
        
        # Scale features
        ('scaler', StandardScaler()),
        
        # Predict
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # 4. TRAIN
    print(f"   -> Training on {len(X_train)} days...")
    model_pipeline.fit(X_train, y_train)

    # 5. EVALUATE
    print("\n[Step 4] Evaluation...")
    predictions = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"   -> Mean Absolute Error: ${mae:.2f}")
    
    # Simple "Accuracy" check
    avg_sales = y_test.mean()
    print(f"   -> Average Daily Sales: ${avg_sales:.2f}")
    print(f"   -> Error Percentage: {(mae/avg_sales)*100:.1f}%")

    print("\n✅ Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()
