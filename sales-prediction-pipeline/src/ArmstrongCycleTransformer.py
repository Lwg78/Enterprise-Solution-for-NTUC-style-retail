import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

class ArmstrongCycleTransformer(BaseEstimator, TransformerMixin):
    """
    A Scikit-Learn compatible transformer that generates features based on 
    Martin Armstrong's Economic Confidence Model (ECM).
    
    Parameters:
    -----------
    cycle_start_date : str
        A known 'turning point' date (YYYY-MM-DD) to align the Pi Cycle.
        Armstrong often cites 1985.65 or similar major peaks.
    target_col : str, optional (default='close')
        The name of the column containing the price or sales data needed for 
        the 'socrates_reversal' feature.
    reversal_window : int, optional (default=20)
        The rolling window size for the moving average used in the reversal logic.
    """
    def __init__(self, cycle_start_date='2015-10-01', target_col='close', reversal_window=20):
        self.cycle_start_date = pd.to_datetime(cycle_start_date)
        self.target_col = target_col
        self.reversal_window = reversal_window
        self.pi_cycle_days = 3141  # 8.6 years approx
        self.quarter_cycle_days = 785 # 2.15 years approx
        
    def fit(self, X, y=None):
        # Transformers generally don't need to 'learn' from data in fit(), 
        # but we adhere to the API.
        return self

    def transform(self, X):
        """
        X must be a DataFrame with a 'date' column.
        """
        X_copy = X.copy()
        
        # Ensure date is datetime
        if 'date' not in X_copy.columns:
            raise ValueError("Input DataFrame must have a 'date' column.")
        
        X_copy['date'] = pd.to_datetime(X_copy['date'])
        
        # Calculate days since cycle start
        days_diff = (X_copy['date'] - self.cycle_start_date).dt.days
        
        # --- FEATURE 1: The Macro Pi Cycle (8.6 Years) ---
        X_copy['ecm_pi_wave'] = np.sin(2 * np.pi * days_diff / self.pi_cycle_days)
        
        # --- FEATURE 2: The Quarter Cycle (2.15 Years) ---
        X_copy['ecm_quarter_wave'] = np.sin(2 * np.pi * days_diff / self.quarter_cycle_days)

        # --- FEATURE 3: The Business Cycle (1 Year) --- NEW! 🌟
        # Most sales data follows a 365.25 day pattern.
        X_copy['business_yearly_wave'] = np.sin(2 * np.pi * days_diff / 365.25)
        
        # --- FEATURE 4: Reversal Logic (The 1% Rule) ---
        if self.target_col not in X_copy.columns:
            # Return features if target is missing
            return X_copy[['ecm_pi_wave', 'ecm_quarter_wave', 'business_yearly_wave']]

        # Calculate rolling mean for reversal detection
        rolling_mean = X_copy[self.target_col].rolling(window=self.reversal_window).mean()
        
        X_copy['socrates_reversal'] = 0
        # 1 = Bullish Breakout, -1 = Bearish Reversal
        # We use fillna(0) to handle the first few rows where rolling_mean is NaN
        X_copy.loc[X_copy[self.target_col] > rolling_mean * 1.01, 'socrates_reversal'] = 1
        X_copy.loc[X_copy[self.target_col] < rolling_mean * 0.99, 'socrates_reversal'] = -1
        X_copy['socrates_reversal'] = X_copy['socrates_reversal'].fillna(0)
        
        # Return ALL features including the new yearly wave
        return X_copy[['ecm_pi_wave', 'ecm_quarter_wave', 'business_yearly_wave', 'socrates_reversal']]

# --- EXAMPLE USAGE IN A PIPELINE ---

import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Import the new pipeline functions for a self-contained example
from data_cleaner import clean_and_merge_data
from preprocessor import preprocess_data

if __name__ == '__main__':
    print("🚀 RUNNING ARMSTRONG TRANSFORMER STANDALONE EXAMPLE")
    print("=====================================================")
    # 1. CLEAN & PREPROCESS (Using the new pipeline functions)
    print("\n[Step 1] Cleaning & Preprocessing Data...")
    raw_df = clean_and_merge_data()
    if raw_df is None:
        print("Aborting example run.")
    else:
        daily_df = preprocess_data(df=raw_df)

        # 2. DEFINE THE PIPELINE
        print("\n[Step 2] Building Pipeline...")
        pipeline = Pipeline([
            ('armstrong_features', ArmstrongCycleTransformer(target_col='sales_amt')),
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # 3. FIT AND EVALUATE
        print("\n[Step 3] Training and Evaluating...")
        X = daily_df[['date', 'sales_amt']] 
        y = daily_df['sales_amt']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)

        print(f"\n✅ Standalone Example Finished. MAE: ${mae:.2f}")