import pandas as pd
import os
import glob

# --- Define paths relative to the project root ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DEFAULT_INPUT_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'raw')
DEFAULT_OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'merged_sales.csv')

def find_header_row(file_path):
    """
    Scans the first 10 rows of an Excel file to find the row that contains 'date' or 'sku'.
    Returns the row index (0-based) to use as header.
    """
    try:
        # Read first 10 rows without a header
        df_preview = pd.read_excel(file_path, header=None, nrows=10)
        
        for idx, row in df_preview.iterrows():
            # Convert row to string and check for keywords
            row_str = row.astype(str).str.lower().tolist()
            if any('date' in x for x in row_str) and any('sku' in x for x in row_str):
                return idx
    except Exception:
        pass
    return 0 # Default to first row if detection fails

def clean_and_merge_data(input_folder=DEFAULT_INPUT_FOLDER, output_file=DEFAULT_OUTPUT_FILE):
    all_data_frames = []
    
    print(f"🕵️‍♂️ Looking for files in: {input_folder}")
    
    files = []
    for ext in ['*.xlsx', '*.xls']:
        files.extend(glob.glob(os.path.join(input_folder, ext)))
    files = [f for f in files if not os.path.basename(f).startswith('~')]
    
    print(f"🔎 Found {len(files)} files to process...")

    for file in files:
        try:
            # 1. SMART HEADER DETECTION
            header_row = find_header_row(file)
            
            # Read the file with the correct header
            df = pd.read_excel(file, header=header_row)
            
            # 2. STANDARDISATION
            # Clean column names: remove spaces, lowercase
            df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
            
            # 3. VERIFICATION
            # If 'date' column is missing after reading, skip this file or warn
            if 'date' not in df.columns:
                print(f"⚠️ Warning: Could not find 'date' column in {os.path.basename(file)}. Columns found: {df.columns.tolist()}")
                continue

            # Add source file for debugging
            df['source_file'] = os.path.basename(file)
            
            # Select only the columns we expect (prevents Unnamed columns)
            # We keep 'sales_amt($)' but let's rename it to something python-friendly
            df.rename(columns={'sales_amt($)': 'sales_amt'}, inplace=True)
            
            all_data_frames.append(df)
            print(f"✅ Loaded: {os.path.basename(file)} (Header at row {header_row})")
            
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    # 4. Compile and Export
    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Drop rows where 'date' is still NaN (just in case)
        final_df.dropna(subset=['date'], inplace=True)
        
        # Filter out "Unnamed" columns if any snuck in
        final_df = final_df.loc[:, ~final_df.columns.str.contains('^unnamed')]
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_df.to_csv(output_file, index=False)
        print(f"🎉 Success! Merged {final_df.shape[0]} CLEAN rows into {output_file}")
        return final_df
    else:
        print("⚠️ No data found to merge.")
        return None

if __name__ == "__main__":
    clean_and_merge_data()