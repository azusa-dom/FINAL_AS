import pandas as pd
import argparse
import os

# --- This is the definitive script that reads the file as an Excel file ---
print("✅✅✅ Running the Definitive Excel Reader Script (v10) ✅✅✅")

def create_balanced_dataset(input_path, output_path):
    """
    Reads the file correctly as an Excel file and creates a clean, balanced dataset.
    """
    print(f"📥 Reading Excel file: {input_path}")
    try:
        # --- CORE FIX: Use pd.read_excel() to correctly read the .xlsx file ---
        # We also specify 'Sheet1' which is the default sheet name.
        df = pd.read_excel(input_path, sheet_name='Sheet1', engine='openpyxl')
        
    except FileNotFoundError:
        print(f"❌ ERROR: Raw data file not found at '{input_path}'!")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred while reading the Excel file: {e}")
        return

    print("✅ Excel file read successfully! Now cleaning and balancing the data.")

    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # Define the target variable
    target_disease_col = 'Disease'
    target_disease_name = 'Ankylosing_Spondylitis'
    label_col = 'label'

    if target_disease_col not in df.columns:
        print(f"❌ ERROR: Cannot find the '{target_disease_col}' column in the data.")
        print(f"   Available columns are: {df.columns.tolist()}")
        return

    print(f"🎯 Creating binary label...")
    df[label_col] = (df[target_disease_col] == target_disease_name).astype(int)

    # Balance the dataset
    df_positive = df[df[label_col] == 1]
    df_negative = df[df[label_col] == 0].sample(n=len(df_positive), random_state=42)
    df_balanced = pd.concat([df_positive, df_negative]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Define final columns, explicitly dropping the original 'Disease' column
    final_columns = [col for col in df.columns if col != target_disease_col]
    df_final = df_balanced[final_columns]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"⚖️ Data balancing complete!")
    print(f"✅ Successfully created a new, clean, balanced dataset: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a clean, balanced dataset from the raw source file.")
    parser.add_argument(
        "--input",
        default="data/raw_data.csv", # Keep this as is, we know it's an Excel file in disguise
        help="Path to the original raw data file."
    )
    parser.add_argument(
        "--output",
        default="data/balanced_clinical.csv",
        help="Path for the clean, balanced output file."
    )
    args = parser.parse_args()
    create_balanced_dataset(args.input, args.output)