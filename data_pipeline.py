import pandas as pd
import glob
import os

SELECTED_FEATURES = [
    'Duration', 'Rate', 'Entropy',
    'Payload_Length', 'Var_Payload',
    'syn_flag_number', 'ack_flag_number',
    'rst_flag_number', 'fin_flag_number',
    'TCP', 'UDP',
    'Number', 'Tot size',
    'AVG', 'Std', 'Min', 'Max'
]

def load_data(base_path="extracted"):
    # Look for files matching the pattern in the extracted directory
    files = glob.glob(f"{base_path}/**/new_feature_csv/**/*.csv", recursive=True)
    print(f"[INFO] Found {len(files)} CSV files")

    if not files:
        print(f"[ERROR] No CSV files found in {base_path}. Please check the extraction path.")
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Could not read {f}: {e}")

    if not dfs:
        return None
        
    df = pd.concat(dfs, ignore_index=True)
    print("[INFO] Raw shape:", df.shape)

    return df

def preprocess(df):
    if df is None:
        return None
        
    df.columns = df.columns.str.strip()

    # Binary label: If attack_type is not 'benign', it's an intrusion
    if "attack_type" in df.columns:
        df["label"] = (df["attack_type"] != "benign").astype(int)
    else:
        print("[ERROR] 'attack_type' column missing. Cannot create labels.")
        return None

    # Filter for selected features
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    if len(available_features) < len(SELECTED_FEATURES):
        missing = set(SELECTED_FEATURES) - set(available_features)
        print(f"[WARNING] Missing columns: {missing}")
        
    df = df[available_features + ["label"]]
    df = df.dropna()

    print("[INFO] Final shape:", df.shape)
    return df

def main():
    # Use the 'extracted' directory as the base
    df = load_data("A-DIDS/data/extracted")
    df = preprocess(df)

    if df is not None:
        output_file = "A-DIDS/data/drone_dataset.parquet"
        df.to_parquet(output_file, index=False)
        print(f"[INFO] Saved {output_file}")
    else:
        print("[ERROR] Data pipeline failed to generate dataset.")

if __name__ == "__main__":
    main()
