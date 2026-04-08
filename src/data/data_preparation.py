import pandas as pd
import os

# Configuration
DATA_PATH = os.getenv("CRIME_DATA_PATH", "data/Crimes_in_india_2001-2013.csv")
OUTPUT_PATH = "cleaned_crime_features.csv"
TOP_CRIMES = ["THEFT", "HURT/GREVIOUS HURT", "OTHER THEFT", "AUTO THEFT", "BURGLARY"]

def normalize_state(s):
    if not s: return ""
    # Standardize to Title Case, remove extra spaces, and unify variants
    s = s.upper().replace("&", " & ").replace("  ", " ").strip()
    if "DELHI" in s: return "Delhi"
    if "A & N" in s: return "A & N Islands"
    if "D & N" in s: return "D & N Haveli"
    if "DAMAN" in s: return "Daman & Diu"
    if "JAMMU" in s: return "Jammu and Kashmir"
    if "ODISHA" in s or "ORISSA" in s: return "Orissa"
    if "UTTARAKHAND" in s or "UTTARANCHAL" in s: return "Uttaranchal"
    # Mapping certain standard casing
    mapping = {"UTTAR PRADESH": "Uttar Pradesh", "MADHYA PRADESH": "Madhya Pradesh", "ANDHRA PRADESH": "Andhra Pradesh", 
               "WEST BENGAL": "West Bengal", "HIMACHAL PRADESH": "Himachal Pradesh", "ARUNACHAL PRADESH": "Arunachal Pradesh",
               "JHARKHAND": "Jharkhand", "CHHATTISGARH": "Chhattisgarh"}
    return mapping.get(s, s.title())

def preprocess(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} missing."); return False

    print(f"Crunching {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 1. Clean: Filter out summary rows and normalize names
    df = df[~df['DISTRICT'].str.contains('TOTAL', case=False, na=False)]
    df['STATE/UT'] = df['STATE/UT'].apply(normalize_state)
    
    # 2. Handle long-format CSV (crime_type + crime_count columns)
    #    Pivot to wide format: one column per crime type
    if 'crime_type' in df.columns and 'crime_count' in df.columns:
        df['crime_type'] = df['crime_type'].str.upper().str.strip()
        # Filter to only the top crime types
        df_filtered = df[df['crime_type'].isin(TOP_CRIMES)]
        # Pivot: rows = (STATE/UT, DISTRICT), columns = crime_type, values = crime_count
        features = df_filtered.pivot_table(
            index=['STATE/UT', 'DISTRICT'],
            columns='crime_type',
            values='crime_count',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        # Ensure all TOP_CRIMES columns exist (fill missing with 0)
        for crime in TOP_CRIMES:
            if crime not in features.columns:
                features[crime] = 0
        features = features[['STATE/UT', 'DISTRICT'] + TOP_CRIMES]
    else:
        # Wide-format CSV: columns already match crime names
        cols = ['STATE/UT', 'DISTRICT'] + TOP_CRIMES
        features = df[cols].groupby(['STATE/UT', 'DISTRICT']).sum().reset_index()
    
    print(f"Exporting cleaned matrix ({len(features)} districts) to {output_csv}")
    features.to_csv(output_csv, index=False)
    return True

if __name__ == "__main__":
    preprocess(DATA_PATH, OUTPUT_PATH)
