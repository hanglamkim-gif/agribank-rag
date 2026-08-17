import csv
import os
from collections import Counter

def inspect_csv(filepath):
    print(f"--- Inspection for {os.path.basename(filepath)} ---")
    if not os.path.exists(filepath):
        print(f"[MISSING] File not found: {filepath}")
        return None
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers:
                print("File is empty.")
                return None
                
            print(f"Columns: {headers}")
            
            rows = []
            null_counts = {col: 0 for col in headers}
            
            for row in reader:
                # pad row if shorter than headers
                row = row + [''] * (len(headers) - len(row))
                rows.append(tuple(row))
                for i, val in enumerate(row):
                    if i < len(headers) and (val is None or str(val).strip() == ""):
                        null_counts[headers[i]] += 1
                        
            print(f"Number of rows: {len(rows)}")
            print("Null values count per column:")
            for col, count in null_counts.items():
                print(f"  {col}: {count}")
                
            duplicates = len(rows) - len(set(rows))
            print(f"Duplicate rows: {duplicates}")
            
            return {"headers": headers, "rows": rows}
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    data_dir = r"c:\agribank-rag\data"
    
    files = [
        "risk_profiles_seed.csv",
        "controls_seed.csv",
        "risk_events_seed.csv",
        "relationships_seed.csv"
    ]
    
    data = {}
    for f in files:
        path = os.path.join(data_dir, f)
        data[f] = inspect_csv(path)
        print("-" * 40)
        
    rel_data = data.get("relationships_seed.csv")
    if rel_data:
        headers = rel_data["headers"]
        if "relationship_type" in headers:
            idx = headers.index("relationship_type")
            types = Counter([row[idx] for row in rel_data["rows"]])
            print("\n--- Relationship Types ---")
            for t, count in types.items():
                print(f"  {t}: {count}")
                
    print("\n--- Summary ---")
    for f in files:
        if data.get(f) is None:
            print(f"Missing data file: {f}. Cannot determine primary/foreign keys or perform detailed graph analysis for this file.")

if __name__ == "__main__":
    main()
