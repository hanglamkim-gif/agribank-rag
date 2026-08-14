import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import rag

print("============================================================")
print("1. BAT DAU INDEX DU LIEU VAO CHROMADB...")
try:
    idx_res = rag.run_index(strategy="hierarchical", reset=True)
    print(f">> Trang thai index: {idx_res.get('status')}")
    print(f">> Collection: {idx_res.get('collection_name')}")
    print(f">> So chunk da index: {idx_res.get('indexed_chunks')}")
except Exception as e:
    print(f"Loi khi Index: {e}")

print("\n============================================================")
print("2. THUC HIEN TRA CUU (QUERY)...")
question = "Co cau lai thoi han tra no duoc quy dinh nhu the nao?"
print(f"Cau hoi: {question}\n")
try:
    res = rag.run_query(question=question, strategy="hierarchical", top_k=5)
    print(f"Trang thai: {res.get('status')}")
    print(f"\n--- CAU TRA LOI ---\n{res.get('answer')}\n")
    print(f"--- TRICH DAN NGUON ({len(res.get('citations', []))}) ---")
    for c in res.get("citations", []):
        print(f" * {c.get('display')}")
except Exception as e:
    print(f"Loi khi Query: {e}")
print("============================================================")
