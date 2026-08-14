import re

rag_path = r"c:\agribank-rag\RAG\rag_foundation\buoi_07\rag.py"
with open(rag_path, "r", encoding="utf-8") as f:
    code = f.read()

# Replace print( with safe_print(
# except for the def safe_print line itself
# and except inside safe_print function
code = re.sub(r'(?<!def )(?<!\b_)print\(', r'safe_print(', code)

with open(rag_path, "w", encoding="utf-8") as f:
    f.write(code)
print("Replaced prints with safe_print")
