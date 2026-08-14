import re

rag_path = r"c:\agribank-rag\RAG\rag_foundation\buoi_07\rag.py"
with open(rag_path, "r", encoding="utf-8") as f:
    code = f.read()

# Comment out prints
code = re.sub(r'(\s+)print\(', r'\1pass # print(', code)

with open(rag_path, "w", encoding="utf-8") as f:
    f.write(code)
print("Prints commented out in rag.py")
