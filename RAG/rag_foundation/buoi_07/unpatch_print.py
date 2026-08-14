import re

rag_path = r"c:\agribank-rag\RAG\rag_foundation\buoi_07\rag.py"
with open(rag_path, "r", encoding="utf-8") as f:
    code = f.read()

# Restore prints
code = re.sub(r'(\s+)pass # print\(', r'\1print(', code)

with open(rag_path, "w", encoding="utf-8") as f:
    f.write(code)
print("Prints restored in rag.py")
