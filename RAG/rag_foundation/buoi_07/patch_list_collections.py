import re

test_files = [r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests\test_index.py", r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests\test_query.py"]

for f in test_files:
    with open(f, "r", encoding="utf-8") as file:
        code = file.read()
        
    # Fix list_collections mock
    code = code.replace('"nhnn-semantic-128-4e8c130d"', 'get_collection_name("semantic", 128, "mock-emb-model")')
    
    with open(f, "w", encoding="utf-8") as file:
        file.write(code)
print("Patched list_collections name.")
