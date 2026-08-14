import sys
import glob

for f in [r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests\test_index.py", r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests\test_query.py"]:
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Add clear_system_cache to tearDown
    teardown_old = "def tearDown(self):\n        pass"
    teardown_new = "def tearDown(self):\n        import chromadb.api.client\n        chromadb.api.client.SharedSystemClient.clear_system_cache()\n        pass"
    content = content.replace(teardown_old, teardown_new)
    
    with open(f, "w", encoding="utf-8") as file:
        file.write(content)
print("Patched chromadb cache clear.")
