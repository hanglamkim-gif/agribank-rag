import glob

patch_code = """
import chromadb
from unittest.mock import patch

GLOBAL_CLIENT = chromadb.EphemeralClient()

"""

for f in [r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests\test_index.py", r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests\test_query.py"]:
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
    
    if "GLOBAL_CLIENT" not in content:
        # Add imports and global client at the top
        content = content.replace("import chromadb", patch_code)
        
        # In setUp, start patching
        setup_old = "def setUp(self):"
        setup_new = "def setUp(self):\n        self.patcher = patch('chromadb.PersistentClient', return_value=GLOBAL_CLIENT)\n        self.patcher.start()"
        content = content.replace(setup_old, setup_new)
        
        # In tearDown, stop patching and clear collections
        teardown_old = "pass"
        teardown_new = "self.patcher.stop()\n        for c in GLOBAL_CLIENT.list_collections():\n            GLOBAL_CLIENT.delete_collection(c.name)\n        pass"
        content = content.replace(teardown_old, teardown_new)
        
        with open(f, "w", encoding="utf-8") as file:
            file.write(content)

print("Patched to use GLOBAL_CLIENT.")
