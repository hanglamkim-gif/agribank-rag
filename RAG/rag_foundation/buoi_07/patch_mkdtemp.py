import sys
import glob

for f in [r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests\test_index.py", r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests\test_query.py"]:
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Replace TemporaryDirectory with mkdtemp
    content = content.replace("self.temp_dir = tempfile.TemporaryDirectory()", "self.temp_dir = tempfile.mkdtemp()")
    content = content.replace("Path(self.temp_dir.name)", "Path(self.temp_dir)")
    content = content.replace("self.temp_dir.cleanup()", "pass")
    
    with open(f, "w", encoding="utf-8") as file:
        file.write(content)
print("Patched mkdtemp.")
