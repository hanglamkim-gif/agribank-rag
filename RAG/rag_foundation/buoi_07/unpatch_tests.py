import sys
import glob

for f in glob.glob(r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests\*.py"):
    with open(f, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Remove reconfigure lines
    with open(f, "w", encoding="utf-8") as file:
        for line in lines:
            if "reconfigure" not in line and not line.startswith("import sys\n"):
                file.write(line)
print("Removed reconfigure lines.")
