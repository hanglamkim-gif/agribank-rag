import sys
import glob

for f in glob.glob(r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests\*.py"):
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Prepend sys.stdout reconfiguration
    prefix = "import sys\nif hasattr(sys.stdout, 'reconfigure'):\n    sys.stdout.reconfigure(encoding='utf-8')\nif hasattr(sys.stderr, 'reconfigure'):\n    sys.stderr.reconfigure(encoding='utf-8')\n"
    if "sys.stdout.reconfigure" not in content:
        content = prefix + content
        with open(f, "w", encoding="utf-8") as file:
            file.write(content)
print("Tests patched for utf-8.")
