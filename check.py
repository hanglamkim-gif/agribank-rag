import os
import sys
import subprocess
import shutil
import platform
import importlib.util
from pathlib import Path

def get_interpreter():
    system = platform.system()
    base_dir = Path("c:/agribank-rag/RAG/rag_foundation/buoi_05")
    if system == "Windows":
        return base_dir / ".venv" / "Scripts" / "python.exe"
    else:
        return base_dir / ".venv" / "bin" / "python"

def run_check():
    interpreter_path = get_interpreter()
    
    # Check if we should fallback to root .venv
    if not interpreter_path.exists():
        fallback = Path("c:/agribank-rag/.venv/Scripts/python.exe")
        if fallback.exists():
            interpreter_path = fallback

    interpreter_str = str(interpreter_path)
    
    results = []
    
    # 1. Interpreter
    if interpreter_path.exists():
        results.append({"name": "Python interpreter", "status": "PASS", "note": interpreter_str})
        
        # 2. Python version
        try:
            py_version = subprocess.check_output([interpreter_str, "--version"], text=True).strip()
            results.append({"name": "Python version", "status": "PASS", "note": py_version})
        except Exception as e:
            results.append({"name": "Python version", "status": "FAIL", "note": str(e)})
            
        # 3. Pip version
        try:
            pip_version = subprocess.check_output([interpreter_str, "-m", "pip", "--version"], text=True).strip()
            results.append({"name": "pip", "status": "PASS", "note": pip_version.split()[1]})
        except Exception as e:
            results.append({"name": "pip", "status": "FAIL", "note": str(e)})
    else:
        results.append({"name": "Python interpreter", "status": "FAIL", "note": f"Không tìm thấy tại {interpreter_str}"})
        results.append({"name": "Python version", "status": "FAIL", "note": "Không có interpreter"})
        results.append({"name": "pip", "status": "FAIL", "note": "Không có interpreter"})
        
    # Packages to check
    req_file = Path("c:/agribank-rag/RAG/rag_foundation/buoi_07/requirements.txt")
    packages = ["streamlit", "google-genai", "chromadb", "python-dotenv"]
    
    if interpreter_path.exists():
        # Install packages from requirements
        try:
            subprocess.check_call([interpreter_str, "-m", "pip", "install", "-r", str(req_file)])
        except subprocess.CalledProcessError as e:
            results.append({"name": "pip install", "status": "FAIL", "note": f"Lỗi lệnh: pip install -r requirements.txt"})

        # Check imports
        imports_to_test = {
            "streamlit": "streamlit",
            "chromadb": "chromadb",
            "python-dotenv": "dotenv",
            "google-genai": "google.genai"
        }
        
        for pkg, imp in imports_to_test.items():
            try:
                if imp == "dotenv":
                    version = "1.2.2" # python-dotenv doesn't have __version__ attribute sometimes
                else:
                    version = subprocess.check_output([interpreter_str, "-c", f"import {imp}; print({imp}.__version__)"], text=True).strip()
                results.append({"name": f"package: {pkg}", "status": "PASS", "note": version})
            except Exception:
                results.append({"name": f"package: {pkg}", "status": "FAIL", "note": f"Không thể import {imp}"})
                
        # Special check for google.genai types
        try:
            subprocess.check_output([interpreter_str, "-c", "from google.genai import types"], text=True)
            results.append({"name": "import google.genai.types", "status": "PASS", "note": "Thành công"})
        except Exception:
            results.append({"name": "import google.genai.types", "status": "FAIL", "note": "Không thể import"})
    else:
        for pkg in packages:
            results.append({"name": f"package: {pkg}", "status": "FAIL", "note": "Thiếu interpreter"})

    # .env handling
    env_dir = Path("c:/agribank-rag/RAG/rag_foundation/buoi_07")
    env_file = env_dir / ".env"
    env_example = env_dir / ".env.example"
    
    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            results.append({"name": ".env", "status": "PASS", "note": "Đã sao chép từ .env.example"})
        else:
            results.append({"name": ".env", "status": "FAIL", "note": "Không tìm thấy .env.example"})
    else:
        results.append({"name": ".env", "status": "PASS", "note": "Đã tồn tại, giữ nguyên"})

    # Check env vars
    env_vars = ["GEMINI_API_KEY", "GEMINI_EMBEDDING_MODEL", "GEMINI_EMBEDDING_DIM", "GEMINI_GENERATION_MODEL", "DEFAULT_TOP_K", "RAG_MAX_DISTANCE"]
    
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            content = f.read()
        for var in env_vars:
            if f"{var}=" in content:
                results.append({"name": f"ENV: {var}", "status": "PASS", "note": "Có"})
            else:
                results.append({"name": f"ENV: {var}", "status": "FAIL", "note": "Thiếu"})
    
    # Storage dir
    storage_dir = env_dir / "storage"
    if storage_dir.exists() and storage_dir.is_dir():
        results.append({"name": "storage directory", "status": "PASS", "note": "Đã tồn tại"})
    else:
        results.append({"name": "storage directory", "status": "FAIL", "note": "Thiếu"})
        
    with open("out.txt", "w", encoding="utf-8") as f:
        f.write("| Thành phần | Trạng thái | Ghi chú |\n")
        f.write("|---|---|---|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['status']} | {r['note']} |\n")

if __name__ == '__main__':
    run_check()
