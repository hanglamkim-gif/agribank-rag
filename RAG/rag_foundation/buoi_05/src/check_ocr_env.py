import os
import sys
from dotenv import load_dotenv

# Tải biến môi trường từ .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

def check_env():
    results = {}
    packages = {
        "fitz": "PyMuPDF",
        "PIL": "Pillow",
        "llama_cloud": "Llama Cloud",
        "pydantic": "Pydantic",
        "streamlit": "Streamlit",
        "dotenv": "python-dotenv"
    }
    
    for mod, name in packages.items():
        try:
            __import__(mod)
            results[name] = "PASS"
        except ImportError:
            results[name] = "FAIL"
            
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if api_key and api_key != "KEY_CỦA_BẠN":
        results["LLAMA_CLOUD_API_KEY"] = "PASS"
    else:
        results["LLAMA_CLOUD_API_KEY"] = "FAIL (Chưa cấu hình Key chuẩn)"

    print("\n=== BẢNG KIỂM TRA MÔI TRƯỜNG BUỔI 5 ===")
    for k, v in results.items():
        print(f"[{'PASS' if 'PASS' in v else 'FAIL'}] {k}: {v}")
    print("=======================================\n")

if __name__ == "__main__":
    check_env()
    