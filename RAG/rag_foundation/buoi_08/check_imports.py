import sys

def check_imports():
    results = []
    
    try:
        import streamlit
        results.append(f"streamlit: {streamlit.__version__}")
    except ImportError as e:
        results.append(f"streamlit: ERROR - {e}")
        
    try:
        from google import genai
        results.append(f"google-genai: {genai.__version__}")
    except ImportError as e:
        results.append(f"google-genai: ERROR - {e}")
        
    try:
        import chromadb
        results.append(f"chromadb: {chromadb.__version__}")
    except ImportError as e:
        results.append(f"chromadb: ERROR - {e}")
        
    try:
        import dotenv
        # python-dotenv may not have __version__ reliably, check if it works
        results.append(f"python-dotenv: OK")
    except ImportError as e:
        results.append(f"python-dotenv: ERROR - {e}")
        
    try:
        import rank_bm25
        results.append(f"rank-bm25: OK")
    except ImportError as e:
        results.append(f"rank-bm25: ERROR - {e}")
        
    try:
        import transformers
        results.append(f"transformers: {transformers.__version__}")
    except ImportError as e:
        results.append(f"transformers: ERROR - {e}")
        
    try:
        import torch
        results.append(f"torch: {torch.__version__}")
    except ImportError as e:
        results.append(f"torch: ERROR - {e}")

    for r in results:
        print(r)

if __name__ == '__main__':
    check_imports()
