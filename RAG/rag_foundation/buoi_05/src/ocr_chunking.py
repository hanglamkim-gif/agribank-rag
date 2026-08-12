import os
import re
import json
import asyncio
import unicodedata
import fitz  # PyMuPDF
from dotenv import load_dotenv
from llama_cloud import AsyncLlamaCloud

# Nạp API KEY từ file .env trong thư mục src
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATADEMO_DIR = os.path.join(BASE_DIR, "datademo")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_nfc(text: str) -> str:
    """Chuẩn hóa chuỗi tiếng Việt về dạng Unicode NFC"""
    return unicodedata.normalize('NFC', text)

async def extract_text_from_pdf(pdf_path: str):
    """Đọc Text layer bằng PyMuPDF, nếu lỗi/rỗng sẽ fallback sang LlamaParse OCR"""
    doc = fitz.open(pdf_path)
    full_text = ""
    ocr_used = False
    
    for page in doc:
        full_text += page.get_text()

    full_text = normalize_nfc(full_text.strip())

    # Kiểm tra nếu text rỗng hoặc bị lỗi font/encoding -> Chuyển sang OCR
    if not full_text or len(full_text) < 30:
        print("Text layer rỗng hoặc lỗi. Đang kích hoạt LlamaParse OCR...")
        ocr_used = True
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        
        client = AsyncLlamaCloud(api_key=api_key)
        file_obj = await client.files.create(file=pdf_path, purpose="parse")
        result = await client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",
            version='latest',
            expand=["markdown_full"]
        )
        full_text = normalize_nfc(result.markdown_full)
    else:
        print("Đọc thành công Text layer trực tiếp từ PyMuPDF (Không cần dùng OCR).")
        
    return full_text, ocr_used

def chunk_fixed_size(text: str, chunk_size=300, overlap=50):
    """Cắt chunk theo độ dài cố định có gối đầu (overlap)"""
    chunks = []
    start = 0
    idx = 1
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "chunk_id": f"fixed_{idx}",
            "strategy": "fixed-size",
            "text": chunk_text
        })
        start += (chunk_size - overlap)
        idx += 1
    return chunks

def chunk_semantic(text: str):
    """Phân đoạn theo ngữ nghĩa (ngắt đoạn, xuống dòng)"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for idx, p in enumerate(paragraphs, 1):
        chunks.append({
            "chunk_id": f"semantic_{idx}",
            "strategy": "semantic",
            "text": p
        })
    return chunks

def chunk_hierarchical(text: str):
    """Chia chunk theo cấu trúc Chương / Điều / Mục"""
    pattern = r'(Chương\s+[IVXLCDM\d]+|Điều\s+\d+|Mục\s+\d+)'
    parts = re.split(pattern, text)
    chunks = []
    idx = 1
    
    if len(parts) <= 1:
        chunks.append({
            "chunk_id": "hier_1", 
            "strategy": "hierarchical", 
            "text": text, 
            "warning": "Cảnh báo: Không phát hiện tiêu đề Chương/Điều chuẩn cấu trúc!"
        })
        return chunks
        
    for i in range(1, len(parts), 2):
        header = parts[i]
        content = parts[i+1] if i+1 < len(parts) else ""
        chunks.append({
            "chunk_id": f"hier_{idx}",
            "strategy": "hierarchical",
            "header": header,
            "text": f"{header}\n{content.strip()}"
        })
        idx += 1
    return chunks

async def main():
    if not os.path.exists(DATADEMO_DIR):
        print(f"Lỗi: Không tìm thấy thư mục {DATADEMO_DIR}")
        return

    pdf_files = [f for f in os.listdir(DATADEMO_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print("Lỗi: Chưa có file PDF mẫu nào trong thư mục datademo!")
        return

    pdf_path = os.path.join(DATADEMO_DIR, pdf_files[0])
    print(f"Đang xử lý file: {pdf_files[0]}")
    
    text, ocr_used = await extract_text_from_pdf(pdf_path)
    
    result_data = {
        "source": pdf_files[0],
        "ocr_used": ocr_used,
        "raw_text": text,
        "fixed_chunks": chunk_fixed_size(text),
        "semantic_chunks": chunk_semantic(text),
        "hierarchical_chunks": chunk_hierarchical(text)
    }
    
    output_path = os.path.join(OUTPUT_DIR, "result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n Hoàn thành! Kết quả đã lưu vào: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
    