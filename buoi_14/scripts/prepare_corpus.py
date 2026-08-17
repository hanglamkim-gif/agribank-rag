import os
import re
import html
from pathlib import Path
import pandas as pd

# Đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR.parent / "ner_kb"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "chunks_normalized.csv"

def clean_html_text(raw_html: str) -> str:
    """Loại bỏ thẻ HTML và chuẩn hóa khoảng trắng văn bản."""
    if not isinstance(raw_html, str) or not raw_html.strip():
        return ""
    # Giải mã ký tự HTML entities (như  , &)
    text = html.unescape(raw_html)
    # Loại bỏ toàn bộ thẻ HTML (, , ,...)
    text = re.sub(r"<[^>]+>", " ", text)
    # Chuẩn hóa khoảng trắng và dòng trống
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_corpus():
    content_path = SOURCE_DIR / "content.csv"
    meta_path = SOURCE_DIR / "metadata.csv"

    if not content_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn tại: {SOURCE_DIR}")

    df_content = pd.read_csv(content_path)
    df_meta = pd.read_csv(meta_path)

    print(f"-> Đọc {len(df_content)} dòng từ content.csv")
    print(f"-> Đọc {len(df_meta)} dòng từ metadata.csv")

    # Ghép nối bảng qua khóa chính 'id'
    merged = pd.merge(df_content, df_meta, on="id", how="left")

    # Chuẩn hóa schema cho pipeline retrieval
    normalized = pd.DataFrame()
    normalized["chunk_id"] = merged["id"].astype(str)
    normalized["document_id"] = merged["id"].astype(str)
    normalized["text"] = merged["content_html"].apply(clean_html_text)
    
    # Metadata phục vụ trích dẫn (Citation)
    doc_type = merged["loai_van_ban"].fillna("Quy định").astype(str)
    so_hieu = merged["so_ky_hieu"].fillna(merged["id"]).astype(str)
    co_quan = merged["co_quan_ban_hanh"].fillna("Agribank").astype(str)
    
    normalized["title"] = doc_type + " " + so_hieu
    normalized["source_file"] = "ner_kb/content.csv"
    normalized["citation"] = "[" + normalized["title"] + " | " + co_quan + " | DocID: " + normalized["document_id"] + "]"

    # Loại bỏ các đoạn văn bản rỗng và trùng lặp
    normalized = normalized[normalized["text"].str.len() > 10].drop_duplicates(subset=["chunk_id"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    print("\n=== KẾT QUẢ CHUẨN HÓA CORPUS ===")
    print(f"✅ Tổng số chunks hợp lệ: {len(normalized)}")
    print(f"✅ File đầu ra: {OUTPUT_FILE}")
    print("\n--- MẪU 2 BẢN GHI ĐẦU TIÊN ---")
    for _, row in normalized.head(2).iterrows():
        print(f"• ID: {row['chunk_id']}")
        print(f"  Citation: {row['citation']}")
        print(f"  Text preview: {row['text'][:120]}...\n")

if __name__ == "__main__":
    prepare_corpus()