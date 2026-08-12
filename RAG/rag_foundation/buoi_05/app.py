import streamlit as st
import json
import os

st.set_page_config(page_title="RAG Foundation - Buổi 5", layout="wide")
st.title("Trực quan hóa OCR & Các Chiến lược Chunking")

# Đường dẫn tới file dữ liệu đầu ra
BASE_DIR = os.path.dirname(__file__)
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "result.json")

if not os.path.exists(OUTPUT_FILE):
    st.error("Chưa tìm thấy file output! Hãy chạy `python RAG/rag_foundation/buoi_05/src/ocr_chunking.py` trước.")
else:
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Hiển thị thông tin tổng quan ở thanh bên (Sidebar)
    st.sidebar.header("Thông tin Văn bản")
    st.sidebar.write(f"**Tên file:** {data.get('source', 'N/A')}")
    st.sidebar.write(f"**Sử dụng OCR LlamaParse:** {data.get('ocr_used', False)}")

    # Chia các Tab để trực quan hóa từng chiến lược
    tab1, tab2, tab3, tab4 = st.tabs([
        "Văn bản thô (Raw Text)", 
        "Fixed-size Chunking", 
        "Semantic Chunking", 
        "Hierarchical Chunking"
    ])

    with tab1:
        st.subheader("Văn bản gốc đã chuẩn hóa Unicode (NFC)")
        st.text_area("Nội dung trích xuất:", data.get("raw_text", ""), height=400)

    with tab2:
        fixed_chunks = data.get("fixed_chunks", [])
        st.subheader(f"Fixed-size Chunking (Tổng số: {len(fixed_chunks)} chunks)")
        st.caption("Chiến lược: Cắt độ dài cố định kèm overlap")
        for item in fixed_chunks:
            with st.expander(f"Chunk ID: {item['chunk_id']}"):
                st.write(item['text'])

    with tab3:
        semantic_chunks = data.get("semantic_chunks", [])
        st.subheader(f"Semantic Chunking (Tổng số: {len(semantic_chunks)} chunks)")
        st.caption("Chiến lược: Phân đoạn dựa trên xuống dòng / ngắt đoạn")
        for item in semantic_chunks:
            with st.expander(f"Chunk ID: {item['chunk_id']}"):
                st.write(item['text'])

    with tab4:
        hier_chunks = data.get("hierarchical_chunks", [])
        st.subheader(f"Hierarchical Chunking (Tổng số: {len(hier_chunks)} chunks)")
        st.caption("Chiến lược: Cấu trúc theo Chương / Mục / Điều")
        for item in hier_chunks:
            with st.expander(f"Chunk ID: {item['chunk_id']} - {item.get('header', '')}"):
                if "warning" in item:
                    st.warning(item["warning"])
                st.write(item['text'])
                