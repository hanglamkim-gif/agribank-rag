import os
from typing import List
from dotenv import load_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.config import Settings
from .validator import DocumentChunk

# Đảm bảo nạp các biến môi trường
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

# Thư mục lưu trữ ChromaDB
STORAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "chroma")

from chromadb import Documents, EmbeddingFunction, Embeddings

class MockEmbeddingFunction(EmbeddingFunction):
    """
    Mock Embedding Function dùng cho mục đích kiểm thử cục bộ.
    Trả về vector giả lập (384 chiều) dựa trên độ dài của văn bản để tránh lỗi cài đặt ONNX/PyTorch.
    """
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            # Tạo vector giả
            val = float(len(text) % 100) / 100.0
            embeddings.append([val] * 384)
        return embeddings

class RAGVectorStore:
    def __init__(self, collection_name: str = "agribank_vanban"):
        os.makedirs(STORAGE_PATH, exist_ok=True)
        
        # Khởi tạo ChromaDB client chế độ Persistent
        self.client = chromadb.PersistentClient(path=STORAGE_PATH)
        
        # Bỏ qua Gemini Embedding do API Key không hợp lệ, và bỏ qua Default ONNX do thiếu DLL trên Windows.
        # Sử dụng Mock Embedding để kiểm thử việc lưu trữ/truy vấn.
        self.embedding_fn = MockEmbeddingFunction()
        
        # Tạo hoặc lấy collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_documents(self, chunks: List[DocumentChunk]):
        """Nạp danh sách các Chunk vào cơ sở dữ liệu Vector"""
        if not chunks:
            print("[CẢNH BÁO] Danh sách nạp rỗng.")
            return

        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            # Lưu lại ID giả lập (thực tế có thể dùng hash hoặc id thực tế)
            doc_id = f"{chunk.metadata.so_hieu}_chunk_{i}"
            
            # ChromaDB không nhận boolean/None trực tiếp trong metadata, phải ép về dạng chuỗi hoặc số
            meta = {
                "so_hieu": chunk.metadata.so_hieu,
                "tieu_de": chunk.metadata.tieu_de,
                "ngay_ban_hanh": chunk.metadata.ngay_ban_hanh or "Unknown",
                "con_hieu_luc": "True" if chunk.metadata.con_hieu_luc else "False",
                "nguon": chunk.metadata.nguon
            }
            
            documents.append(chunk.content)
            metadatas.append(meta)
            ids.append(doc_id)
            
        # Nạp (Upsert) vào ChromaDB
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[THÀNH CÔNG] Đã nạp/cập nhật {len(documents)} vectors vào ChromaDB.")

    def search(self, query: str, n_results: int = 2):
        """Tìm kiếm ngữ nghĩa"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
