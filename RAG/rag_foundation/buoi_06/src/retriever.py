from typing import List, Dict, Any
from .embedding import RAGVectorStore

class DocumentRetriever:
    def __init__(self, vector_store: RAGVectorStore):
        """Khởi tạo Retriever với VectorStore đã được cấu hình"""
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Thực hiện tìm kiếm Semantic Search và trả về danh sách các document thô"""
        results = self.vector_store.search(query, n_results=top_k)
        
        extracted_docs = []
        if not results or not results['documents'] or len(results['documents']) == 0:
            return extracted_docs
            
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for doc, meta, dist in zip(docs, metadatas, distances):
            extracted_docs.append({
                "content": doc,
                "metadata": meta,
                "distance": dist
            })
            
        return extracted_docs

    def format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Định dạng các document thành một chuỗi văn bản hoàn chỉnh để chèn vào Prompt (Grounding & Citation)"""
        if not docs:
            return "Không tìm thấy tài liệu nào liên quan."
            
        context_parts = []
        for i, doc in enumerate(docs):
            meta = doc["metadata"]
            so_hieu = meta.get("so_hieu", "Không rõ")
            tieu_de = meta.get("tieu_de", "Không có tiêu đề")
            content = doc["content"]
            
            # Định dạng Citation: Nguồn [Số hiệu]: [Tiêu đề]
            citation_header = f"--- Nguồn [{so_hieu}]: {tieu_de} ---"
            
            # Ghép lại thành một đoạn văn bản
            chunk_text = f"{citation_header}\n{content}\n"
            context_parts.append(chunk_text)
            
        return "\n".join(context_parts)

    def retrieve_and_format(self, query: str, top_k: int = 3) -> str:
        """Hàm tiện ích gom chung truy xuất và định dạng"""
        docs = self.retrieve(query, top_k)
        return self.format_context(docs)
