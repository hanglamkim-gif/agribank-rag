import os
import chromadb

# Khởi tạo ChromaDB lưu cục bộ tại thư mục ./chroma_db
client = chromadb.PersistentClient(path="./chroma_db")

# Tạo hoặc gọi collection lưu trữ văn bản pháp luật / nhân sự
collection = client.get_or_create_collection(name="agribank_legal_docs")

# Dữ liệu mẫu (hoặc bạn có thể trích xuất từ các file trong thư mục ner_kb/ của bạn)
documents = [
    "Nghị định 123/2023/NĐ-CP quy định về cơ chế tài chính, quản lý thuế và các hoạt động liên quan đối với tổ chức tín dụng, ngân hàng thương mại.",
    "Thông tư 45/2023/TT-BTC hướng dẫn thực hiện chế độ kế toán, hạch toán và báo cáo tài chính áp dụng cho hệ thống các ngân hàng thương mại tại Việt Nam.",
    "Quyết định 789/2024/QĐ-NHNN ban hành quy định chi tiết về phân loại nợ, mức trích, phương pháp trích lập dự phòng rủi ro trong hoạt động tín dụng."
]
metadatas = [
    {"so_ky_hieu": "123/2023/NĐ-CP"}, 
    {"so_ky_hieu": "45/2023/TT-BTC"}, 
    {"so_ky_hieu": "789/2024/QĐ-NHNN"}
]
ids = ["doc_1", "doc_2", "doc_3"]

# Thêm dữ liệu vào ChromaDB collection
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("Khởi tạo và nạp dữ liệu thành công vào thư mục ./chroma_db!")