from sentence_transformers import SentenceTransformer

# 1. Khởi tạo danh sách các phân đoạn giả lập (chunks)
chunks = [
    {"id": "chunk_0", "type": "Chương", "content": "Chương 1: Quy định chung", "parent": "Document"},
    {"id": "chunk_1", "type": "Mục", "content": "Mục 1: Phạm vi điều chỉnh", "parent": "Chương 1: Quy định chung"},
    {"id": "chunk_2", "type": "Điều", "content": "Điều 1: Phạm vi", "parent": "Mục 1: Phạm vi điều chỉnh"},
    {"id": "chunk_3", "type": "Đoạn văn", "content": "Luật này quy định về các hoạt động ngân hàng và tổ chức tín dụng.", "parent": "Điều 1: Phạm vi"}
]

print("1. Đang tải mô hình AI tiếng Việt (thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5)...")
# Tải mô hình ngôn ngữ tiếng Việt theo yêu cầu bài thực hành
model = SentenceTransformer('thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5')

print("\n2. Đang tạo vector nhúng (embedding)...")
for chunk in chunks:
    # Chuyển đổi văn bản thành vector số học
    vector = model.encode(chunk['content'])
    chunk['embedding'] = vector.tolist() 
    print(f" - Đã nhúng: '{chunk['content']}' -> Kích thước: {len(chunk['embedding'])} chiều")

print("\nHoàn thành Bước 2! Đã sẵn sàng nạp dữ liệu vào đồ thị.")
