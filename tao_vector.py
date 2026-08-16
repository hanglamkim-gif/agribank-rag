from sentence_transformers import SentenceTransformer

# 1. Khoi tao danh sach cac phan doan gia lap
chunks = [
    {"id": "chunk_0", "type": "Chương", "content": "Chương 1: Quy định chung", "parent": "Document"},
    {"id": "chunk_1", "type": "Mục", "content": "Mục 1: Phạm vi điều chỉnh", "parent": "Chương 1: Quy định chung"},
    {"id": "chunk_2", "type": "Điều", "content": "Điều 1: Phạm vi", "parent": "Mục 1: Phạm vi điều chỉnh"},
    {"id": "chunk_3", "type": "Đoạn văn", "content": "Luật này quy định về các hoạt động ngân hàng và tổ chức tín dụng.", "parent": "Điều 1: Phạm vi"}
]

print("1. Dang tai mo hinh AI tieng Viet (khi chay lan dau se mat 1-3 phut)...")
model = SentenceTransformer('thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5')

print("\n2. Dang tao vector nhung (embedding)...")
for chunk in chunks:
    vector = model.encode(chunk['content'])
    chunk['embedding'] = vector.tolist() 
    print(f" - Da nhung: '{chunk['content']}' -> Kich thuoc vector: {len(chunk['embedding'])} chieu")

print("\nHoan thanh Buoc 2! Da san sang nap du lieu vao do thi.")
