import json
import os
from typing import List
from pydantic import ValidationError
from .validator import DocumentMetadata, DocumentChunk

class DocumentLoader:
    """
    Trình nạp tài liệu cho quy trình RAG. Hỗ trợ đọc các file (JSON, TXT, MD) 
    và xác thực/làm sạch nội dung bằng Pydantic Validator.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_json(self, file_path: str) -> List[DocumentChunk]:
        """
        Nạp tài liệu từ file JSON có cấu trúc.
        Mỗi đối tượng trong file JSON cần có: so_hieu, tieu_de, ngay_ban_hanh, con_hieu_luc, content
        """
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"[CẢNH BÁO] File {file_path} không chứa danh sách (list) các đối tượng JSON hợp lệ.")
                return chunks

            for item in data:
                try:
                    # 1. Trích xuất và validate Metadata
                    metadata = DocumentMetadata(
                        so_hieu=item.get("so_hieu", ""),
                        tieu_de=item.get("tieu_de", ""),
                        ngay_ban_hanh=item.get("ngay_ban_hanh"),
                        con_hieu_luc=item.get("con_hieu_luc", True),
                        nguon=file_path
                    )
                    
                    # 2. Xử lý phần content
                    content = item.get("content", "")
                    
                    # Giả sử ở bước Loader, ta tạo một chunk cơ bản toàn bộ nội dung.
                    # Bước cắt (chunking) thực tế sẽ thực hiện trên content này sau.
                    chunk = DocumentChunk(content=content, metadata=metadata)
                    chunks.append(chunk)

                except ValidationError as e:
                    print(f"[LỖI XÁC THỰC] Bỏ qua văn bản '{item.get('so_hieu', 'N/A')}' do lỗi: {e}")
                except Exception as e:
                    print(f"[LỖI KHÔNG XÁC ĐỊNH] Lỗi khi xử lý item: {e}")
                    
        except FileNotFoundError:
            print(f"[LỖI] Không tìm thấy file: {file_path}")
        except json.JSONDecodeError:
            print(f"[LỖI] File JSON không hợp lệ: {file_path}")
        
        return chunks

    def load_all(self) -> List[DocumentChunk]:
        """
        Quét qua toàn bộ thư mục data_dir và nạp tất cả các file JSON (sau này mở rộng thêm txt/md)
        """
        all_chunks = []
        if not os.path.exists(self.data_dir):
            print(f"[LỖI] Thư mục '{self.data_dir}' không tồn tại.")
            return all_chunks

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Hiện tại chỉ hỗ trợ JSON làm mẫu nạp
                if file.endswith(".json"):
                    print(f"Đang nạp file: {file_path}...")
                    chunks = self.load_json(file_path)
                    all_chunks.extend(chunks)
                else:
                    print(f"Bỏ qua định dạng file chưa được hỗ trợ: {file}")

        print(f"Đã nạp thành công tổng cộng {len(all_chunks)} đoạn văn bản hợp lệ.")
        return all_chunks
