import os
import sys
import json

# Thêm đường dẫn để có thể import từ src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader import DocumentLoader

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
TEST_FILE = os.path.join(TEST_DATA_DIR, "sample_docs.json")

def setup_module():
    """Khởi tạo thư mục và file dữ liệu mẫu dùng cho kiểm thử"""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Dữ liệu chứa 1 hợp lệ, 2 lỗi ngày tháng, 3 thiếu tiêu đề, 4 thiếu content
    sample_data = [
        {
            "so_hieu": "QD-01/2026/NHNO",
            "tieu_de": "Tài liệu hợp lệ",
            "ngay_ban_hanh": "2026-05-12",
            "con_hieu_luc": True,
            "content": "Đây là nội dung tài liệu hợp lệ, không có khoảng   trắng thừa."
        },
        {
            "so_hieu": "QD-02/2026/NHNO",
            "tieu_de": "Lỗi ngày tháng sai định dạng",
            "ngay_ban_hanh": "12/05/2026", # Validator sẽ chuyển đổi được định dạng DD/MM/YYYY
            "con_hieu_luc": False,
            "content": "Nội dung tài liệu thứ hai hợp lệ."
        },
        {
            "so_hieu": "QD-03/2026/NHNO",
            "tieu_de": "Lỗi ngày tháng hoàn toàn sai",
            "ngay_ban_hanh": "2026/12/31", # Không hỗ trợ định dạng này
            "con_hieu_luc": True,
            "content": "Nội dung tài liệu thứ ba."
        },
        {
            "so_hieu": "QD-04/2026/NHNO",
            "tieu_de": "", # Bị trống, vi phạm Validator
            "ngay_ban_hanh": "2026-01-01",
            "con_hieu_luc": True,
            "content": "Nội dung này có tiêu đề rỗng."
        },
        {
            "so_hieu": "QD-05/2026/NHNO",
            "tieu_de": "Lỗi thiếu nội dung content",
            "ngay_ban_hanh": "2026-01-01",
            "con_hieu_luc": True,
            "content": "    " # Chỉ có khoảng trắng, vi phạm validator của DocumentChunk
        }
    ]
    
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

def teardown_module():
    """Xóa các file mẫu sau khi test xong"""
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)
    if os.path.exists(TEST_DATA_DIR):
        os.rmdir(TEST_DATA_DIR)

def test_document_loader_validation():
    try:
        setup_module()
        loader = DocumentLoader(TEST_DATA_DIR)
        chunks = loader.load_all()
        
        assert len(chunks) == 2, f"Kỳ vọng load được 2 item hợp lệ, nhưng được {len(chunks)}."
        
        # Kiểm tra việc convert định dạng ngày tháng của item 1
        assert chunks[0].metadata.ngay_ban_hanh == "2026-05-12"
        assert chunks[1].metadata.ngay_ban_hanh == "2026-05-12"
        
        # Kiểm tra xử lý chuỗi content thừa dấu cách ở item 0
        assert "khoảng trắng thừa" in chunks[0].content
        
        print("Tất cả bài kiểm tra đã VƯỢT QUA (PASSED)!")
    finally:
        teardown_module()

if __name__ == "__main__":
    test_document_loader_validation()
