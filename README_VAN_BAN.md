# 📜 Ứng dụng Quản lý Danh mục Văn bản Nội bộ (Agribank KTNB)

Ứng dụng web phục vụ nhóm **Kiểm toán Nội bộ (KTNB) Agribank** quản lý danh mục văn bản, quyết định, quy trình nội bộ. Đây là bước đệm chuẩn bị dữ liệu cho hệ thống tra cứu văn bản tự động bằng RAG (Retrieval-Augmented Generation).

---

## ✨ Tính năng chính

1. **Danh sách văn bản**: 
   - Quản lý đầy đủ 4 trường thông tin: **Số hiệu** (`so_hieu`), **Tiêu đề** (`tieu_de`), **Ngày ban hành** (`ngay_ban_hanh`), **Trạng thái hiệu lực** (`con_hieu_luc`).
   - Đếm nhanh thống kê Dashboard (Tổng số văn bản, Còn hiệu lực, Hết hiệu lực).
2. **Tìm kiếm thời gian thực (Real-time Search)**:
   - Tìm theo **Số hiệu** (ví dụ: `15/2026/QĐ-KTNB`, `QĐ-88/2024/NHNO`) hoặc **Tiêu đề văn bản**.
3. **Bộ lọc trạng thái (Status Filter)**:
   - Lọc danh sách theo trạng thái: **Tất cả**, **Còn hiệu lực**, **Hết hiệu lực**.
4. **Quản lý CRUD linh hoạt**:
   - **Thêm mới**: Nút "+ Thêm Văn Bản Mới" mở Modal form có kiểm tra dữ liệu đầu vào.
   - **Chỉnh sửa**: Cập nhật thông tin nhanh qua Modal.
   - **Đổi hiệu lực 1-Click**: Nút chuyển đổi trạng thái hiệu lực tức thì.
   - **Xóa văn bản**: Xóa kèm hộp thoại xác nhận an toàn.
5. **Thiết kế chuẩn Agribank UI/UX**:
   - Màu sắc nhận diện ngân hàng: Đỏ đô (`#800000`) & Vàng kim (`#DAA520`).
   - Giao diện tiếng Việt chuẩn, hỗ trợ hiển thị đẹp trên cả Máy tính & Di động.

---

## 🚀 Hướng dẫn khởi chạy ứng dụng

### Cách 1: Khởi chạy với Backend Python (Khuyên dùng)

Ứng dụng tích hợp sẵn Backend REST API viết bằng thư viện chuẩn của Python (`http.server`), **không cần cài thêm bất kỳ thư viện ngoài nào (Flask, Django...)**.

1. Mở cửa sổ Terminal (PowerShell / Command Prompt) tại thư mục dự án `c:\agribank-rag`.
2. Chạy lệnh sau:
   ```bash
   python app.py
   ```
3. Mở trình duyệt web bất kỳ (Chrome, Edge, Firefox...) và truy cập địa chỉ:
   ```
   http://localhost:8000
   ```

---

### Cách 2: Mở trực tiếp file HTML (Chế độ Standalone)

Nếu không muốn chạy lệnh Python, ứng dụng hỗ trợ cơ chế tự động fallback chạy trực tiếp dữ liệu trong bộ nhớ trình duyệt JavaScript:

1. Mở thư mục `c:\agribank-rag`.
2. Click đúp vào file `index.html` để mở trực tiếp trên trình duyệt.

---

## 📁 Cấu trúc thư mục dự án

```
c:/agribank-rag/
├── app.py                  # Backend HTTP REST API bằng Python (In-Memory Store)
├── index.html              # Frontend Web App giao diện Agribank KTNB
├── van_ban_final.json      # File dữ liệu văn bản ban đầu
├── SPEC_van_ban.md         # Mô tả yêu cầu kỹ thuật chi tiết
└── README_VAN_BAN.md       # Tài liệu hướng dẫn sử dụng ứng dụng
```

---

## 💡 Cấu trúc mẫu dữ liệu Văn bản

```json
{
  "id": 1,
  "so_hieu": "15/2026/QĐ-KTNB",
  "tieu_de": "Quy định kiểm toán nội bộ về an toàn thông tin và bảo mật dữ liệu khách hàng",
  "ngay_ban_hanh": "2026-03-15",
  "con_hieu_luc": true
}
```
