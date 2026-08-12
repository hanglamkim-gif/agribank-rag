# SPEC Ứng dụng Quản lý Danh mục Văn bản Nội bộ (Agribank KTNB)

## 1. Mục tiêu
Xây dựng ứng dụng quản lý danh mục văn bản nội bộ dành riêng cho nhóm Kiểm toán Nội bộ (KTNB) Agribank. Đây là sản phẩm bước đệm cho hệ thống RAG tra cứu tài liệu thông minh trong tương lai.

## 2. Chức năng bắt buộc
- **Danh sách văn bản**: Hiển thị danh sách đầy đủ thông tin văn bản.
- **Thêm văn bản mới**: Nhập thông tin số hiệu, tiêu đề, ngày ban hành và trạng thái hiệu lực.
- **Sửa thông tin văn bản**: Cho phép chỉnh sửa bất kỳ thông tin nào của văn bản đã chọn.
- **Xóa văn bản**: Xóa văn bản khỏi danh mục với xác nhận từ người dùng.
- **Tìm kiếm**: Tìm kiếm thông minh theo số hiệu hoặc tiêu đề (không phân biệt hoa thường).
- **Lọc theo hiệu lực**: Lọc danh sách theo trạng thái (Tất cả / Còn hiệu lực / Hết hiệu lực).
- **Thống kê nhanh**: Đếm tổng số văn bản, số văn bản còn hiệu lực và hết hiệu lực.

## 3. Cấu trúc dữ liệu (Model)
Mỗi văn bản gồm 4 trường thông tin chính:
- `so_hieu`: Chuỗi ký tự (Ví dụ: `15/2026/QĐ-KTNB`, `QĐ-88/2024/NHNO`)
- `tieu_de`: Chuỗi ký tự (Ví dụ: `Quy định về an toàn thông tin và bảo mật dữ liệu KTNB`)
- `ngay_ban_hanh`: Chuỗi ký tự ngày theo định dạng YYYY-MM-DD hoặc DD/MM/YYYY (Ví dụ: `2026-03-15`)
- `con_hieu_luc`: Kiểu Boolean (`true` = Còn hiệu lực, `false` = Hết hiệu lực)

## 4. Ràng buộc & Tiêu chuẩn
- Giao diện tiếng Việt chuẩn, giữ đúng dấu.
- Dữ liệu lưu tạm trong bộ nhớ (In-memory storage).
- Mã nguồn gọn gàng, dễ đọc, có chú thích chi tiết.
- Màu sắc chủ đạo: Đỏ đô Agribank (`#8B0000`), Vàng kim (`#DAA520`).
