# SPEC Ứng dụng quản lý nhân sự (agribank-nhansu)

## Mục tiêu
Web app quản lý nhân sự nội bộ cho một nhóm KTNB nhỏ.

## Chức năng bắt buộc
- Thêm nhân sự mới (Mã NV, Họ và tên, Chức vụ, Phòng ban, Trạng thái)
- Sửa thông tin nhân sự
- Xóa nhân sự
- Đổi trạng thái làm việc (Đang làm việc / Đã nghỉ)
- Tìm kiếm nhân sự theo Mã NV hoặc Tên
- Lọc danh sách theo trạng thái: Tất cả / Đang làm việc / Đã nghỉ

## Dữ liệu
- Lưu tạm trong bộ nhớ (chưa cần database ở buổi này)
- Mỗi nhân sự gồm: ma_nv, ho_ten, chuc_vu, phong_ban, trang_thai

## Ràng buộc
- Giao diện tiếng Việt
- Code gọn, dễ đọc, có chú thích
- Không hardcode dữ liệu nhạy cảm
[ROLE] Bạn là một full-stack developer chuyên nghiệp.
[CONTEXT] Đây là dự án agribank-rag của một nhóm KTNB Agribank. Sản phẩm là ứng dụng quản lý nhân sự nội bộ nhóm.
[GOAL] Tạo web app quản lý nhân sự (HTML/CSS/JS): danh sách nhân sự, thêm/sửa/xóa nhân sự, đánh dấu trạng thái làm việc, lọc và tìm kiếm.
[CONSTRAINTS] Giao diện tiếng Việt, lưu dữ liệu tạm trong bộ nhớ, code gọn, dễ đọc, có chú thích chi tiết.
[OUTPUT] Toàn bộ mã nguồn ứng dụng chạy được ngay + hướng dẫn chi tiết cách chạy ứng dụng.
