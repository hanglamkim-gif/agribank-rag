# SPEC Ứng dụng quản lý công việc (agribank-todo)

## Mục tiêu
Web app quản lý công việc nội bộ cho một nhóm KTNB nhỏ.

## Chức năng bắt buộc
- Thêm công việc mới (tên, người phụ trách)
- Sửa thông tin công việc
- Xóa công việc
- Đánh dấu hoàn thành / chưa hoàn thành
- Lọc danh sách theo trạng thái: tất cả / đang làm / xong

## Dữ liệu
- Lưu tạm trong bộ nhớ (chưa cần database ở buổi này)
- Mỗi công việc gồm: ten, nguoi_phu_trach, trang_thai

## Ràng buộc
- Giao diện tiếng Việt
- Code gọn, dễ đọc, có chú thích
- Không hardcode dữ liệu nhạy cảm
[ROLE] Bạn là một full-stack developer chuyên nghiệp.
[CONTEXT] Đây là dự án agribank-rag của một nhóm KTNB Agribank. Sản phẩm đầu tiên là ứng dụng quản lý công việc nội bộ nhóm.
[GOAL] Tạo web app quản lý công việc (HTML/CSS/JS hoặc Python Streamlit/Flask đơn giản): danh sách task, thêm/sửa/xóa công việc, đánh dấu hoàn thành, lọc theo trạng thái (tất cả / đang làm / xong).
[CONSTRAINTS] Giao diện tiếng Việt, lưu dữ liệu tạm trong bộ nhớ (hoặc LocalStorage), code gọn, dễ đọc, có chú thích chi tiết. Mỗi công việc gồm các trường: ten, nguoi_phu_trach, trang_thai.
[OUTPUT] Toàn bộ mã nguồn ứng dụng chạy được ngay + hướng dẫn chi tiết cách chạy ứng dụng.
