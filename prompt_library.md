# Prompt Template Library - KTNB Agribank

## Mẫu 1 — Sinh ứng dụng CRUD đơn giản
[ROLE] Bạn là full-stack developer.
[CONTEXT] Đây là dự án agribank-rag của một nhóm KTNB. Sản phẩm đầu tiên là ứng dụng quản lý công việc nội bộ nhóm.
[GOAL] Tạo web app quản lý công việc: danh sách task, thêm/sửa/xóa, đánh dấu hoàn thành, lọc theo trạng thái (tất cả / đang làm / xong).
[CONSTRAINTS] Giao diện tiếng Việt, lưu dữ liệu tạm trong bộ nhớ, code gọn, dễ đọc, có chú thích. Mỗi công việc gồm: ten, nguoi_phu_trach, trang_thai.
[OUTPUT] Ứng dụng chạy được + hướng dẫn chạy.

---

## Mẫu 2 — Self-review edge case
[ROLE] Bạn là QA Automation và Senior Developer.
[CONTEXT] Ứng dụng quản lý công việc vừa sinh ra cần được rà soát chất lượng code.
[GOAL] Tự rà soát lại code vừa sinh: kiểm tra các trường hợp biên (danh sách rỗng, sửa/xóa công việc không tồn tại, nhập tên công việc trống) và sửa nếu phát hiện lỗi.
[CONSTRAINTS] Giữ nguyên cấu trúc giao diện cũ, chỉ chỉnh sửa logic xử lý lỗi.
[OUTPUT] Báo cáo giải thích ngắn gọn lỗi nào đã sửa + mã nguồn đã tối ưu.
