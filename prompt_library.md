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
---

## Mẫu 3 — Sinh code ứng dụng từ SPEC
[ROLE] Bạn là full-stack developer.
[CONTEXT] Dự án agribank-rag của nhóm KTNB. Sản phẩm là ứng dụng quản lý danh mục văn bản nội bộ.
[GOAL] Tạo web app quản lý văn bản với đầy đủ tính năng: danh sách, thêm/sửa/xóa, tìm kiếm, lọc trạng thái.
[CONSTRAINTS] Giao diện tiếng Việt, giữ đúng dấu, lưu dữ liệu tạm trong bộ nhớ, code gọn.
[OUTPUT] Mã nguồn hoàn chỉnh chạy được ngay + hướng dẫn chạy.

---

## Mẫu 4 — Debug lỗi logic/runtime
[ROLE] Bạn là chuyên gia Python/JavaScript Debugger.
[CONTEXT] Ứng dụng quản lý văn bản đang gặp lỗi khi thực hiện thao tác tìm kiếm từ khóa tiếng Việt.
[GOAL] Tìm ra nguyên nhân gây lỗi và cung cấp đoạn mã đã sửa.
[CONSTRAINTS] Giữ nguyên cấu trúc hàm hiện tại, giải thích rõ lý do bị lỗi.
[OUTPUT] Giải thích nguyên nhân lỗi + đoạn code đã khắc phục.

---

## Mẫu 5 — Refactor sửa lỗi chức năng cụ thể
[ROLE] Bạn là Senior Developer.
[CONTEXT] Chức năng lọc theo trạng thái hiệu lực hiển thị không chính xác.
[GOAL] Sửa lại logic lọc theo đúng trạng thái "Còn hiệu lực" và "Hết hiệu lực".
[CONSTRAINTS] Chỉ thay đổi code liên quan đến hàm filter, không làm ảnh hưởng đến các chức năng khác.
[OUTPUT] Đoạn code hàm lọc đã tinh chỉnh.

---

## Mẫu 6 — Security Review & Khắc phục bảo mật
[ROLE] Bạn là chuyên gia Cyber Security và Code Auditor.
[CONTEXT] Mã nguồn ứng dụng web cần được rà soát an toàn thông tin trước khi đưa vào vận hành nội bộ.
[GOAL] Phát hiện và loại bỏ các chuỗi Hardcode secret/API key, thêm validate dữ liệu đầu vào.
[CONSTRAINTS] Không làm ảnh hưởng đến luồng hoạt động chính của người dùng.
[OUTPUT] Mã nguồn đã được tăng cường bảo mật.

---

## Mẫu 7 — Sinh tài liệu kỹ thuật (README)
[ROLE] Bạn là Technical Writer.
[CONTEXT] Dự án web app quản lý văn bản nội bộ cho nhóm KTNB Agribank.
[GOAL] Viết file README.md chi tiết hướng dẫn cài đặt, cấu hình và sử dụng ứng dụng.
[CONSTRAINTS] Trình bày rõ ràng bằng tiếng Việt, sử dụng định dạng Markdown chuẩn.
[OUTPUT] Nội dung file README.md hoàn chỉnh.

---

## Mẫu 8 — Đối chiếu văn bản nội bộ
[ROLE] Bạn là Chuyên viên Kiểm toán nội bộ.
[CONTEXT] Cần đối chiếu sự thay đổi giữa hai phiên bản quy định/văn bản nội bộ.
[GOAL] So sánh hai đoạn văn bản mẫu và liệt kê danh sách các điểm khác biệt chính.
[CONSTRAINTS] Trình bày dưới dạng bảng so sánh đối chiếu, nhấn mạnh các thay đổi về quyền hạn và quy trình.
[OUTPUT] Bảng tổng hợp đối chiếu điểm khác biệt.

---

## Mẫu 9 — Sinh Checklist kiểm toán ứng dụng
[ROLE] Bạn là Trưởng nhóm Kiểm toán CNTT (IT Auditor).
[CONTEXT] Chuẩn bị kiểm thử đánh giá chất lượng cho một ứng dụng web nội bộ mới phát triển.
[GOAL] Sinh bảng Checklist kiểm thử toàn diện các chức năng và khía cạnh an toàn thông tin.
[CONSTRAINTS] Bao gồm các cột: STT, Hạng mục kiểm tra, Cách thực hiện, Trạng thái (Đạt/Không đạt).
[OUTPUT] Bảng Checklist kiểm toán ở định dạng Markdown.

---

## Mẫu 10 — Self-review edge case dữ liệu
[ROLE] Bạn là Senior QA Engineer.
[CONTEXT] Rà soát các trường hợp biên của ứng dụng quản lý văn bản.
[GOAL] Bổ sung logic xử lý các trường hợp: danh sách rỗng, nhập chuỗi quá dài, ngày tháng sai định dạng.
[CONSTRAINTS] Đảm bảo ứng dụng hiển thị thông báo lỗi thân thiện thay vì bị crash.
[OUTPUT] Mã nguồn đã được bổ sung hàm kiểm tra edge case.

