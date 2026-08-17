# BÁO CÁO KIỂM THỬ BẢO MẬT (SECURITY AUDIT) — BUỔI 15

| Test ID | Tên bài test | Vai trò kiểm thử | Kết quả | Ghi chú |
|---|---|---|---|---|
| SEC-01 | Guest tra cứu Nghị định 123 (Quyền HR) | ['Guest'] | ✅ PASS | Chặn truy cập trái phép thành công. |
| SEC-02 | Staff tra cứu Nghị định 123 (Quyền HR) | ['Staff'] | ✅ PASS | Chặn truy cập trái phép thành công. |
| SEC-03 | HR Manager tra cứu Nghị định 123 | ['HR_Manager'] | ✅ PASS | Truy cập tài liệu hợp lệ thành công. |
| SEC-04 | Staff tra cứu Thông tư 45 (Quyền Staff) | ['Staff'] | ✅ PASS | Truy cập tài liệu hợp lệ thành công. |
| SEC-05 | Admin tra cứu toàn quyền | ['Admin'] | ✅ PASS | Truy cập tài liệu hợp lệ thành công. |

## KẾT LUẬN

✅ **HỆ THỐNG ĐẠT CHỨNG NHẬN AN TOÀN RBAC MỨC CƠ BẢN**.