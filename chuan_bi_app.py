import json

# ==========================================
# 1. PHẦN CÔNG VIỆC (Đủ 6 công việc)
# ==========================================
danh_sach_cong_viec = [
    {"ten": "Rà soát Thông tư 41/2016", "nguoi_phu_trach": "Nguyễn Văn A", "trang_thai": "dang_lam"},
    {"ten": "Lập checklist kiểm toán CNTT", "nguoi_phu_trach": "Trần Thị B", "trang_thai": "xong"},
    {"ten": "Đối chiếu quy định tín dụng", "nguoi_phu_trach": "Lê Văn C", "trang_thai": "xong"},
    {"ten": "Kiểm tra an toàn hệ thống Core Banking", "nguoi_phu_trach": "Nguyễn Văn A", "trang_thai": "chua_lam"},
    {"ten": "Đánh giá rủi ro vận hành Q3", "nguoi_phu_trach": "Phạm Văn D", "trang_thai": "dang_lam"},
    {"ten": "Tổng hợp báo cáo kiểm toán nội bộ", "nguoi_phu_trach": "Trần Thị B", "trang_thai": "chua_lam"}
]

def thong_ke_cong_viec(danh_sach):
    """Hàm thống kê số lượng công việc theo từng trạng thái."""
    thong_ke = {
        "chua_lam": 0,
        "dang_lam": 0,
        "xong": 0
    }
    for cv in danh_sach:
        trang_thai = cv.get("trang_thai")
        if trang_thai in thong_ke:
            thong_ke[trang_thai] += 1
    return thong_ke


# ==========================================
# 2. PHẦN VĂN BẢN (Đủ 6 văn bản)
# ==========================================
danh_sach_van_ban = [
    {"so_hieu": "QD-15/2026/NHNO", "loai_van_ban": "Quy định", "con_hieu_luc": True},
    {"so_hieu": "QT-02/2025/NHNO", "loai_van_ban": "Quy trình", "con_hieu_luc": False},
    {"so_hieu": "QD-88/2024/NHNO", "loai_van_ban": "Quy định", "con_hieu_luc": True},
    {"so_hieu": "HD-05/2026/NHNO", "loai_van_ban": "Hướng dẫn", "con_hieu_luc": True},
    {"so_hieu": "QT-10/2023/NHNO", "loai_van_ban": "Quy trình", "con_hieu_luc": False},
    {"so_hieu": "QD-101/2026/NHNO", "loai_van_ban": "Quy định", "con_hieu_luc": True}
]

def tim_kiem_van_ban(danh_sach, tu_khoa):
    """Hàm tìm kiếm văn bản có số hiệu chứa từ khóa."""
    ket_qua = []
    for vb in danh_sach:
        if tu_khoa.lower() in vb["so_hieu"].lower():
            ket_qua.append(vb)
    return ket_qua


# ==========================================
# 3. IN KẾT QUẢ RA MÀN HÌNH
# ==========================================
print("=== KẾT QUẢ THỐNG KÊ CÔNG VIỆC ===")
ket_qua_thong_ke = thong_ke_cong_viec(danh_sach_cong_viec)
print(f"Chưa làm: {ket_qua_thong_ke['chua_lam']}")
print(f"Đang làm: {ket_qua_thong_ke['dang_lam']}")
print(f"Đã xong:  {ket_qua_thong_ke['xong']}")

print("\n=== KẾT QUẢ TÌM KIẾM VĂN BẢN (Từ khóa: '2026') ===")
ket_qua_tim_kiem = tim_kiem_van_ban(danh_sach_van_ban, "2026")
for vb in ket_qua_tim_kiem:
    print(f"- Số hiệu: {vb['so_hieu']} | Loại: {vb['loai_van_ban']} | Còn hiệu lực: {vb['con_hieu_luc']}")


# ==========================================
# 4. LƯU RA 2 FILE JSON
# ==========================================
with open("cong_viec_final.json", "w", encoding="utf-8") as f:
    json.dump(danh_sach_cong_viec, f, ensure_ascii=False, indent=2)

with open("van_ban_final.json", "w", encoding="utf-8") as f:
    json.dump(danh_sach_van_ban, f, ensure_ascii=False, indent=2)

print("\n-> Đã tạo thành công cong_viec_final.json và van_ban_final.json!")
