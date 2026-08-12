# ==========================================
# BƯỚC 2.3: VÒNG LẶP & ĐIỀU KIỆN
# ==========================================

# Mẫu: Duyệt qua danh sách công việc và in trạng thái
print("\n--- BƯỚC 2.3: TRẠNG THÁI CÔNG VIỆC ---")
for cv in danh_sach_cong_viec:
    if cv["trang_thai"] == "xong":
        print(cv["ten"], "- ĐÃ XONG")
    elif cv["trang_thai"] == "dang_lam":
        print(cv["ten"], "- ĐANG LÀM")
    else:
        print(cv["ten"], "- CHƯA LÀM")

# TỰ LÀM: Duyệt qua danh_sach_van_ban, chỉ in các văn bản con_hieu_luc == True
print("\n--- VĂN BẢN CÒN HIỆU LỰC ---")
for vb in danh_sach_van_ban:
    if vb["con_hieu_luc"] == True:
        print(f"Số hiệu: {vb['so_hieu']} | Loại: {vb['loai_van_ban']}")
        # ==========================================
# BƯỚC 2.4: HÀM (FUNCTION)
# ==========================================

# Mẫu: Hàm đếm số công việc theo trạng thái
def dem_theo_trang_thai(danh_sach, trang_thai):
    """Đếm số công việc có trạng thái chỉ định."""
    dem = 0
    for cv in danh_sach:
        if cv["trang_thai"] == trang_thai:
            dem += 1
    return dem

so_xong = dem_theo_trang_thai(danh_sach_cong_viec, "xong")
print("\n--- BƯỚC 2.4 ---")
print("Số công việc đã xong:", so_xong)


# TỰ LÀM 1: Hàm đánh dấu hoàn thành công việc theo tên
def danh_dau_hoan_thanh(danh_sach, ten_cong_viec):
    for cv in danh_sach:
        if cv["ten"] == ten_cong_viec:
            cv["trang_thai"] = "xong"
    return danh_sach


# TỰ LÀM 2: Hàm lọc văn bản hết hiệu lực (con_hieu_luc == False)
def loc_van_ban_het_hieu_luc(danh_sach):
    ket_qua = []
    for vb in danh_sach:
        if vb["con_hieu_luc"] == False:
            ket_qua.append(vb)
    return ket_qua


# Gọi thử nghiệm các hàm vừa viết
danh_sach_cong_viec = danh_dau_hoan_thanh(danh_sach_cong_viec, "Lập checklist kiểm toán CNTT")
van_ban_het_hieu_luc = loc_van_ban_het_hieu_luc(danh_sach_van_ban)
print("Danh sách văn bản hết hiệu lực:", van_ban_het_hieu_luc)
# ==========================================
# BƯỚC 2.5: THƯ VIỆN JSON
# ==========================================
import json

# Mẫu: Lưu danh sách công việc ra file JSON
with open("cong_viec.json", "w", encoding="utf-8") as f:
    json.dump(danh_sach_cong_viec, f, ensure_ascii=False, indent=2)

# TỰ LÀM: Lưu danh_sach_van_ban ra file van_ban.json và đọc lại vào biến mới
with open("van_ban.json", "w", encoding="utf-8") as f:
    json.dump(danh_sach_van_ban, f, ensure_ascii=False, indent=2)

# Đọc lại từ file van_ban.json
with open("van_ban.json", "r", encoding="utf-8") as f:
    danh_sach_van_ban_doc_lai = json.load(f)

print("\n--- BƯỚC 2.5 ---")
print("Đọc thành công từ van_ban.json:", danh_sach_van_ban_doc_lai)
# Khai báo danh sách công việc trước
danh_sach_cong_viec = [
    {"ten": "Rà soát Thông tư 41/2016", "trang_thai": "dang_lam"},
    {"ten": "Lập checklist kiểm toán CNTT", "trang_thai": "chua_lam"},
    {"ten": "Đối chiếu quy định tín dụng", "trang_thai": "xong"}
]

# Sau đó mới đến vòng lặp duyệt danh sách
print("\n--- BƯỚC 2.3: TRẠNG THÁI CÔNG VIỆC ---")
for cv in danh_sach_cong_viec:
    if cv["trang_thai"] == "xong":
        print(cv["ten"], "- ĐÃ XONG")
    elif cv["trang_thai"] == "dang_lam":
        print(cv["ten"], "- ĐANG LÀM")
    else:
        print(cv["ten"], "- CHƯA LÀM")
        