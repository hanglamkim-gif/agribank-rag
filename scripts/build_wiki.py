import os
import csv
from collections import defaultdict

# Đảm bảo cấu trúc thư mục wiki tồn tại
os.makedirs("wiki/risks", exist_ok=True)
os.makedirs("wiki/controls", exist_ok=True)
os.makedirs("wiki/events", exist_ok=True)

# 1. Đọc entities.csv
entities = {}
try:
    with open("outputs/entities.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entities[row["id"]] = row
except Exception as e:
    print(f"Lỗi đọc outputs/entities.csv: {e}")

# 2. Đọc relations.csv và nhóm theo entity ID
relations_by_source = defaultdict(list)
relations_by_target = defaultdict(list)

try:
    with open("outputs/relations.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            relations_by_source[row["source_id"]].append(row)
            relations_by_target[row["target_id"]].append(row)
except Exception as e:
    print(f"Lỗi đọc outputs/relations.csv: {e}")

# Hàm helper để làm sạch tên file an toàn
def safe_filename(name):
    return "".join([c if c.isalnum() else "_" for c in name])

page_count = 0

# 3. Tạo trang Markdown cho từng Entity
for eid, ent in entities.items():
    etype = ent["type"]
    name = ent["name"]
    
    # Phân loại thư mục lưu trữ
    if etype == "RuiRo":
        folder = "wiki/risks"
    elif etype == "KiemSoat":
        folder = "wiki/controls"
    elif etype == "SuKienRuiRo":
        folder = "wiki/events"
    else:
        folder = "wiki"
        
    filename = f"{folder}/{eid}.md"
    
    # Xây dựng nội dung Markdown với YAML frontmatter
    content = f"""---
id: {eid}
type: {etype}
verification_status: {ent.get('verification_status', 'PROPOSED')}
data_origin: {ent.get('data_origin', 'Unknown')}
---

# {name}

- **Mã định danh:** {eid}
- **Loại:** {etype}
- **Mô tả:** {ent.get('description', 'Không có mô tả')}

## Mối quan hệ liên kết
"""

    # Thêm các quan hệ mà entity này đóng vai trò source
    if eid in relations_by_source:
        for rel in relations_by_source[eid]:
            target_ent = entities.get(rel["target_id"], {"name": rel["target_id"]})
            content += f"- **{rel['relationship_type']}** -> [[{target_ent['name']}]] *(Chứng cứ: {rel['evidence_quote']} - Độ tin cậy: {rel['confidence']})*\n"

    # Thêm các quan hệ mà entity này đóng vai trò target
    if eid in relations_by_target:
        for rel in relations_by_target[eid]:
            source_ent = entities.get(rel["source_id"], {"name": rel["source_id"]})
            content += f"- Được liên kết bởi **{rel['relationship_type']}** từ [[{source_ent['name']}]] *(Chứng cứ: {rel['evidence_quote']})*\n"

    try:
        with open(filename, mode="w", encoding="utf-8") as f:
            f.write(content)
        page_count += 1
    except Exception as e:
        print(f"Lỗi ghi file {filename}: {e}")

print(f"Đã tạo thành công {page_count} trang Wiki Markdown trong thư mục wiki/.")

# 4. Tạo trang chủ wiki/Home.md
home_content = f"""---
title: Wiki Risk Graph Home
---

# Trợ lý Tri thức Rủi ro (Wiki Risk Graph)

Chào mừng bạn đến với hệ thống quản lý tri thức rủi ro tích hợp đồ thị.

## Thống kê hệ thống
- Tổng số thực thể (Nodes): {len(entities)}
- Tổng số quan hệ (Edges): {sum(len(v) for v in relations_by_source.values())}

## Danh mục tra cứu
- [[risks/RR-001|Danh mục Rủi ro (Risks)]]
- [[controls/KS-001|Danh mục Kiểm soát (Controls)]]
- [[events/SK-001|Danh mục Sự kiện (Events)]]
"""

try:
    with open("wiki/Home.md", mode="w", encoding="utf-8") as f:
        f.write(home_content)
    print("Đã tạo file wiki/Home.md thành công.")
except Exception as e:
    print(f"Lỗi tạo Home.md: {e}")