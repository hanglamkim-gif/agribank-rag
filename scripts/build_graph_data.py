import os
import csv

# Đảm bảo thư mục outputs tồn tại
os.makedirs("outputs", exist_ok=True)

# 1. Xây dựng entities.csv từ 3 file seed (risk_profiles, controls, risk_events)
entities = []

# Đọc risk_profiles_seed.csv -> RuiRo
try:
    with open("data/risk_profiles_seed.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entities.append({
                "id": row["id"],
                "type": "RuiRo",
                "name": row["name"],
                "description": row["description"],
                "source_file": "risk_profiles_seed.csv",
                "data_origin": row["data_origin"],
                "verification_status": row["verification_status"]
            })
except Exception as e:
    print(f"Lỗi đọc risk_profiles_seed.csv: {e}")

# Đọc controls_seed.csv -> KiemSoat
try:
    with open("data/controls_seed.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entities.append({
                "id": row["id"],
                "type": "KiemSoat",
                "name": row["name"],
                "description": row["name"], # Sử dụng tên làm mô tả ngắn nếu thiếu
                "source_file": "controls_seed.csv",
                "data_origin": row["data_origin"],
                "verification_status": row["verification_status"]
            })
except Exception as e:
    print(f"Lỗi đọc controls_seed.csv: {e}")

# Đọc risk_events_seed.csv -> SuKienRuiRo
try:
    with open("data/risk_events_seed.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entities.append({
                "id": row["id"],
                "type": "SuKienRuiRo",
                "name": row["description"],
                "description": row["description"],
                "source_file": "risk_events_seed.csv",
                "data_origin": row["data_origin"],
                "verification_status": row["verification_status"]
            })
except Exception as e:
    print(f"Lỗi đọc risk_events_seed.csv: {e}")

# Lưu entities.csv
entity_fields = ["id", "type", "name", "description", "source_file", "data_origin", "verification_status"]
with open("outputs/entities.csv", mode="w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=entity_fields)
    writer.writeheader()
    writer.writerows(entities)

print(f"Đã tạo outputs/entities.csv với tổng số {len(entities)} thực thể.")

# 2. Xử lý relations.csv từ relationships_seed.csv
relations = []
entity_ids = {e["id"] for e in entities}
orphan_count = 0

try:
    with open("data/relationships_seed.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row["source_id"]
            tgt = row["target_id"]
            # Kiểm tra orphan reference
            if src not in entity_ids or tgt not in entity_ids:
                print(f"Cảnh báo: Phát hiện liên kết mồ côi (Orphan): {src} -> {tgt}")
                orphan_count += 1
            
            relations.append({
                "source_id": src,
                "relationship_type": row["relationship_type"],
                "target_id": tgt,
                "source": row["source"],
                "evidence_quote": row["evidence_quote"],
                "confidence": row["confidence"],
                "verification_status": row["verification_status"],
                "data_origin": row["data_origin"]
            })
except Exception as e:
    print(f"Lỗi đọc relationships_seed.csv: {e}")

relation_fields = ["source_id", "relationship_type", "target_id", "source", "evidence_quote", "confidence", "verification_status", "data_origin"]
with open("outputs/relations.csv", mode="w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=relation_fields)
    writer.writeheader()
    writer.writerows(relations)

print(f"Đã tạo outputs/relations.csv với tổng số {len(relations)} quan hệ (Số liên kết mồ côi: {orphan_count}).")
