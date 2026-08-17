import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "chunks_normalized.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "chunks_secure.csv"

def determine_roles(row):
    cid = str(row["chunk_id"])
    # Phân bổ 3 cấp độ bảo mật để kiểm thử RBAC
    if cid == "1":
        # Tài liệu mức Bảo mật cao: Chỉ Admin và HR
        return ["Admin", "HR_Manager"]
    elif cid == "2":
        # Tài liệu mức Nghiệp vụ: Admin, Risk Officer và Staff
        return ["Admin", "Risk_Officer", "Staff"]
    else:
        # Tài liệu mức Công khai: Toàn bộ các vai trò
        return ["Admin", "HR_Manager", "Risk_Officer", "Staff", "Guest"]

def main():
    df = pd.read_csv(INPUT_PATH)
    df["allowed_roles"] = df.apply(determine_roles, axis=1)
    df["allowed_roles_str"] = df["allowed_roles"].apply(json.dumps)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    
    print(f"✅ Đã phân cấp bảo mật cho {len(df)} chunks -> {OUTPUT_PATH}")
    for _, row in df.iterrows():
        print(f"• Chunk [{row['chunk_id']}] {row['title']} -> Quyền xem: {row['allowed_roles']}")

if __name__ == "__main__":
    main()