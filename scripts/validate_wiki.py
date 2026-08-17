import os
import re
import csv

def validate_wiki():
    print("Bắt đầu kiểm tra Wiki Risk Graph...")
    
    # 1. Đọc entities để lấy danh sách ID hợp lệ
    entity_ids = set()
    entity_names = set()
    try:
        with open("outputs/entities.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entity_ids.add(row["id"])
                entity_names.add(row["name"])
    except Exception as e:
        print(f"Lỗi đọc entities.csv: {e}")

    # 2. Kiểm tra các file Markdown trong wiki/
    markdown_files = []
    for root, dirs, files in os.walk("wiki"):
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))

    print(f"Tổng số file Markdown tìm thấy: {len(markdown_files)}")

    wikilink_pattern = re.compile(r'\[\[(.*?)\]\]')
    broken_links = 0
    total_links = 0
    orphan_pages = 0

    report_content = "# Báo cáo Kiểm định Wiki Risk Graph\n\n"
    report_content += f"- Tổng số trang Markdown: {len(markdown_files)}\n"
    report_content += f"- Tổng số thực thể trong DB: {len(entity_ids)}\n\n"
    report_content += "## Chi tiết kiểm tra liên kết\n"

    for file_path in markdown_files:
        if "Home.md" in file_path:
            continue
            
        with open(file_path, mode="r", encoding="utf-8") as f:
            content = f.read()
            
        links = wikilink_pattern.findall(content)
        total_links += len(links)
        
        # Kiểm tra liên kết gãy đơn giản
        for link in links:
            # Xử lý cú pháp [[target|display_text]]
            clean_link = link.split("|")[0].strip()
            # Kiểm tra xem link có khớp với tên entity hoặc id không
            if clean_link not in entity_names and clean_link not in entity_ids and not clean_link.startswith("risks/") and not clean_link.startswith("controls/") and not clean_link.startswith("events/"):
                broken_links += 1
                report_content += f"- [CẢNH BÁO] Trang `{file_path}` chứa liên kết chưa xác thực: `[[{link}]]`\n"

    report_content += f"\n- Tổng số wikilink quét được: {total_links}\n"
    report_content += f"- Số lượng liên kết có thể chưa tồn tại: {broken_links}\n"

    # Đảm bảo thư mục outputs tồn tại và ghi báo cáo
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/wiki_validation_report.md", mode="w", encoding="utf-8") as f:
        f.write(report_content)

    print("Đã tạo báo cáo kiểm định thành công tại outputs/wiki_validation_report.md.")

if __name__ == "__main__":
    validate_wiki()