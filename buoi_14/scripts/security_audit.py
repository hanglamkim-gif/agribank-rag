import os
import sys
from pathlib import Path

# Cố định số luồng OpenBLAS để tránh lỗi bộ nhớ trên Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.secure_retriever import SecureRetriever

SECURE_CSV = BASE_DIR / "data" / "processed" / "chunks_secure.csv"
REPORT_FILE = BASE_DIR / "outputs" / "security_audit_report.md"

def run_audit():
    retriever = SecureRetriever(SECURE_CSV)
    test_cases = [
        {"id": "SEC-01", "name": "Guest tra cứu Nghị định 123 (Quyền HR)", "query": "Nghị định 123", "roles": ["Guest"], "target_id": "1", "expect_allowed": False},
        {"id": "SEC-02", "name": "Staff tra cứu Nghị định 123 (Quyền HR)", "query": "Nghị định 123", "roles": ["Staff"], "target_id": "1", "expect_allowed": False},
        {"id": "SEC-03", "name": "HR Manager tra cứu Nghị định 123", "query": "Nghị định 123", "roles": ["HR_Manager"], "target_id": "1", "expect_allowed": True},
        {"id": "SEC-04", "name": "Staff tra cứu Thông tư 45 (Quyền Staff)", "query": "Thông tư 45", "roles": ["Staff"], "target_id": "2", "expect_allowed": True},
        {"id": "SEC-05", "name": "Admin tra cứu toàn quyền", "query": "Nghị định 123", "roles": ["Admin"], "target_id": "1", "expect_allowed": True},
    ]

    report_lines = [
        "# BÁO CÁO KIỂM THỬ BẢO MẬT (SECURITY AUDIT) — BUỔI 15\n\n",
        "| Test ID | Tên bài test | Vai trò kiểm thử | Kết quả | Ghi chú |\n",
        "|---|---|---|---|---|\n"
    ]

    print("🚀 Bắt đầu thực hiện Security Integration Tests...\n")
    all_passed = True

    for tc in test_cases:
        results, _ = retriever.search(tc["query"], tc["roles"], method="Hybrid + Rerank", top_k=3)
        retrieved_ids = [str(r["document_id"]) for r in results]

        if tc["expect_allowed"]:
            passed = str(tc["target_id"]) in retrieved_ids
            status = "✅ PASS" if passed else "❌ FAIL"
            note = "Truy cập tài liệu hợp lệ thành công."
        else:
            leakage = str(tc["target_id"]) in retrieved_ids
            passed = not leakage
            status = "✅ PASS" if passed else "🚨 FAIL (LEAKAGE)"
            note = "Chặn truy cập trái phép thành công." if passed else "Cảnh báo: Rò rỉ dữ liệu nhạy cảm!"

        if not passed:
            all_passed = False

        report_lines.append(f"| {tc['id']} | {tc['name']} | {tc['roles']} | {status} | {note} |\n")
        print(f"• [{tc['id']}] {tc['name']} -> {status}")

    report_lines.append("\n## KẾT LUẬN\n\n" + ("✅ **HỆ THỐNG ĐẠT CHỨNG NHẬN AN TOÀN RBAC MỨC CƠ BẢN**." if all_passed else "❌ **HỆ THỐNG CẦN KHẮC PHỤC LỖ HỔNG BẢO MẬT**."))

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"\n✅ Đã xuất báo cáo kiểm định bảo mật tại: {REPORT_FILE}")

if __name__ == "__main__":
    run_audit()