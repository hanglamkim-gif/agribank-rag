import os
import sys
from pathlib import Path

# Giới hạn số luồng tính toán để tránh tràn bộ nhớ OpenBLAS
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
from tabulate import tabulate

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import NeuralReranker

CORPUS_PATH = BASE_DIR / "data" / "processed" / "chunks_normalized.csv"
OUTPUT_REPORT = BASE_DIR / "outputs" / "evaluation_report.md"

def evaluate():
    print("1. Đang tải dữ liệu và khởi tạo 4 mô hình Retrieval...")
    df = pd.read_csv(CORPUS_PATH)
    bm25 = BM25Retriever(df)
    dense = DenseRetriever(df, cache_dir=BASE_DIR / "cache")
    hybrid = HybridRetriever(bm25, dense)
    reranker = NeuralReranker()

    queries = [
        ("Nghị định 123/2023/NĐ-CP", "EXACT_KEYWORD"),
        ("Sửa đổi bổ sung một số điều của nghị định cũ", "SEMANTIC"),
        ("Bãi bỏ Thông tư số 12/2021/TT-BTC", "MIXED")
    ]

    report_lines = ["# BÁO CÁO ĐÁNH GIÁ RETRIEVAL — BUỔI 14\n\n"]
    
    print("2. Đang thực thi đánh giá so sánh từng câu truy vấn...")
    for q, q_type in queries:
        report_lines.append(f"## Query: {q} ({q_type})\n\n")
        b_res = bm25.search(q, top_k=2)
        d_res = dense.search(q, top_k=2)
        h_res = hybrid.search(q, candidate_k=5, top_k=2)
        r_res = reranker.rerank(q, h_res, top_k=2)

        table_data = []
        for i in range(2):
            table_data.append([
                i + 1,
                b_res[i]['chunk_id'] if i < len(b_res) else "-",
                d_res[i]['chunk_id'] if i < len(d_res) else "-",
                h_res[i]['chunk_id'] if i < len(h_res) else "-",
                r_res[i]['chunk_id'] if i < len(r_res) else "-"
            ])
        
        table_md = tabulate(table_data, headers=["Rank", "BM25", "Dense", "Hybrid", "Hybrid+Rerank"], tablefmt="github")
        report_lines.append(table_md)
        report_lines.append("\n\n")

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"✅ Đã tạo thành công báo cáo đánh giá tại: {OUTPUT_REPORT}")

if __name__ == "__main__":
    evaluate()