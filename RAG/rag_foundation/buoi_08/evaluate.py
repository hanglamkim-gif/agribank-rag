import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

from advanced_rag import advanced_generate_answer, load_advanced_config

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "eval"
REPORTS_DIR = BASE_DIR / "reports"

def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 1.0 # If no relevant ids (e.g. out of scope), finding anything or nothing is technically 1.0 recall for that query if it's considered handled? Actually, out of scope usually recall is undefined or 1.0 if we return 0. Let's return 1.0 if no relevant docs.
    
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for cid in relevant_ids if cid in retrieved_k)
    return hits / len(relevant_ids)

def compute_mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
        
    retrieved_k = retrieved_ids[:k]
    for i, cid in enumerate(retrieved_k):
        if cid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def compute_ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
        
    retrieved_k = retrieved_ids[:k]
    dcg = 0.0
    for i, cid in enumerate(retrieved_k):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
            
    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_mode(questions: list[dict], mode: str, strategy: str, k: int) -> dict:
    recalls = []
    mrrs = []
    ndcgs = []
    latencies = []
    errors = 0
    
    for q in questions:
        relevant_ids = q.get("relevant_chunk_ids", [])
        question_text = q["question"]
        
        try:
            # We use advanced_generate_answer but mock generation to save cost
            res = advanced_generate_answer(question_text, strategy, mode=mode, custom_gen_fn=lambda x: "skipped")
            
            # For recall/mrr/ndcg we use all evidence up to k
            # Actually, we should evaluate the retrieval performance, so we take the top k evidence
            retrieved_ids = [ev["chunk_id"] for ev in res["evidence"]]
            
            latencies.append(res["trace"]["latency_ms"]["total"])
            
            # If out of scope (no relevant ids), we might just skip metrics or assume 1.0.
            if not relevant_ids:
                continue
                
            recalls.append(compute_recall_at_k(retrieved_ids, relevant_ids, k))
            mrrs.append(compute_mrr_at_k(retrieved_ids, relevant_ids, k))
            ndcgs.append(compute_ndcg_at_k(retrieved_ids, relevant_ids, k))
            
        except Exception as e:
            print(f"[ERROR] Failed on query {q.get('query_id')}: {e}")
            errors += 1
            
    if not recalls:
        return {
            "Recall@K": 0.0, "MRR@K": 0.0, "nDCG@K": 0.0, 
            "Latency_mean_ms": 0.0, "Latency_p50_ms": 0.0, "errors": errors
        }
        
    latencies.sort()
    p50 = latencies[len(latencies)//2] if latencies else 0.0
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    
    return {
        "Recall@K": sum(recalls) / len(recalls),
        "MRR@K": sum(mrrs) / len(mrrs),
        "nDCG@K": sum(ndcgs) / len(ndcgs),
        "Latency_mean_ms": mean_lat,
        "Latency_p50_ms": p50,
        "errors": errors
    }

def run_evaluation(strategy: str, k: int, modes: list[str] = None):
    q_file = EVAL_DIR / "questions.json"
    if not q_file.exists():
        raise FileNotFoundError(f"Không tìm thấy {q_file}")
        
    with open(q_file, "r", encoding="utf-8-sig") as f:
        questions = json.load(f)
        
    if not modes:
        modes = ["bm25", "semantic", "hybrid", "hybrid_rerank"]
        
    needs_human = any(q.get("needs_human_review", False) for q in questions)
    
    print(f"\n[INFO] Bắt đầu Evaluation cho strategy '{strategy}' với K={k}")
    print(f"Tổng số câu hỏi: {len(questions)}")
    
    results = {}
    for mode in modes:
        print(f"  - Đang chạy mode: {mode}...")
        metrics = evaluate_mode(questions, mode, strategy, k)
        results[mode] = metrics
        
    # Generate report
    config = load_advanced_config()
    report = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy,
        "k": k,
        "config_identity": {
            "embedding": config.get("GEMINI_EMBEDDING_MODEL"),
            "reranker": config.get("RERANKER_MODEL")
        },
        "warnings": [],
        "metrics": results
    }
    
    if needs_human:
        report["warnings"].append("Dataset contains needs_human_review=true. No official winner declared.")
        
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / f"eval_report_{int(time.time())}.json"
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print("\n=== EVALUATION REPORT ===")
    if needs_human:
        print("[WARNING] Bộ gold questions cần human review. Không kết luận mode chiến thắng.")
        
    for m in modes:
        rm = results[m]
        print(f"\nMode: {m}")
        print(f"  Recall@{k}: {rm['Recall@K']:.4f}")
        print(f"  MRR@{k}:    {rm['MRR@K']:.4f}")
        print(f"  nDCG@{k}:   {rm['nDCG@K']:.4f}")
        print(f"  Latency:   {rm['Latency_mean_ms']:.1f}ms (mean) / {rm['Latency_p50_ms']:.1f}ms (p50)")
        if rm["errors"] > 0:
            print(f"  Errors:    {rm['errors']}")
            
    print(f"\n[SUCCESS] Đã lưu báo cáo tại {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Advanced RAG")
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    
    args = parser.parse_args()
    try:
        run_evaluation(args.strategy, args.k)
    except Exception as e:
        print(f"[FATAL] {e}")
