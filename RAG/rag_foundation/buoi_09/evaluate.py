import json
import time
import uuid
import math
from pathlib import Path
from hierarchical_rag import generate_answer, load_buoi09_config, _load_hierarchy_store

def compute_mrr(retrieved_ids, relevant_ids):
    for idx, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / idx
    return 0.0

def compute_ndcg(retrieved_ids, relevant_ids, k):
    dcg = 0.0
    for idx, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(idx + 1)
            
    idcg = 0.0
    for idx in range(1, min(k, len(relevant_ids)) + 1):
        idcg += 1.0 / math.log2(idx + 1)
        
    return dcg / idcg if idcg > 0 else 0.0

def evaluate():
    base_dir = Path(__file__).resolve().parent
    eval_file = base_dir / "eval" / "questions.json"
    report_dir = base_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    
    if not eval_file.exists():
        print("Không tìm thấy eval/questions.json")
        return
        
    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    config = load_buoi09_config()
    modes = ["single_flat", "multi_flat", "single_parent", "multi_parent"]
    
    # Store validation
    try:
        registry, manifest = _load_hierarchy_store()
    except Exception as e:
        print(f"Lỗi store: {e}")
        return
        
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "MULTI_QUERY_COUNT": config.get("MULTI_QUERY_COUNT", 3),
            "PER_QUERY_CANDIDATES": config.get("PER_QUERY_CANDIDATES", 20),
            "PARENT_CANDIDATES": config.get("PARENT_CANDIDATES", 20),
            "FINAL_PARENT_TOP_K": config.get("FINAL_PARENT_TOP_K", 5),
            "RERANK_MIN_SCORE": config.get("RERANK_MIN_SCORE", 0.50),
            "corpus_identity": manifest.get("fingerprint"),
            "parent_count": manifest.get("parent_count"),
            "child_count": manifest.get("child_count")
        },
        "per_question_results": [],
        "aggregate_metrics_per_mode": {},
        "human_review_warning": any(q.get("needs_human_review", False) for q in questions)
    }
    
    for mode in modes:
        report["aggregate_metrics_per_mode"][mode] = {
            "child_recall_at_k": 0.0,
            "parent_recall_at_k": 0.0,
            "mrr_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "mean_latency_ms": 0.0,
            "mean_context_chars": 0.0,
            "total_queries_generated": 0,
            "total_embedding_calls": 0,
            "success_count": 0,
            "valid_q_count": 0
        }
    
    for q_item in questions:
        q_id = q_item["question_id"]
        q_text = q_item["question"]
        rel_children = set(q_item.get("relevant_child_ids", []))
        rel_parents = set(q_item.get("relevant_parent_ids", []))
        
        q_res = {
            "question_id": q_id,
            "question": q_text,
            "type": q_item.get("question_type"),
            "modes": {}
        }
        
        for mode in modes:
            print(f"Evaluating {q_id} with mode {mode}...")
            # We use retrieval-only by providing custom_gen_fn that returns empty
            res = generate_answer(q_text, config, mode=mode, custom_gen_fn=lambda x: "")
            
            ev = res.get("evidence", [])
            tr = res.get("trace", {})
            status = res.get("status")
            
            lat = tr.get("latency_ms", {}).get("total", 0.0)
            chars = sum(len(e["text"]) for e in ev if e.get("accepted"))
            
            retrieved_c = []
            retrieved_p = []
            
            if "parent" in mode:
                retrieved_p = [e["parent_id"] for e in ev if e.get("accepted")]
                for e in ev:
                    if e.get("accepted"):
                        retrieved_c.extend(e.get("supporting_child_ids", []))
            else:
                retrieved_c = [e["chunk_id"] for e in ev if e.get("accepted")]
                
            child_intersect = len(set(retrieved_c).intersection(rel_children))
            parent_intersect = len(set(retrieved_p).intersection(rel_parents))
            
            c_recall = child_intersect / len(rel_children) if rel_children else (1.0 if not retrieved_c else 0.0)
            p_recall = parent_intersect / len(rel_parents) if rel_parents else (1.0 if not retrieved_p else 0.0)
            
            k = config.get("FINAL_PARENT_TOP_K", 5) if "parent" in mode else config.get("PER_QUERY_CANDIDATES", 20)
            target_rel = rel_parents if "parent" in mode else rel_children
            target_ret = retrieved_p if "parent" in mode else retrieved_c
            
            mrr = compute_mrr(target_ret, target_rel)
            ndcg = compute_ndcg(target_ret, target_rel, k)
            
            q_res["modes"][mode] = {
                "status": status,
                "child_recall": c_recall,
                "parent_recall": p_recall,
                "mrr": mrr,
                "ndcg": ndcg,
                "latency_ms": lat,
                "context_chars": chars,
                "api_calls": tr.get("api_call_counts", {})
            }
            
            # Accumulate
            agg = report["aggregate_metrics_per_mode"][mode]
            agg["latency_ms"] = agg.get("latency_ms", 0.0) + lat
            agg["mean_context_chars"] += chars
            agg["total_queries_generated"] += len(res.get("queries", [])) - 1
            agg["total_embedding_calls"] += len(res.get("queries", []))
            
            if rel_children or rel_parents:
                agg["child_recall_at_k"] += c_recall
                agg["parent_recall_at_k"] += p_recall
                agg["mrr_at_k"] += mrr
                agg["ndcg_at_k"] += ndcg
                agg["valid_q_count"] += 1
                
            if status != "error":
                agg["success_count"] += 1
                
        report["per_question_results"].append(q_res)
        
    for mode in modes:
        agg = report["aggregate_metrics_per_mode"][mode]
        vq = agg["valid_q_count"]
        sq = agg["success_count"]
        if vq > 0:
            agg["child_recall_at_k"] /= vq
            agg["parent_recall_at_k"] /= vq
            agg["mrr_at_k"] /= vq
            agg["ndcg_at_k"] /= vq
        if sq > 0:
            agg["mean_latency_ms"] = agg.get("latency_ms", 0) / sq
            agg["mean_context_chars"] /= sq
        del agg["valid_q_count"]
        if "latency_ms" in agg:
            del agg["latency_ms"]
            
    out_file = report_dir / f"report_{uuid.uuid4().hex[:8]}.json"
    latest_file = report_dir / "latest_report.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"Đã lưu report tại: {out_file.name} và latest_report.json")
    if report["human_review_warning"]:
        print("WARNING: Không khẳng định mode thắng vì dữ liệu cần human_review.")

def cmd_evaluate():
    evaluate()

if __name__ == "__main__":
    cmd_evaluate()
