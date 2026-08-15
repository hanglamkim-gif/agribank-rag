# Specification Buổi 09: Multi-Query & Hierarchical RAG

## 1. Mục tiêu và khác biệt Buổi 08/09
Buổi 09 nhằm khắc phục các giới hạn của Flat RAG (đã xây dựng ở Buổi 08) thông qua hai cải tiến lớn:
- **Multi-Query Retrieval**: LLM sẽ phân rã câu hỏi ban đầu (Q0) thành nhiều biến thể, qua đó giảm tỷ lệ trượt từ khoá hoặc mất bối cảnh ngữ nghĩa.
- **Hierarchical RAG (Parent-Child)**: Dữ liệu văn bản pháp luật thường mang tính phân cấp (Chương > Điều > Khoản). Khi query khớp với một Khoản (child), hệ thống sẽ tìm ngược lên Điều (parent) để cung cấp trọn vẹn ngữ cảnh pháp lý thay vì chỉ cung cấp một mảnh ghép cắt vụn.

## 2. Sơ đồ Pipeline
`Q0 (Original Query)` → `LLM Query Variants` → `Per-query Hybrid Search (BM25 + Semantic)` → `Cross-query RRF` → `Child-to-Parent Mapping` → `Parent Aggregation` → `Parent Reranker` → `Context Budgeting` → `Generation with Citations`.

## 3. Bốn Mode Truy vấn
- `single_flat`: Chỉ dùng Q0, retrieval trên các chunk nhỏ, không map parent (giống Buổi 08).
- `multi_flat`: Tạo các query variants, retrieval trên chunk nhỏ, không map parent.
- `single_parent`: Chỉ dùng Q0, retrieval trên child chunk rồi map lên parent.
- `multi_parent`: Đầy đủ nhất. Dùng multi-query, retrieval trên child, map lên parent, aggregate, rerank và generation.

## 4. QueryVariant Schema & Validation
```python
class QueryVariant:
    text: str
    weight: float
```
- Validate: Nếu LLM trả rỗng hoặc lỗi, fallback về `Q0` với `weight = MULTI_QUERY_ORIGINAL_WEIGHT`. Số lượng variant tối đa là `MULTI_QUERY_COUNT`. 

## 5. Hierarchy Registry Schema
- Dùng một in-memory dict hoặc JSON để lưu thông tin về các Parent.
- Lưu trữ ánh xạ `child_id -> parent_id`.

## 6. ParentDocument Schema
```python
class ParentDocument:
    parent_id: str
    source: str
    structure_metadata: dict
    full_text: str
    children_ids: list[str]
```

## 7. MultiQueryChildHit & ParentCandidate Schema
- `MultiQueryChildHit`: Lưu chunk_id, số lần xuất hiện ở các variant, cross-query RRF score.
- `ParentCandidate`: Gom nhóm các Child Hit lại, tổng hợp điểm số.

## 8. Quy tắc Hierarchy Resolution và Ambiguous Warning
- Nếu `child_id` thuộc về 1 `parent_id`, fetch text của Parent đó.
- Nếu không tìm thấy Parent hoặc mapping lỗi: Fallback dùng chính `child_id` làm `parent_id` (Text của chính nó).
- Cảnh báo (Ambiguous Warning) nếu phát hiện một Child ánh xạ ra nhiều Parent (bất thường về Data).

## 9. Công thức Cross-Query RRF và Parent Aggregation
- **Cross-Query RRF**: Tính RRF score cho mỗi chunk_id trên toàn bộ các câu hỏi variant, nhân với `weight` của biến thể đó.
- **Parent Aggregation**: `Parent_Score = Max(Child_Scores) + Sum(Child_Scores) * Penalty_Factor`, tuy nhiên ở đây giới hạn tính `PARENT_SCORE_CHILD_LIMIT` top child có điểm cao nhất để cộng dồn, tránh ưu ái các Parent quá dài.

## 10. Context Budget và Citation Contract
- Giới hạn tổng độ dài Context truyền vào LLM theo `TOTAL_CONTEXT_MAX_CHARS` (16000 ký tự). Parent sẽ bị cắt bớt hoặc loại bỏ nếu quá dung lượng.
- LLM sinh ra Citation trỏ tới `parent_id` thay vì `child_id`.

## 11. Status/Failure Contract
- Tương tự Buổi 08, fail an toàn, không im lặng bỏ qua. Báo lỗi `reranker_unavailable` hoặc `insufficient_evidence` khi cần.

## 12. Testability / Dependency Injection
- Mọi hàm gọi LLM (Tạo biến thể, Sinh câu trả lời) và hàm Rerank phải nhận `callable` parameter để Mock trong Unittest mà không cần Internet.

## 13. Evaluation Metrics và Acceptance Criteria
- Đánh giá bằng Recall/MRR/nDCG trên cấp độ *Parent*.
- Pipeline không được crash, tốc độ phải chịu được multi-query.

## 14. Xác nhận Phạm vi
Chỉ ghi và sửa trong thư mục `rag_foundation/buoi_09`. Tuyệt đối không can thiệp Buổi 05-08.
