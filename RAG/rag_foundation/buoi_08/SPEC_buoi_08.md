# SPEC Buổi 08: Advanced RAG
Tài liệu đặc tả kiến trúc cho Buổi 08: BM25, Semantic, RRF, Cross-Encoder Reranker.

1. Workspace và security: Chỉ hoạt động trong buoi_08. Không dùng dữ liệu thật/nhạy cảm.
2. Quan hệ với Buổi 05 và Buổi 07: Kế thừa dữ liệu Buổi 05, dùng code Buổi 07 làm baseline tham chiếu.
3. Data contract: Chunks dùng schema từ json.
4. BM25 tokenizer/retrieval contract: Keyword search.
5. Semantic candidate contract: Vector search.
6. RRF fusion contract: Kết hợp điểm BM25 và Semantic bằng Reciprocal Rank Fusion.
7. Cross-encoder reranker contract: Đánh giá lại thứ tự bằng mô hình Cross-encoder.
8. Final evidence và citation contract: Kết quả cuối cùng và mapping trích dẫn.
9. Pipeline trace contract: Logging toàn trình.
10. Evaluation metrics contract: Đo lường MRR, NDCG.
11. Offline testing contract: Fixtures mô phỏng không gọi API.
12. UI comparison contract: Giao diện so sánh Baseline vs Advanced RAG.
