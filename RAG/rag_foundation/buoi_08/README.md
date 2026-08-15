# Advanced RAG Project (Buổi 08)

## 1. Mục tiêu và khác biệt Buổi 07/08
Khác với kiến trúc đơn giản ở Buổi 07 (chỉ sử dụng Semantic Search bằng ChromaDB), Buổi 08 triển khai hệ thống **Advanced RAG** với luồng Multi-stage Retrieval hoàn chỉnh, kết hợp Hybrid Search (BM25 + Semantic) và Re-ranking bằng Cross-Encoder model. Giao diện cũng được nâng cấp lên dạng Diagnostic UI, hỗ trợ truy vấn chẩn đoán các thông số dưới mui xe.

## 2. Sơ đồ Kiến trúc
`User Query` → (Lexical BM25 Search + Vector Semantic Search) 
             → `Reciprocal Rank Fusion (RRF)` 
             → `Cross-Encoder Reranker` 
             → `Threshold Gating` 
             → `Gemini LLM Generation (with Citations)`

## 3. Cấu trúc Project
```text
rag_foundation/buoi_08/
├── app.py                  # Giao diện Streamlit UI
├── advanced_rag.py         # Advanced RAG core (BM25, RRF, Reranker, Generation)
├── evaluate.py             # Script đánh giá Recall/MRR/nDCG
├── rag.py                  # Base library (từ Buổi 07: load chunk, semantic, config)
├── requirements.txt        # Các dependency
├── .env.example            # Template biến môi trường
├── README.md               # Tài liệu này
├── eval/
│   └── questions.json      # Gold dataset cho evaluate
├── reports/                # Nơi lưu JSON report từ evaluate.py
├── storage/                # Lưu ChromaDB và HF Model Cache
└── tests/                  # Bộ unittest đầy đủ
```

## 4. Setup
1. Tạo môi trường ảo: `python -m venv .venv`
2. Kích hoạt môi trường: `.venv\Scripts\activate` (Windows) hoặc `source .venv/bin/activate` (Mac/Linux)
3. Cài đặt thư viện: `pip install -r requirements.txt`
4. Copy `.env.example` thành `.env` và điền `GEMINI_API_KEY`.

## 5. Cảnh báo Tài nguyên Reranker
Mô hình Reranker mặc định (`BAAI/bge-reranker-v2-m3`) là một Cross-Encoder. Trong lần chạy đầu tiên, hệ thống sẽ tải khoảng 2GB model từ Hugging Face về thư mục `storage/huggingface/`.
Quá trình này cần kết nối Internet mạnh, ổ cứng đủ trống và tiêu tốn nhiều RAM/VRAM khi inference. Hãy sử dụng tham số `RERANK_DEVICE` trong `.env` để kiểm soát (`auto`, `cuda`, hoặc `cpu`).

## 6. Các lệnh CLI chính (advanced_rag.py)
*Lưu ý: Thay `hierarchical` bằng tên strategy bạn muốn.*
- Xem trạng thái: `python advanced_rag.py status --strategy hierarchical`
- Tạo index Semantic: `python advanced_rag.py prepare-semantic --strategy hierarchical`
- Search BM25: `python advanced_rag.py bm25 --strategy hierarchical --question "Điều 7 quy định gì?"`
- Hybrid (BM25 + Semantic + RRF): `python advanced_rag.py hybrid --strategy hierarchical --question "..."`
- Rerank (Hybrid + Reranker): `python advanced_rag.py rerank --strategy hierarchical --question "..."`
- So sánh Rank (Compare): `python advanced_rag.py compare --strategy hierarchical --question "..."`
- Hỏi đáp đầy đủ (Query): `python advanced_rag.py query --mode hybrid_rerank --strategy hierarchical --question "..."`

## 7. Các lệnh Test, Evaluate & UI
- Chạy toàn bộ Unittest: `python -m unittest discover tests -v`
- Chạy Evaluation: `python evaluate.py --strategy hierarchical --k 5`
- Mở Streamlit App: `python -m streamlit run app.py`

## 8. Giải thích các loại Điểm số
- **BM25 Score**: Điểm số Lexical đo mức độ trùng khớp từ khóa. Khoảng giá trị mở (không giới hạn trên). Cao hơn là tốt hơn.
- **Cosine Distance (Semantic)**: Khoảng cách không gian vector. Nhỏ hơn là tốt hơn (0 = giống hệt).
- **RRF Score**: Điểm hợp nhất nghịch đảo thứ hạng: `w / (k + rank)`. Cao hơn là tốt hơn, giới hạn trên phụ thuộc vào `RRF_K`.
- **Rerank Score**: Logit gốc của mô hình Cross-encoder được đưa qua hàm Sigmoid để ép về `[0, 1]`. Cao hơn là tốt hơn, nhưng **không phải là xác suất đúng tuyệt đối**.

## 9. Tham số Candidate K và Final K
- `BM25_CANDIDATES` & `SEMANTIC_CANDIDATES`: Số lượng chunk lấy từ mỗi nhánh trước khi gộp (Fusion).
- `RERANK_CANDIDATES`: Số lượng chunk sau khi Fusion được đẩy vào mô hình Reranker.
- `FINAL_TOP_K`: Số chunk cuối cùng (top đầu) được trả về hoặc đưa vào LLM Context.

## 10. Evaluation Metrics
- `Recall@K`: Tỷ lệ các chunk phù hợp (relevant) có mặt trong top K trả về.
- `MRR@K`: Trung bình nghịch đảo thứ hạng của chunk phù hợp xuất hiện đầu tiên.
- `nDCG@K`: Chất lượng xếp hạng dựa trên vị trí của các chunk phù hợp, có chiết khấu theo log.
*(Lưu ý: Nếu dataset có `needs_human_review=true`, report chỉ mang tính tham khảo và không kết luận mode chiến thắng vì dữ liệu gold chưa được con người nghiệm thu tuyệt đối).*

## 11. Troubleshooting
- **Lỗi tải Reranker**: Kiểm tra kết nối mạng. Nếu tải bị gián đoạn, xóa thư mục `storage/huggingface` và chạy lại.
- **CPU quá chậm**: Chuyển đổi `RERANK_BATCH_SIZE=1` để tránh nghẽn RAM, hoặc dùng thiết bị có GPU CUDA.
- **Thiếu RAM (OOM)**: Giảm `RERANK_CANDIDATES` xuống (ví dụ 10 hoặc 5).
- **Lỗi API Gemini**: Xác nhận `GEMINI_API_KEY` trong `.env` hợp lệ. Nếu gặp lỗi Limit, hãy chờ vài phút.

## 12. Miễn trừ trách nhiệm
Mọi kết quả sinh ra từ hệ thống chỉ nhằm mục đích nghiên cứu công nghệ Retrieval-Augmented Generation, **KHÔNG PHẢI LÀ TƯ VẤN PHÁP LÝ**. Vui lòng luôn tham chiếu đến văn bản pháp luật gốc có hiệu lực.

## 13. Câu hỏi so sánh thủ công (Manual Comparison Questions)
Bạn có thể dùng lệnh `compare` với các câu hỏi sau để thấy sự phân tầng rõ rệt:
- **Exact legal reference**: `Điều 7 quy định như thế nào về cơ cấu lại thời hạn trả nợ?` (BM25 thường làm tốt).
- **Paraphrase semantic**: `Khách hàng gặp khó khăn có thể được điều chỉnh kỳ hạn trả nợ ra sao?` (Semantic làm tốt).
- **Multi-concept**: `Phân loại nợ và trích lập dự phòng được thực hiện như thế nào?` (Hybrid làm tốt).
- **Out-of-scope**: `Ngân hàng nào có lãi suất tiết kiệm cao nhất hôm nay?` (Hệ thống phải tự chối).
