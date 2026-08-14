# Hướng Dẫn Thực Hành RAG (Retrieval-Augmented Generation)

## 1. Mục tiêu
Dự án này là minh họa hệ thống RAG cơ bản bằng Python, bao gồm các chức năng tạo vector nhúng (embedding), lưu trữ vào vector database (ChromaDB), và sinh câu trả lời (generation) với Google Gemini. Mục tiêu giúp người mới hiểu rõ quy trình lưu trữ, truy xuất (retrieval) và ánh xạ trích dẫn (citation) dựa trên dữ liệu thật.

## 2. Quan hệ với Buổi 05 và Buổi 06
- **Buổi 05**: Nơi chứa dữ liệu văn bản đã được phân tách (chunking). RAG (Buổi 07) sẽ trực tiếp đọc dữ liệu `chunks` (JSON) từ `rag_foundation/buoi_05/output/chunks/`.
- **Buổi 06**: (Đã bỏ qua / không nằm trong phạm vi buổi này).
- **Buổi 07**: Chịu trách nhiệm khởi tạo Semantic Index từ dữ liệu của Buổi 05 và cung cấp Pipeline Hỏi - Đáp (RAG) thông qua CLI và giao diện Streamlit.

## 3. Sơ đồ pipeline
```mermaid
graph TD
    A[JSON Chunks từ Buổi 05] -->|Validate| B[ChromaDB Indexing]
    B -->|Gemini Embedding| C[(Chroma PersistentDB)]
    D[Câu hỏi của người dùng] -->|Query Embedding| E[Semantic Search trên Chroma]
    E --> F{Distance < Threshold?}
    F -- Không --> G[Báo lỗi: Thiếu thông tin (retrieval-only)]
    F -- Có --> H[Ghép Context vào Prompt]
    H --> I[Gemini Generation]
    I --> J[Answer + Citations]
```

## 4. Cấu trúc thư mục
```text
buoi_07/
├── rag.py               # Chứa core logic: load, validate, index, search, generate
├── app.py               # Giao diện người dùng bằng Streamlit
├── requirements.txt     # Danh sách thư viện Python cần thiết
├── .env.example         # Template cấu hình môi trường
├── tests/               # Unit test kiểm thử pipeline
└── storage/             # (Tự sinh) ChromaDB lưu trữ SQLite và Vector
```

## 5. Điều kiện đầu vào
- Máy tính có cài đặt Python 3.10 trở lên.
- Đã cài đặt thư viện cần thiết.
- Đã có dữ liệu file `.json` hợp lệ trong folder `buoi_05/output/chunks/`.
- Cần có `GEMINI_API_KEY` từ Google AI Studio.

## 6. Cách dùng `.venv`
Luôn sử dụng Python interpreter của môi trường ảo (virtual environment) thay vì Python hệ thống:
```powershell
c:\agribank-rag\.venv\Scripts\python.exe -m pip list
```
*(Trên hệ thống Windows, dùng đường dẫn tuyệt đối như trên hoặc activate môi trường trước khi chạy)*.

## 7. Cách cài requirements
Mở terminal và chạy lệnh sau để cài đặt các thư viện:
```powershell
c:\agribank-rag\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 8. Cách tạo `.env`
Sao chép file mẫu thành file cấu hình chính thức:
- **Windows PowerShell**: `Copy-Item .env.example .env`
- **Linux/macOS**: `cp .env.example .env`
Mở file `.env` bằng text editor và điền thông tin API Key.

## 9. Giải thích từng biến môi trường
- `GEMINI_API_KEY`: API Key để gọi Gemini, lấy từ Google AI Studio. Bắt buộc phải có để hệ thống hoạt động.
- `GEMINI_EMBEDDING_MODEL`: Tên model dùng tạo vector (vd: `gemini-embedding-2`).
- `GEMINI_EMBEDDING_DIM`: Số chiều (dimension) của vector. Đối với `gemini-embedding-2`, có thể dùng 768.
- `GEMINI_GENERATION_MODEL`: Tên model dùng để sinh câu trả lời (vd: `gemini-3.5-flash-lite`).
- `DEFAULT_TOP_K`: Số lượng tài liệu (evidence) trả về tối đa khi tìm kiếm (mặc định: 5).
- `RAG_MAX_DISTANCE`: Khoảng cách (Cosine distance) tối đa cho phép. Nếu khoảng cách của evidence tốt nhất lớn hơn giá trị này, hệ thống sẽ từ chối trả lời (Confidence Gate).

## 10. Lệnh validate
Kiểm tra tính hợp lệ của dữ liệu đầu vào mà chưa cần index:
```powershell
c:\agribank-rag\.venv\Scripts\python.exe rag.py validate --strategy hierarchical
```

## 11. Lệnh status
Xem trạng thái hiện tại của hệ thống (Model, API Key, số record hiện có):
```powershell
c:\agribank-rag\.venv\Scripts\python.exe rag.py status --strategy hierarchical
```

## 12. Lệnh index
Tạo embeddings và lưu trữ dữ liệu vào ChromaDB:
```powershell
c:\agribank-rag\.venv\Scripts\python.exe rag.py index --strategy hierarchical
```
*Lưu ý: Quá trình này sẽ gọi API Gemini, có thể tốn thời gian tùy thuộc vào số lượng dữ liệu.*

## 13. Lệnh reset đúng collection
Nếu muốn xóa sạch dữ liệu của collection hiện tại (theo strategy) và index lại từ đầu:
```powershell
c:\agribank-rag\.venv\Scripts\python.exe rag.py index --strategy hierarchical --reset
```

## 14. Lệnh query CLI
Hỏi đáp trực tiếp trên terminal:
```powershell
c:\agribank-rag\.venv\Scripts\python.exe rag.py query --strategy hierarchical --top_k 5 "Câu hỏi của bạn?"
```

## 15. Lệnh chạy test
Chạy toàn bộ 47 unit test kiểm thử:
```powershell
c:\agribank-rag\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

## 16. Lệnh chạy Streamlit
Mở giao diện UI trên trình duyệt:
```powershell
c:\agribank-rag\.venv\Scripts\python.exe -m streamlit run app.py
```

## 17. Giải thích các khái niệm
- **strategy**: Phương pháp phân tách dữ liệu (vd: `semantic`, `hierarchical`, `fixed-size`).
- **embedding model**: AI Model có nhiệm vụ chuyển đổi đoạn text thành dãy số (vector).
- **embedding dimension**: Số chiều của dãy số vector.
- **collection identity**: ChromaDB tách dữ liệu thành các "bảng" (collection) phân biệt bằng tên. Tên collection trong hệ thống này được hash (băm) từ strategy, model và dimension để đảm bảo không bị nhầm lẫn dữ liệu giữa các cấu hình khác nhau.
- **top-k**: K giới hạn số lượng kết quả liên quan nhất được lấy ra từ database.
- **cosine distance**: Thước đo khoảng cách giữa 2 vector. Khoảng cách càng nhỏ (gần 0) thì nội dung càng liên quan.
- **RAG_MAX_DISTANCE**: Ngưỡng khoảng cách tối đa. Nếu evidence tìm được vượt ngưỡng này, hệ thống xem như không có thông tin.
- **confidence gate**: Chốt chặn niềm tin; cơ chế chặn sinh câu trả lời rác (hallucination) khi không có context thỏa mãn ngưỡng.
- **retrieval-only**: Trạng thái hệ thống chỉ truy xuất được thông tin nhưng không gọi Generation (hoặc generation bị lỗi).
- **citation**: Ánh xạ bằng chứng (nguồn, trang, id) đính kèm vào câu trả lời để người dùng kiểm chứng.

## 18. Cách dừng Streamlit
Để tắt server UI, bấm tổ hợp phím **Ctrl + C** trên cửa sổ terminal đang chạy.

## 19. Troubleshooting
- **thiếu package**: Nếu báo lỗi `ModuleNotFoundError`, hãy đảm bảo bạn đã dùng đúng Python interpreter (`.venv\Scripts\python.exe`) và đã chạy lệnh `pip install -r requirements.txt`.
- **sai interpreter**: Kiểm tra xem terminal có trỏ đến `c:\agribank-rag\.venv\Scripts\python.exe` hay đang dùng `python` của hệ thống.
- **thiếu API key**: Ứng dụng sẽ báo lỗi hoặc không cho index. Hãy điền `GEMINI_API_KEY` vào `.env`.
- **collection rỗng**: Nếu query báo lỗi empty collection, có nghĩa bạn chưa chạy lệnh `index`.
- **model/dimension mismatch**: Đổi config sẽ tự tạo collection mới. Nếu muốn dùng data cũ, phải dùng đúng cấu hình cũ.
- **JSON lỗi**: Dữ liệu Buổi 05 không đúng định dạng. Cần sửa định dạng JSON hoặc chạy lệnh `validate` để xem chi tiết.
- **embedding lỗi/rate limit**: Google API giới hạn số requests. Nếu bị lỗi, hãy chờ một lát và thử index lại. Code hỗ trợ index lại (idempotent), không bị trùng lặp chunk.

## 20. Giới hạn của demo
- Không có OCR, tài liệu hình ảnh không được trích xuất.
- Không có Reranker để lọc evidence độ chính xác cao.
- Không có Hybrid Search (tìm kiếm từ khóa kết hợp vector).
- Không có phân quyền (RBAC) và chưa sẵn sàng để deploy public.

## 21. Cảnh báo
- **Không phải tư vấn pháp lý**: Các câu trả lời của AI chỉ mang tính tham khảo nội bộ từ tài liệu, không thay thế văn bản quy phạm.
- **Threshold cần hiệu chỉnh**: Tham số `RAG_MAX_DISTANCE` hiện đang set mặc định (0.45). Trên thực tế cần tinh chỉnh theo từng bộ dữ liệu mới chính xác.
- **Retrieval có thể bỏ sót thông tin**: Quá trình tìm kiếm vector có rủi ro không lọt Top-K.
- **Bảo mật dữ liệu**: Nội dung chunk (văn bản thật) được gửi tới server của Google Gemini khi embedding và generation. Tuyệt đối **chỉ dùng dữ liệu mà người vận hành được phép gửi tới dịch vụ đám mây bên ngoài**.

---

## Kế hoạch Kiểm Thử Thủ Công (Manual Test Plan)

A. Có khả năng thuộc tài liệu:
`Cơ cấu lại thời hạn trả nợ được quy định như thế nào?`
*(Dùng UI hoặc CLI để hỏi, kiểm tra xem hệ thống có trả lời và trích xuất đúng nguồn tài liệu hay không)*.

B. Có khả năng thuộc tài liệu:
`Việc phân loại nợ và trích lập dự phòng được thực hiện như thế nào?`
*(Kiểm tra sự kết hợp các nguồn evidence liên quan từ tài liệu)*.

C. Ngoài phạm vi:
`Ngân hàng nào có lãi suất tiết kiệm cao nhất hôm nay?`
- **Kỳ vọng**: Khoảng cách (distance) của evidence tìm được sẽ vượt qua ngưỡng `RAG_MAX_DISTANCE`. Hệ thống sẽ chặn Generation (không gọi LLM) và thông báo: `Không tìm thấy đủ thông tin liên quan trong tài liệu đã cung cấp.` Hệ thống tuyệt đối không được bịa tên ngân hàng hay tự sáng tác lãi suất.
*(Ghi chú: Nếu hệ thống vẫn trả lời, đây được xem là False Positive của Retrieval/Gate do chưa tinh chỉnh threshold, không được can thiệp sửa câu trả lời thủ công).*
