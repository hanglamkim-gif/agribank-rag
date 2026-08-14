# 🚀 Hệ thống RAG Chatbot Nhân sự (Agribank KTNB)

Dự án này là kết quả của quá trình tích hợp kiến trúc **RAG (Retrieval-Augmented Generation)** nhằm xây dựng một trợ lý ảo thông minh, giúp giải đáp và trích xuất các quy định nhân sự nội bộ một cách chính xác, minh bạch và hoàn toàn loại bỏ hiện tượng "ảo giác" (hallucination) của AI.

## 🏗️ Kiến trúc Hệ thống (Pipeline)

Hệ thống RAG được chia thành 4 thành phần lõi (nằm trong thư mục `RAG/rag_foundation/buoi_06/src/`):

1. **Document Loader (`loader.py`) & Validator (`validator.py`)**: 
   - Đọc dữ liệu JSON đầu vào.
   - Sử dụng **Pydantic** để xác thực chặt chẽ cấu trúc văn bản (yêu cầu bắt buộc phải có Số hiệu, Tiêu đề; chuẩn hóa kiểu ngày tháng).
   - Loại bỏ các bản ghi bị lỗi, đảm bảo dữ liệu "sạch" trước khi nạp vào AI.
2. **Vector Database (`embedding.py`)**:
   - Sử dụng **ChromaDB** để lưu trữ cơ sở dữ liệu Vector cục bộ (persistent storage tại `storage/chroma`).
   - Tối ưu hóa việc tìm kiếm các đoạn văn bản (chunks) có ý nghĩa tương đồng với câu hỏi (Semantic Search).
3. **Retriever & Grounding (`retriever.py`)**:
   - Truy xuất `Top K` tài liệu phù hợp nhất.
   - Định dạng văn bản truy xuất thành dạng **Citation/Context** có cấu trúc rõ ràng: `--- Nguồn [Số hiệu]: [Tiêu đề] ---` giúp AI dễ dàng trích dẫn nguồn khi sinh câu trả lời.
4. **LLM Generator (`generator.py`)**:
   - Tích hợp **Google Gemini 1.5 Flash**.
   - Thiết lập **Prompt Engineering** chống ảo giác (Anti-hallucination): Cấm AI tự bịa thông tin và bắt buộc từ chối trả lời nếu dữ liệu không tồn tại trong DB.

## 🛠️ Hướng dẫn Cài đặt & Chạy ứng dụng

### Bước 1: Cấu hình Môi trường
Ứng dụng sử dụng API của Google Gemini. Bạn cần tạo một file `.env` ở thư mục gốc của dự án (`c:\agribank-rag\.env`) với nội dung:
```env
GEMINI_API_KEY=điền_key_của_bạn_vào_đây
```

### Bước 2: Chạy Kiểm thử Tự động (Automated Tests)
Hệ thống được tích hợp bộ Test Suite tiêu chuẩn bằng `unittest` để đánh giá toàn trình. Chạy lệnh sau trong Terminal (đã kích hoạt `.venv`):
```bash
python RAG/rag_foundation/buoi_06/tests/test_suite.py
```
*Kết quả hiển thị `OK` nghĩa là các chức năng Loader, Retriever, và LLM Error Handling đều đang hoạt động hoàn hảo.*

### Bước 3: Trải nghiệm Giao diện Chatbot

Bạn có 2 lựa chọn để tương tác với hệ thống AI này:

**Cách 1: Giao diện Web Độc lập (Streamlit)**
Ứng dụng có một giao diện chatbot phong cách ChatGPT, hỗ trợ lưu trữ lịch sử chat và có tính năng mở/đóng "Nguồn trích dẫn" cực kỳ gọn gàng.
- Khởi chạy lệnh:
  ```bash
  streamlit run streamlit_app.py
  ```
- Truy cập: `http://localhost:8501`

**Cách 2: Giao diện Tích hợp Dashboard (Backend Python thuần)**
Chatbot được gắn dưới dạng "Bong bóng (Widget)" ngay trong trang Dashboard Quản lý Nhân sự của bạn. Backend tự động khởi tạo RAG khi bật server.
- Khởi chạy lệnh:
  ```bash
  python app.py
  ```
- Truy cập: `http://localhost:8000/nhansu`
- Click vào nút 💬 góc phải dưới màn hình để bắt đầu Chat.

---
**🎉 Chúc mừng bạn đã sở hữu một hệ thống AI Trợ lý Nhân sự khép kín, an toàn và cực kỳ mạnh mẽ!**
