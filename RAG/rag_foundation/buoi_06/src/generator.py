import os
import google.generativeai as genai
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

class RAGGenerator:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Khởi tạo Generator với Gemini LLM"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Không tìm thấy GEMINI_API_KEY trong cấu hình!")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_answer(self, query: str, context: str) -> str:
        """Sinh câu trả lời dựa vào Context (Grounding)"""
        
        prompt = f"""Bạn là trợ lý nhân sự ảo của ngân hàng Agribank (agribank-nhansu). Nhiệm vụ của bạn là giải đáp các thắc mắc về quy định, quy chế nhân sự cho cán bộ công nhân viên.

YÊU CẦU QUAN TRỌNG NHẤT (CHỐNG ẢO GIÁC - HALLUCINATION):
1. Bạn CHỈ ĐƯỢC PHÉP dựa vào các thông tin trong phần "TÀI LIỆU THAM KHẢO" bên dưới để trả lời.
2. Tuyệt đối KHÔNG tự bịa ra thông tin. Nếu trong tài liệu không có thông tin để trả lời, hãy nói thẳng: "Xin lỗi, hiện tại tôi chưa có thông tin quy định về vấn đề này trong cơ sở dữ liệu."
3. Bắt buộc TRÍCH DẪN NGUỒN khi sử dụng thông tin. Ví dụ: "(Theo Số hiệu: QD-..., Tiêu đề: ...)"

TÀI LIỆU THAM KHẢO (Đã được truy xuất - Grounding Context):
{context}

CÂU HỎI CỦA NGƯỜI DÙNG: 
{query}

CÂU TRẢ LỜI CỦA BẠN:
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"[LỖI LLM GENERATION]: {str(e)}"
