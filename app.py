import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ứng dụng Backend REST API Server Quản lý Danh mục Văn bản Nội bộ (Agribank KTNB)
Sử dụng thư viện chuẩn của Python (http.server) - Khởi chạy dữ liệu RAM.
Đã bổ sung xử lý các trường hợp biên (Edge Cases):
1. Tìm kiếm thông minh hỗ trợ gõ tiếng Việt có dấu & không dấu.
2. Kiểm tra chặt chẽ điều kiện số hiệu/tiêu đề trống hoặc trùng lặp.
3. Kiểm tra tính hợp lệ của định dạng ngày ban hành.
"""

import http.server
import socketserver
import json
import os
import urllib.parse
import re
import sys
import io
import unicodedata

# Thêm import cho hệ thống RAG
from RAG.rag_foundation.buoi_06.src.loader import DocumentLoader
from RAG.rag_foundation.buoi_06.src.embedding import RAGVectorStore
from RAG.rag_foundation.buoi_06.src.retriever import DocumentRetriever
from RAG.rag_foundation.buoi_06.src.generator import RAGGenerator

# Biến toàn cục RAG
rag_retriever = None
rag_generator = None

# Đảm bảo Windows console in UTF-8 không bị lỗi charmap
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PORT = 8000
DATA_FILE = os.path.join(os.path.dirname(__file__), "van_ban_final.json")

# Kho lưu trữ dữ liệu trong bộ nhớ (In-memory storage)
danh_sach_van_ban = []

def normalize_text(text):
    """Bỏ dấu tiếng Việt và đưa về chữ thường để tìm kiếm thông minh"""
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.replace('đ', 'd').replace('Đ', 'd')
    return text.strip()

def is_valid_date(date_str):
    """Kiểm tra ngày ban hành có đúng định dạng YYYY-MM-DD hoặc DD/MM/YYYY"""
    if not date_str or not date_str.strip():
        return True # Cho phép để trống ngày ban hành
    date_str = date_str.strip()
    
    # Định dạng YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        try:
            year, month, day = map(int, date_str.split('-'))
            return 1 <= month <= 12 and 1 <= day <= 31 and year > 1900
        except ValueError:
            return False
            
    # Định dạng DD/MM/YYYY
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", date_str):
        try:
            day, month, year = map(int, date_str.split('/'))
            return 1 <= month <= 12 and 1 <= day <= 31 and year > 1900
        except ValueError:
            return False
            
    return False

def doc_du_lieu_ban_dau():
    """Tải dữ liệu ban đầu từ van_ban_final.json vào bộ nhớ RAM."""
    global danh_sach_van_ban
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                danh_sach_van_ban = json.load(f)
                print(f"[KHỞI TẠO] Đã tải {len(danh_sach_van_ban)} văn bản từ file vào bộ nhớ RAM.")
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể đọc file dữ liệu: {e}")
            danh_sach_van_ban = []
    else:
        danh_sach_van_ban = [
            {
                "id": 1,
                "so_hieu": "15/2026/QĐ-KTNB",
                "tieu_de": "Quy định kiểm toán nội bộ về an toàn thông tin và bảo mật dữ liệu",
                "ngay_ban_hanh": "2026-03-15",
                "con_hieu_luc": True
            }
        ]

def init_rag_system():
    """Khởi tạo hệ thống RAG (Load dữ liệu, ChromaDB, và LLM)"""
    global rag_retriever, rag_generator
    try:
        print("[RAG] Đang khởi tạo hệ thống AI Chatbot...")
        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp_rag_data")
        os.makedirs(tmp_dir, exist_ok=True)
        with open(os.path.join(tmp_dir, "data.json"), "w", encoding="utf-8") as f:
            json.dump(danh_sach_van_ban, f, ensure_ascii=False)
        
        loader = DocumentLoader(tmp_dir)
        valid_chunks = loader.load_all()
        
        vector_store = RAGVectorStore(collection_name="agribank_app_collection")
        vector_store.add_documents(valid_chunks)
        
        rag_retriever = DocumentRetriever(vector_store)
        rag_generator = RAGGenerator()
        print("[RAG] Khởi tạo hệ thống AI Chatbot thành công!")
    except Exception as e:
        print(f"[RAG CẢNH BÁO] Khởi tạo RAG thất bại (Chatbot có thể không hoạt động): {e}")

class VanBanRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Bộ xử lý Yêu cầu HTTP cho Web App & REST API"""

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # Serving API: Danh sách văn bản
        if path == "/api/vanban":
            query_params = urllib.parse.parse_qs(parsed_url.query)
            search_raw = query_params.get("search", [""])[0]
            search_norm = normalize_text(search_raw)
            status_filter = query_params.get("status", ["all"])[0].strip()

            ket_qua = []
            for vb in danh_sach_van_ban:
                so_hieu_norm = normalize_text(vb.get("so_hieu", ""))
                tieu_de_norm = normalize_text(vb.get("tieu_de", ""))

                # Tìm kiếm hỗ trợ gõ không dấu lẫn có dấu
                matches_search = (
                    not search_norm or 
                    search_norm in so_hieu_norm or 
                    search_norm in tieu_de_norm
                )

                # Lọc theo trạng thái hiệu lực (xử lý chuẩn kiểu dữ liệu boolean/string)
                is_active = (vb.get("con_hieu_luc") is True or vb.get("con_hieu_luc") == "true" or vb.get("con_hieu_luc") == 1)
                matches_status = True
                if status_filter == "con_hieu_luc":
                    matches_status = (is_active is True)
                elif status_filter == "het_hieu_luc":
                    matches_status = (is_active is False)

                if matches_search and matches_status:
                    ket_qua.append(vb)

            self.gui_response_json(200, ket_qua)
            return

        # Serving static index.html cho trang chủ
        if path == "/" or path == "/index.html":
            self.path = "/index.html"
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
            
        if path == "/nhansu" or path == "/nhansu/":
            self.path = "/nhansu-app/index.html"
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        """Xử lý Thêm văn bản hoặc Chat API"""
        if self.path == "/api/chat":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(body)
                query = data.get("query", "").strip()
                if not query:
                    self.gui_response_json(400, {"loi": "Câu hỏi không được để trống!"})
                    return
                
                if not rag_retriever or not rag_generator:
                    self.gui_response_json(500, {"loi": "Hệ thống RAG chưa sẵn sàng!"})
                    return
                
                print(f"[CHAT] Đang xử lý câu hỏi: {query}")
                context = rag_retriever.retrieve_and_format(query, top_k=2)
                answer = rag_generator.generate_answer(query, context)
                
                self.gui_response_json(200, {"answer": answer, "context": context})
            except Exception as e:
                self.gui_response_json(500, {"loi": f"Lỗi xử lý Chat: {str(e)}"})
            return

        if self.path == "/api/vanban":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')

            try:
                data = json.loads(body)
                
                so_hieu = str(data.get("so_hieu", "")).strip()
                tieu_de = str(data.get("tieu_de", "")).strip()
                ngay_ban_hanh = str(data.get("ngay_ban_hanh", "")).strip()
                con_hieu_luc = bool(data.get("con_hieu_luc", True))

                # XỬ LÝ BÊN 1: Thiếu số hiệu hoặc tiêu đề
                if not so_hieu:
                    self.gui_response_json(400, {"loi": "Số hiệu văn bản không được để trống!"})
                    return

                if not tieu_de:
                    self.gui_response_json(400, {"loi": "Tiêu đề văn bản không được để trống!"})
                    return

                # XỬ LÝ BIÊN 2: Kiểm tra ngày ban hành sai định dạng
                if ngay_ban_hanh and not is_valid_date(ngay_ban_hanh):
                    self.gui_response_json(400, {"loi": "Ngày ban hành sai định dạng! Vui lòng chọn ngày hợp lệ (YYYY-MM-DD)."})
                    return

                # XỬ LÝ BIÊN 3: Kiểm tra trùng số hiệu văn bản
                for vb in danh_sach_van_ban:
                    if vb.get("so_hieu", "").lower() == so_hieu.lower():
                        self.gui_response_json(400, {"loi": f"Số hiệu văn bản '{so_hieu}' đã tồn tại trong hệ thống!"})
                        return

                max_id = max([vb.get("id", 0) for vb in danh_sach_van_ban], default=0)
                new_doc = {
                    "id": max_id + 1,
                    "so_hieu": so_hieu,
                    "tieu_de": tieu_de,
                    "ngay_ban_hanh": ngay_ban_hanh,
                    "con_hieu_luc": con_hieu_luc
                }

                danh_sach_van_ban.append(new_doc)
                print(f"[THÊM VĂN BẢN] {so_hieu} - {tieu_de}")
                self.gui_response_json(201, {"thong_bao": "Thêm văn bản thành công", "van_ban": new_doc})

            except Exception as e:
                self.gui_response_json(400, {"loi": f"Dữ liệu gửi lên không hợp lệ: {str(e)}"})
            return

        self.gui_response_json(404, {"loi": "Endpoint không tồn tại"})

    def do_PUT(self):
        """Cập nhật thông tin văn bản trong bộ nhớ"""
        match = re.match(r"^/api/vanban/(\d+)$", self.path)
        if match:
            doc_id = int(match.group(1))
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')

            try:
                data = json.loads(body)
                target_doc = None
                for vb in danh_sach_van_ban:
                    if vb.get("id") == doc_id:
                        target_doc = vb
                        break

                if not target_doc:
                    self.gui_response_json(404, {"loi": "Không tìm thấy văn bản với ID tương ứng"})
                    return

                new_so_hieu = str(data.get("so_hieu", target_doc["so_hieu"])).strip()
                new_tieu_de = str(data.get("tieu_de", target_doc["tieu_de"])).strip()
                new_ngay_ban_hanh = str(data.get("ngay_ban_hanh", target_doc.get("ngay_ban_hanh", ""))).strip()

                if not new_so_hieu:
                    self.gui_response_json(400, {"loi": "Số hiệu văn bản không được để trống!"})
                    return

                if not new_tieu_de:
                    self.gui_response_json(400, {"loi": "Tiêu đề văn bản không được để trống!"})
                    return

                if new_ngay_ban_hanh and not is_valid_date(new_ngay_ban_hanh):
                    self.gui_response_json(400, {"loi": "Ngày ban hành sai định dạng! Vui lòng chọn ngày hợp lệ (YYYY-MM-DD)."})
                    return

                # Kiểm tra trùng số hiệu với văn bản khác
                for vb in danh_sach_van_ban:
                    if vb.get("id") != doc_id and vb.get("so_hieu", "").lower() == new_so_hieu.lower():
                        self.gui_response_json(400, {"loi": f"Số hiệu văn bản '{new_so_hieu}' đã trùng với một văn bản khác!"})
                        return

                target_doc["so_hieu"] = new_so_hieu
                target_doc["tieu_de"] = new_tieu_de
                target_doc["ngay_ban_hanh"] = new_ngay_ban_hanh
                if "con_hieu_luc" in data:
                    target_doc["con_hieu_luc"] = bool(data["con_hieu_luc"])

                print(f"[CẬP NHẬT VĂN BẢN ID={doc_id}] {target_doc['so_hieu']}")
                self.gui_response_json(200, {"thong_bao": "Cập nhật thành công", "van_ban": target_doc})

            except Exception as e:
                self.gui_response_json(400, {"loi": f"Dữ liệu không hợp lệ: {str(e)}"})
            return

        self.gui_response_json(404, {"loi": "Endpoint không tồn tại"})

    def do_DELETE(self):
        """Xóa văn bản khỏi bộ nhớ"""
        global danh_sach_van_ban
        match = re.match(r"^/api/vanban/(\d+)$", self.path)
        if match:
            doc_id = int(match.group(1))
            ban_dau = len(danh_sach_van_ban)
            danh_sach_van_ban = [vb for vb in danh_sach_van_ban if vb.get("id") != doc_id]

            if len(danh_sach_van_ban) < ban_dau:
                print(f"[XÓA VĂN BẢN ID={doc_id}] Xóa thành công.")
                self.gui_response_json(200, {"thong_bao": "Đã xóa văn bản thành công"})
            else:
                self.gui_response_json(404, {"loi": "Không tìm thấy văn bản để xóa"})
            return

        self.gui_response_json(404, {"loi": "Endpoint không tồn tại"})

    def gui_response_json(self, code, data):
        """Hàm tiện ích gửi kết quả JSON cho Client"""
        self.send_response(code)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

def run_server():
    doc_du_lieu_ban_dau()
    init_rag_system()
    handler = VanBanRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print("=" * 65)
        print(f"🚀 SERVER QUẢN LÝ VĂN BẢN AGRIBANK KTNB ĐÃ KHỞI CHẠY!")
        print(f"👉 Mở trình duyệt và truy cập: http://localhost:{PORT}")
        print("=" * 65)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[ĐÃ DỪNG] Server đã đóng thành công.")

if __name__ == "__main__":
    run_server()
