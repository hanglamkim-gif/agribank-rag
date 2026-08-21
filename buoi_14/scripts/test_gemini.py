import google.generativeai as genai

# Gắn trực tiếp chiếc chìa khóa của bạn
API_KEY = "API_KEY_CUA_TOI"
genai.configure(api_key=API_KEY)

print("Đang quét danh sách mô hình được cấp phép cho API Key này...")
try:
    models = list(genai.list_models())
    if not models:
        print("❌ CẢNH BÁO: API Key không bị sai, nhưng Google không cấp quyền cho bất kỳ model nào (Danh sách rỗng).")
    else:
        print("✅ Thành công! Các model bạn có quyền sử dụng là:")
        for m in models:
            # Lọc ra các model hỗ trợ tạo văn bản (generateContent)
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
except Exception as e:
    print("❌ LỖI XÁC THỰC TÀI KHOẢN:", e)