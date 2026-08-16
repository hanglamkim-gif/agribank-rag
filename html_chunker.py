import json
from bs4 import BeautifulSoup

html_content = """
Chương 1: Quy định chung
Mục 1: Phạm vi điều chỉnh
Điều 1: Phạm vi
Luật này quy định về các hoạt động ngân hàng và tổ chức tín dụng.
Các tổ chức tín dụng phải tuân thủ nghiêm ngặt các quy định này.
"""

def test_chunking(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    chunks = []
    current_chapter = None
    current_section = None
    current_article = None

    for element in soup.find_all(['h1', 'h2', 'h3', 'p']):
        text = element.get_text(strip=True)
        if not text:
            continue

        if element.name == 'h1':
            current_chapter = text
            current_section = None
            current_article = None
            chunk_type = 'Chương'
            parent = 'Document'
        elif element.name == 'h2':
            current_section = text
            current_article = None
            chunk_type = 'Mục'
            parent = current_chapter
        elif element.name == 'h3':
            current_article = text
            chunk_type = 'Điều'
            parent = current_section if current_section else current_chapter
        elif element.name == 'p':
            chunk_type = 'Đoạn văn'
            parent = current_article

        chunk_data = {
            'id': f"chunk_{len(chunks)}",
            'type': chunk_type,
            'content': text,
            'parent': parent
        }
        chunks.append(chunk_data)
    return chunks

results = test_chunking(html_content)
print(f"Tổng số chunks bóc tách được: {len(results)}")
print(json.dumps(results, indent=4, ensure_ascii=False))
