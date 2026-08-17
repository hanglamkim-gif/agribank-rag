import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

class Neo4jRetriever:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        
        # Khởi tạo kết nối đến Neo4j
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def get_document_info(self, so_ky_hieu):
        """
        Lấy các thông tin liên quan đến một văn bản (Cơ quan ban hành, Người ký, Lĩnh vực)
        và các văn bản có quan hệ với nó (Sửa đổi, Thay thế, Viện dẫn).
        """
        query = """
        MATCH (d:Document {so_ky_hieu: $so_ky_hieu})-[r]->(target)
        RETURN type(r) AS relation_type, 
               labels(target)[0] AS target_type, 
               target.name AS target_name, 
               target.so_ky_hieu AS target_so_ky_hieu
        """
        with self.driver.session() as session:
            result = session.run(query, so_ky_hieu=so_ky_hieu)
            records = [record.data() for record in result]
            return records

# Chạy test thử trực tiếp khi thực thi file này
if __name__ == "__main__":
    retriever = Neo4jRetriever()
    
    # Thử nghiệm với một số ký hiệu có thật trong danh sách vừa in ra
    test_so_ky_hieu = "123/2023/NĐ-CP"
    
    print(f"\nĐang tìm kiếm thông tin quan hệ cho văn bản: {test_so_ky_hieu}")
    relations = retriever.get_document_info(test_so_ky_hieu)
    
    if relations:
        for rel in relations:
            print(f"- {rel['relation_type']} -> [{rel['target_type']}] {rel.get('target_name') or rel.get('target_so_ky_hieu')}")
    else:
        print("Văn bản này không có mối quan hệ hướng ra ngoài nào trong đồ thị.")
        
    retriever.close()
    
    if relations:
        for rel in relations:
            print(f"- {rel['relation_type']} -> [{rel['target_type']}] {rel.get('target_name') or rel.get('target_so_ky_hieu')}")
    else:
        print("Không tìm thấy mối quan hệ nào hoặc sai Số ký hiệu.")
        
    retriever.close()
