import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

def verify_graph():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session(database=DATABASE) as session:
        # 1. Đếm số lượng Nodes theo nhãn
        print("--- THỐNG KÊ NODES ---")
        node_result = session.run("MATCH (n) RETURN labels(n) AS label, count(n) as count")
        for record in node_result:
            print(f"Nhãn: {record['label']} - Số lượng: {record['count']}")

        # 2. Đếm số lượng Relationships theo loại
        print("\n--- THỐNG KÊ RELATIONSHIPS ---")
        rel_result = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) as count")
        for record in rel_result:
            print(f"Quan hệ: {record['rel_type']} - Số lượng: {record['count']}")

        # 3. Truy vấn mẫu chuỗi kiểm soát -> rủi ro -> sự kiện
        print("\n--- MẪU ĐƯỜNG ĐI TRUY VẤN (PATH) ---")
        path_result = session.run("""
            MATCH (k:KiemSoat)-[r1:MITIGATES]->(rr:RuiRo)-[r2:OBSERVED_AS]->(sk:SuKienRuiRo)
            RETURN k.name AS kiem_soat, rr.name AS rui_ro, sk.name AS su_kien
        """)
        for i, record in enumerate(path_result, 1):
            print(f"{i}. [Kiểm soát] {record['kiem_soat']} --(MITIGATES)--> [Rủi ro] {record['rui_ro']} --(OBSERVED_AS)--> [Sự kiện] {record['su_kien']}")

    driver.close()

if __name__ == "__main__":
    verify_graph()