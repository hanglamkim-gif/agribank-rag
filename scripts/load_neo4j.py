import os
import csv
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Tải cấu hình từ file .env
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

def load_data_to_neo4j():
    if not os.path.exists("outputs/entities.csv") or not os.path.exists("outputs/relations.csv"):
        print("Lỗi: Không tìm thấy file outputs/entities.csv hoặc outputs/relations.csv. Hãy chạy các bước trước đó trước.")
        return

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    try:
        driver.verify_connectivity()
        print("Kết nối thành công đến Neo4j!")
    except Exception as e:
        print(f"Không thể kết nối đến Neo4j. Hãy đảm bảo cơ sở dữ liệu đang chạy và cấu hình .env chính xác: {e}")
        driver.close()
        return

    with driver.session(database=DATABASE) as session:
        # 1. Import Nodes
        print("Đang nạp các Nodes vào Neo4j...")
        with open("outputs/entities.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ent_id = row["id"]
                ent_type = row["type"]
                name = row["name"]
                desc = row["description"]
                origin = row["data_origin"]
                status = row["verification_status"]

                # Sử dụng dynamic label an toàn bằng câu lệnh Cypher tương ứng
                if ent_type == "RuiRo":
                    query = """
                    MERGE (n:RuiRo {id: $id})
                    ON CREATE SET n.name = $name, n.description = $description, n.data_origin = $data_origin, n.verification_status = $verification_status
                    ON MATCH SET n.name = $name, n.description = $description, n.data_origin = $data_origin, n.verification_status = $verification_status
                    """
                elif ent_type == "KiemSoat":
                    query = """
                    MERGE (n:KiemSoat {id: $id})
                    ON CREATE SET n.name = $name, n.description = $description, n.data_origin = $data_origin, n.verification_status = $verification_status
                    ON MATCH SET n.name = $name, n.description = $description, n.data_origin = $data_origin, n.verification_status = $verification_status
                    """
                elif ent_type == "SuKienRuiRo":
                    query = """
                    MERGE (n:SuKienRuiRo {id: $id})
                    ON CREATE SET n.name = $name, n.description = $description, n.data_origin = $data_origin, n.verification_status = $verification_status
                    ON MATCH SET n.name = $name, n.description = $description, n.data_origin = $data_origin, n.verification_status = $verification_status
                    """
                else:
                    continue

                session.run(query, id=ent_id, name=name, description=desc, data_origin=origin, verification_status=status)

        # 2. Import Relationships
        print("Đang nạp các Relationships vào Neo4j...")
        with open("outputs/relations.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row["source_id"]
                tgt = row["target_id"]
                rel_type = row["relationship_type"]
                evidence = row["evidence_quote"]
                confidence = float(row["confidence"])
                status = row["verification_status"]

                # Chỉ xử lý các loại quan hệ hợp lệ trong MVP
                if rel_type == "MITIGATES":
                    q = """
                    MATCH (s:KiemSoat {id: $src})
                    MATCH (t:RuiRo {id: $tgt})
                    MERGE (s)-[r:MITIGATES]->(t)
                    SET r.evidence_quote = $evidence, r.confidence = $confidence, r.verification_status = $status
                    """
                elif rel_type == "OBSERVED_AS":
                    q = """
                    MATCH (s:RuiRo {id: $src})
                    MATCH (t:SuKienRuiRo {id: $tgt})
                    MERGE (s)-[r:OBSERVED_AS]->(t)
                    SET r.evidence_quote = $evidence, r.confidence = $confidence, r.verification_status = $status
                    """
                else:
                    continue

                session.run(q, src=src, tgt=tgt, evidence=evidence, confidence=confidence, status=status)

    driver.close()
    print("Hoàn tất nạp dữ liệu thành công vào Neo4j!")

if __name__ == "__main__":
    load_data_to_neo4j()
   