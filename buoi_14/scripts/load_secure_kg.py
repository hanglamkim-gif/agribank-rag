import json
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR.parent / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
SECURE_CSV = BASE_DIR / "data" / "processed" / "chunks_secure.csv"

def update_graph_security():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    df = pd.read_csv(SECURE_CSV)

    with driver.session(database=NEO4J_DATABASE) as session:
        print("🔹 Đang cập nhật quyền allowed_roles vào Neo4j...")
        for _, row in df.iterrows():
            cid = str(row["chunk_id"])
            doc_id = str(row["document_id"])
            roles = json.loads(row["allowed_roles_str"])

            session.run("""
                MATCH (d:DieuKhoan {id: $cid, lab_session: 'buoi_14'})
                SET d.allowed_roles = $roles
                WITH d
                MATCH (v:VanBan {id: $doc_id, lab_session: 'buoi_14'})
                SET v.allowed_roles = $roles
            """, cid=cid, doc_id=doc_id, roles=roles)

        result = session.run("""
            MATCH (n {lab_session: 'buoi_14'})
            WHERE n.allowed_roles IS NOT NULL
            RETURN count(n) AS secured_nodes
        """)
        count = result.single()["secured_nodes"]
        print(f"✅ Đã gắn quyền RBAC cho {count} nodes trên Neo4j!")
    driver.close()

if __name__ == "__main__":
    update_graph_security()