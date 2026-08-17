import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from neo4j import GraphDatabase

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR.parent / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

SOURCE_DIR = BASE_DIR.parent / "ner_kb"

def load_mini_graph():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    df_meta = pd.read_csv(SOURCE_DIR / "metadata.csv")
    df_content = pd.read_csv(BASE_DIR / "data" / "processed" / "chunks_normalized.csv")
    rel_path = SOURCE_DIR / "relationships.csv"
    df_rel = pd.read_csv(rel_path) if rel_path.exists() else None

    with driver.session(database=NEO4J_DATABASE) as session:
        # Chỉ dọn sạch dữ liệu thuộc Buổi 14
        session.run("MATCH (n {lab_session: 'buoi_14'}) DETACH DELETE n")
        print("🔹 Đã dọn sạch phân vùng dữ liệu lab_session: 'buoi_14'")

        # 1. Tạo node VanBan
        for _, row in df_meta.iterrows():
            doc_id = str(row["id"])
            so_hieu = str(row.get("so_ky_hieu", doc_id))
            loai_vb = str(row.get("loai_van_ban", "Quy định"))
            co_quan = str(row.get("co_quan_ban_hanh", "Agribank"))

            session.run("""
                MERGE (v:VanBan {id: $id, lab_session: 'buoi_14'})
                SET v.title = $title, 
                    v.so_ky_hieu = $so_hieu,
                    v.loai_van_ban = $loai_vb,
                    v.co_quan = $co_quan
            """, id=doc_id, title=f"{loai_vb} {so_hieu}", so_hieu=so_hieu, loai_vb=loai_vb, co_quan=co_quan)

        # 2. Tạo node DieuKhoan và liên kết CONTAINS
        for _, row in df_content.iterrows():
            cid = str(row["chunk_id"])
            doc_id = str(row["document_id"])
            text = str(row["text"])

            session.run("""
                MERGE (d:DieuKhoan {id: $id, lab_session: 'buoi_14'})
                SET d.text = $text, d.document_id = $doc_id
                WITH d
                MATCH (v:VanBan {id: $doc_id, lab_session: 'buoi_14'})
                MERGE (v)-[:CONTAINS {lab_session: 'buoi_14'}]->(d)
            """, id=cid, text=text, doc_id=doc_id)

        # 3. Tạo các liên kết thực tế từ relationships.csv (nếu có)
        if df_rel is not None and not df_rel.empty:
            for _, r in df_rel.iterrows():
                src = str(r.iloc[0])
                tgt = str(r.iloc[1])
                rel_type = "THAM_CHIEU"
                if len(r) > 2 and pd.notna(r.iloc[2]):
                    rel_type = str(r.iloc[2]).strip().upper().replace(" ", "_")

                session.run(f"""
                    MATCH (a {{id: $src, lab_session: 'buoi_14'}}), (b {{id: $tgt, lab_session: 'buoi_14'}})
                    MERGE (a)-[:{rel_type} {{lab_session: 'buoi_14'}}]->(b)
                """, src=src, tgt=tgt)

        print("✅ Nạp Mini Knowledge Graph vào Neo4j thành công!")
    driver.close()

if __name__ == "__main__":
    load_mini_graph()