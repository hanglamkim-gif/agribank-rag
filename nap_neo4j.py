from neo4j import GraphDatabase

# BƯỚC 3: CẤU HÌNH KẾT NỐI (Đã điền sẵn mật khẩu của bạn)
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "Kh@141086")

# Dữ liệu giả lập đã bóc tách
chunks = [
    {"id": "chunk_0", "type": "Chương", "content": "Chương 1: Quy định chung", "parent": "doc_1"},
    {"id": "chunk_1", "type": "Mục", "content": "Mục 1: Phạm vi điều chỉnh", "parent": "chunk_0"},
    {"id": "chunk_2", "type": "Điều", "content": "Điều 1: Phạm vi", "parent": "chunk_1"},
    {"id": "chunk_3", "type": "Đoạn văn", "content": "Luật này quy định về hoạt động ngân hàng.", "parent": "chunk_2"}
]

# BƯỚC 4: NẠP DỮ LIỆU VÀO NEO4J
def load_data_to_neo4j(driver):
    with driver.session(database="neo4j") as session: 
        session.run("""
        MERGE (d:Document {id: 'doc_1'})
        SET d.title = 'Luật Ngân hàng mẫu'
        """)
        print("- Đã tạo nút (:Document)")

        for chunk in chunks:
            session.run("""
            MERGE (c:Chunk {id: $id})
            SET c.type = $type, c.content = $content
            
            WITH c
            MATCH (p {id: $parent_id})
            MERGE (p)-[:PARENT_OF]->(c)
            MERGE (c)-[:PART_OF]->(p)
            """, id=chunk['id'], type=chunk['type'], content=chunk['content'], parent_id=chunk['parent'])
        print("- Đã nạp các nút (:Chunk) và quan hệ [:PARENT_OF], [:PART_OF]")

        for i in range(len(chunks) - 1):
            session.run("""
            MATCH (c1:Chunk {id: $id1}), (c2:Chunk {id: $id2})
            MERGE (c1)-[:NEXT]->(c2)
            """, id1=chunks[i]['id'], id2=chunks[i+1]['id'])
        print("- Đã nạp các quan hệ [:NEXT]")

# Thực thi
try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    print("Kết nối Neo4j thành công!")
    load_data_to_neo4j(driver)
    driver.close()
    print("Hoàn thành Bước 3 và Bước 4!")
except Exception as e:
    print(f"Lỗi kết nối hoặc Cypher: {e}")
    