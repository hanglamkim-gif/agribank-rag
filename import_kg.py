import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

def main():
    load_dotenv()
    
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    
    if not uri or not user or not password:
        print("FAIL: Missing Neo4j credentials in .env")
        return
        
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # Check connection
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"FAIL: Cannot connect to Neo4j. {e}")
        return
    
    try:
        with driver.session() as session:
            # 3. Create uniqueness constraints
            constraints = [
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT document_sokh IF NOT EXISTS FOR (d:Document) REQUIRE d.so_ky_hieu IS UNIQUE",
                "CREATE CONSTRAINT coquan_name IF NOT EXISTS FOR (c:CoQuan) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT nguoiky_name IF NOT EXISTS FOR (n:NguoiKy) REQUIRE n.name IS UNIQUE",
                "CREATE CONSTRAINT doituongapdung_name IF NOT EXISTS FOR (d:DoiTuongApDung) REQUIRE d.name IS UNIQUE",
                "CREATE CONSTRAINT linhvuc_name IF NOT EXISTS FOR (l:LinhVuc) REQUIRE l.name IS UNIQUE"
            ]
            for query in constraints:
                session.run(query)
                
            # Import Documents
            print("Importing Documents...")
            docs_df = pd.read_csv('ner_kb/cleaned_documents.csv').fillna("")
            docs_imported = 0
            for _, row in docs_df.iterrows():
                query = """
                MERGE (d:Document {so_ky_hieu: $so_ky_hieu})
                ON CREATE SET 
                    d.id = $id,
                    d.loai_van_ban = $loai_van_ban,
                    d.ngay_ban_hanh = $ngay_ban_hanh,
                    d.content_clean = $content_clean
                """
                session.run(query, 
                    so_ky_hieu=str(row['so_ky_hieu']),
                    id=str(row['id']),
                    loai_van_ban=str(row['loai_van_ban']),
                    ngay_ban_hanh=str(row['ngay_ban_hanh']),
                    content_clean=str(row['content_clean'])
                )
                docs_imported += 1
                
                # Relations from document
                if row['co_quan_ban_hanh']:
                    session.run("""
                    MERGE (c:CoQuan {name: $coquan})
                    WITH c
                    MATCH (d:Document {so_ky_hieu: $sokh})
                    MERGE (d)-[:BAN_HANH_BOI]->(c)
                    """, coquan=str(row['co_quan_ban_hanh']).strip(), sokh=str(row['so_ky_hieu']))
                if row['nguoi_ky']:
                    session.run("""
                    MERGE (n:NguoiKy {name: $nguoiky})
                    WITH n
                    MATCH (d:Document {so_ky_hieu: $sokh})
                    MERGE (d)-[:KY_BOI]->(n)
                    """, nguoiky=str(row['nguoi_ky']).strip(), sokh=str(row['so_ky_hieu']))
                if row['linh_vuc']:
                    session.run("""
                    MERGE (l:LinhVuc {name: $linhvuc})
                    WITH l
                    MATCH (d:Document {so_ky_hieu: $sokh})
                    MERGE (d)-[:THUOC_LINH_VUC]->(l)
                    """, linhvuc=str(row['linh_vuc']).strip(), sokh=str(row['so_ky_hieu']))
                    
            # Import Entities
            print("Importing Entities...")
            ent_df = pd.read_csv('ner_kb/entities.csv').fillna("")
            ent_imported = 0
            for _, row in ent_df.iterrows():
                canonical_name = str(row['canonical_name']).strip()
                entity_type = str(row['entity_type']).strip()
                if not canonical_name or not entity_type: continue
                
                if entity_type == 'CoQuan':
                    session.run("MERGE (e:CoQuan {name: $name})", name=canonical_name)
                    ent_imported += 1
                elif entity_type == 'NguoiKy':
                    session.run("MERGE (e:NguoiKy {name: $name})", name=canonical_name)
                    ent_imported += 1
                elif entity_type == 'DoiTuongApDung':
                    session.run("MERGE (e:DoiTuongApDung {name: $name})", name=canonical_name)
                    ent_imported += 1
                elif entity_type == 'LinhVuc':
                    session.run("MERGE (e:LinhVuc {name: $name})", name=canonical_name)
                    ent_imported += 1
                elif entity_type == 'Document':
                    session.run("MERGE (e:Document {so_ky_hieu: $name})", name=canonical_name)
                    ent_imported += 1

            # Import Relationships
            print("Importing Relationships...")
            rel_df = pd.read_csv('ner_kb/relationships.csv').fillna("")
            rel_imported = 0
            errors = 0
            
            for _, row in rel_df.iterrows():
                source = str(row['source']).strip()
                target = str(row['target']).strip()
                rel_type = str(row['relationship_type']).strip()
                if not source or not target or not rel_type: continue
                
                # We need to find source and target nodes.
                # Since we don't know their exact labels in advance for relations, we can match across possible labels.
                # Usually source and targets are Document or Entities, identified by `so_ky_hieu` or `name`.
                
                query = f"""
                MATCH (s) WHERE (s.so_ky_hieu = $source OR s.name = $source)
                MATCH (t) WHERE (t.so_ky_hieu = $target OR t.name = $target)
                MERGE (s)-[r:{rel_type}]->(t)
                RETURN count(r) as c
                """
                res = session.run(query, source=source, target=target)
                record = res.single()
                
                if record and record["c"] > 0:
                    rel_imported += 1
                else:
                    print(f"Error importing relationship: {source} -[{rel_type}]-> {target}. Source or target not found.")
                    errors += 1

            # Get stats
            print("\n--- Import Statistics ---")
            labels_res = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as c")
            for record in labels_res:
                print(f"Node {record['label']}: {record['c']}")
                
            rels_res = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as c")
            for record in rels_res:
                print(f"Relationship {record['type']}: {record['c']}")
                
            print(f"Import Errors (missing nodes): {errors}")

    except Exception as e:
        print(f"FAIL: Error during import. {e}")
        return
    finally:
        driver.close()
        
    print("\nPASS")

if __name__ == "__main__":
    main()
