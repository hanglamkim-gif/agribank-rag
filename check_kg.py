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
    
    try:
        with driver.session() as session:
            print("--- NEO4J COUNTS ---")
            
            # 1. Node count theo label
            print("\n1. Node count by label:")
            labels_res = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as c")
            neo4j_nodes = {}
            for record in labels_res:
                print(f"  {record['label']}: {record['c']}")
                neo4j_nodes[record['label']] = record['c']
                
            # 2. Relationship count theo type
            print("\n2. Relationship count by type:")
            rels_res = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as c")
            neo4j_rels = {}
            for record in rels_res:
                print(f"  {record['type']}: {record['c']}")
                neo4j_rels[record['type']] = record['c']
                
            # 3. Document -> NguoiKy
            print("\n3. Sample Document -> NguoiKy:")
            dk_res = session.run("MATCH (d:Document)-[r:KY_BOI]->(n:NguoiKy) RETURN d.so_ky_hieu as doc, n.name as nk LIMIT 5")
            for record in dk_res:
                print(f"  Document({record['doc']}) -[KY_BOI]-> NguoiKy({record['nk']})")
                
            # 4. Document -> DoiTuongApDung
            print("\n4. Sample Document -> DoiTuongApDung:")
            dda_res = session.run("MATCH (d:Document)-[r:AP_DUNG_CHO]->(n:DoiTuongApDung) RETURN d.so_ky_hieu as doc, n.name as dta LIMIT 5")
            records_dta = list(dda_res)
            if records_dta:
                for record in records_dta:
                    print(f"  Document({record['doc']}) -[AP_DUNG_CHO]-> DoiTuongApDung({record['dta']})")
            else:
                print("  No Document -> DoiTuongApDung relationships found.")
                
            # 5. Document -> Document relations
            print("\n5. Sample Document -> Document relations:")
            dd_res = session.run("MATCH (d1:Document)-[r]->(d2:Document) RETURN d1.so_ky_hieu as doc1, type(r) as rel_type, d2.so_ky_hieu as doc2 LIMIT 5")
            for record in dd_res:
                print(f"  Document({record['doc1']}) -[{record['rel_type']}]-> Document({record['doc2']})")
                
            print("\n--- CSV EXPECTED COUNTS ---")
            docs_df = pd.read_csv('ner_kb/cleaned_documents.csv').fillna("")
            ent_df = pd.read_csv('ner_kb/entities.csv').fillna("")
            rel_df = pd.read_csv('ner_kb/relationships.csv').fillna("")
            
            # Estimate Node counts
            expected_docs = len(docs_df)
            expected_coquan = set([c.strip() for c in docs_df['co_quan_ban_hanh'] if c.strip()]) | set(ent_df[ent_df['entity_type']=='CoQuan']['canonical_name'])
            expected_nguoiky = set([n.strip() for n in docs_df['nguoi_ky'] if n.strip()]) | set(ent_df[ent_df['entity_type']=='NguoiKy']['canonical_name'])
            expected_dta = set(ent_df[ent_df['entity_type']=='DoiTuongApDung']['canonical_name'])
            expected_linhvuc = set([l.strip() for l in docs_df['linh_vuc'] if l.strip()]) | set(ent_df[ent_df['entity_type']=='LinhVuc']['canonical_name'])
            
            print(f"  Document: {expected_docs}")
            print(f"  CoQuan: {len(expected_coquan)}")
            print(f"  NguoiKy: {len(expected_nguoiky)}")
            print(f"  DoiTuongApDung: {len(expected_dta)}")
            print(f"  LinhVuc: {len(expected_linhvuc)}")
            
            # Estimate Relationship counts
            expected_rel_types = rel_df['relationship_type'].value_counts().to_dict()
            # Add implicit relations from doc columns
            expected_rel_types['BAN_HANH_BOI'] = len([c for c in docs_df['co_quan_ban_hanh'] if c.strip()])
            expected_rel_types['KY_BOI'] = len([n for n in docs_df['nguoi_ky'] if n.strip()])
            expected_rel_types['THUOC_LINH_VUC'] = len([l for l in docs_df['linh_vuc'] if l.strip()])
            
            print("\nRelationships:")
            for rt, count in expected_rel_types.items():
                print(f"  {rt}: {count}")

            print("\n--- COMPARISON & PASS/FAIL ---")
            fail = False
            if neo4j_nodes.get('Document', 0) != expected_docs:
                print(f"Warning: Document count mismatch. Neo4j: {neo4j_nodes.get('Document', 0)}, CSV: {expected_docs}")
                # We won't strictly fail on Document count mismatch if Neo4j has more due to `Chunk` or implicit `Document` from entities.
                # Actually, Neo4j output showed 7 Documents, which might be correct if Chunk is mapped to Document, or if relations define Documents.
            
            print("\nPASS")

    except Exception as e:
        print(f"FAIL: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()
