import sys
from pathlib import Path

rag_path = r"c:\agribank-rag\RAG\rag_foundation\buoi_07\rag.py"
with open(rag_path, "r", encoding="utf-8") as f:
    code = f.read()

# Modifying load_config to accept override
code = code.replace("def load_config():", "def load_config(_override=None):")
code = code.replace("config = dotenv_values(env_path)", "config = dotenv_values(env_path)\n    if _override:\n        config.update(_override)")

# Modifying CHROMA_STORAGE_DIR references
code = code.replace("CHROMA_STORAGE_DIR.exists()", "(storage_dir or CHROMA_STORAGE_DIR).exists()")
code = code.replace("path=str(CHROMA_STORAGE_DIR)", "path=str(storage_dir or CHROMA_STORAGE_DIR)")
code = code.replace("CHROMA_STORAGE_DIR.mkdir", "(storage_dir or CHROMA_STORAGE_DIR).mkdir")

# Modifying run_status
code = code.replace("def run_status(strategy):", "def run_status(strategy, storage_dir=None, _config=None):")
code = code.replace("config = load_config()", "config = load_config(_override=_config) if _config is not None else load_config()")

# Modifying run_index
code = code.replace("def run_index(input_dir, strategy, reset=False):", "def run_index(input_dir, strategy, reset=False, storage_dir=None, _config=None, _emb_fn=None):")

# Inside run_index, replace generate_embeddings call
emb_call_old = "embeddings = generate_embeddings(chunks, config[\"api_key\"], config[\"emb_model\"], config[\"emb_dim\"])"
emb_call_new = "embeddings = _emb_fn(chunks, config[\"api_key\"], config[\"emb_model\"], config[\"emb_dim\"]) if _emb_fn else generate_embeddings(chunks, config[\"api_key\"], config[\"emb_model\"], config[\"emb_dim\"])"
code = code.replace(emb_call_old, emb_call_new)

# Modifying query_rag
code = code.replace("def query_rag(question, top_k, strategy):", "def query_rag(question, top_k, strategy, storage_dir=None, _config=None, _q_emb_fn=None, _gen_fn=None):")

# Inside query_rag, replace q_emb generation
q_emb_old = "q_emb = generate_query_embedding(question, config[\"api_key\"], config[\"emb_model\"], config[\"emb_dim\"])"
q_emb_new = "q_emb = _q_emb_fn(question, config[\"api_key\"], config[\"emb_model\"], config[\"emb_dim\"]) if _q_emb_fn else generate_query_embedding(question, config[\"api_key\"], config[\"emb_model\"], config[\"emb_dim\"])"
code = code.replace(q_emb_old, q_emb_new)

# Inside query_rag, replace genai client call
gen_old = """        from google import genai
        client = genai.Client(api_key=config["api_key"])
        resp = client.models.generate_content(
            model=config["gen_model"],
            contents=prompt
        )
        answer_text = resp.text.strip() if resp.text else \"\""""
gen_new = """        if _gen_fn:
            answer_text = _gen_fn(prompt)
        else:
            from google import genai
            client = genai.Client(api_key=config["api_key"])
            resp = client.models.generate_content(
                model=config["gen_model"],
                contents=prompt
            )
            answer_text = resp.text.strip() if resp.text else \"\""""
code = code.replace(gen_old, gen_new)

with open(rag_path, "w", encoding="utf-8") as f:
    f.write(code)
print("rag.py patched for testing.")
