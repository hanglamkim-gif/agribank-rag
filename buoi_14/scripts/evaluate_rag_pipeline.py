# === THỦ THUẬT ĐÁNH LỪA PYTHON VƯỢT LỖI VERTEX AI ===
import sys
try:
    import langchain_google_vertexai
    sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai
except ImportError:
    pass

# === CODE CHÍNH CHUẨN RAGAS ===
import os
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import Faithfulness
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai import HarmCategory, HarmBlockThreshold
from ragas.run_config import RunConfig

# ---------------------------------------------------------
# CẤU HÌNH API KEY VÀ MODEL
# ---------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = "API_KEY_CUA_TOI"

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# ĐÃ SỬA: Chuyển sang gemini-3.5-flash để lấy 20 lượt gọi mới tinh
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", 
    temperature=None,
    safety_settings=safety_settings,
    timeout=60,
    max_retries=3
)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

eval_llm = LangchainLLMWrapper(gemini_llm)
eval_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

# Đường dẫn lưu file
EVAL_DIR = "buoi_14/data/eval"
OUTPUT_DIR = "buoi_14/outputs"
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
QA_DATASET_PATH = os.path.join(EVAL_DIR, "qa_dataset.csv")
EVAL_RESULTS_PATH = os.path.join(EVAL_DIR, "evaluation_results.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "ragas_evaluation_report.md")

# ---------------------------------------------------------
# HÀM BỔ TRỢ
# ---------------------------------------------------------
def load_documents_for_generation():
    return [] 

def query_your_rag_pipeline(question: str):
    return "Lãi suất ngân hàng tùy thuộc vào từng kỳ hạn và thời điểm.", ["Ngân hàng có nhiều mức lãi suất khác nhau tùy thuộc vào kỳ hạn 1 tháng, 3 tháng, 6 tháng hay 12 tháng."]

# ---------------------------------------------------------
# CÁC BƯỚC THỰC THI CHÍNH
# ---------------------------------------------------------
def generate_golden_dataset():
    print("1. Đang khởi tạo Golden Dataset...")
    docs = load_documents_for_generation()
    if not docs:
        print("⚠️ Sử dụng Dataset mẫu (1 câu hỏi) để tránh vượt quá giới hạn API...")
        df = pd.DataFrame({
            "question": ["Lãi suất hiện tại là bao nhiêu?"],
            "ground_truth": ["Lãi suất tùy kỳ hạn."]
        })
        df.to_csv(QA_DATASET_PATH, index=False, encoding='utf-8')
        return df
    try:
        generator = TestsetGenerator(llm=eval_llm, embedding_model=eval_embeddings)
        testset = generator.generate_with_langchain_docs(docs, test_size=1)
        df = testset.to_pandas()
        df.to_csv(QA_DATASET_PATH, index=False, encoding='utf-8')
        return df
    except Exception as e:
        print(f"Lỗi khởi tạo dataset: {e}")
        raise

def run_rag_pipeline_for_eval(qa_df):
    print("2. Đang thực thi RAG Pipeline...")
    answers, contexts = [], []
    for _, row in qa_df.iterrows():
        ans, ctx = query_your_rag_pipeline(row["question"])
        answers.append(ans)
        contexts.append(ctx)
    qa_df["answer"] = answers
    qa_df["contexts"] = contexts
    return qa_df

def evaluate_with_ragas(qa_df):
    print("3. Đang chấm điểm bằng Ragas...")
    eval_dataset = Dataset.from_pandas(qa_df)
    
    custom_run_config = RunConfig(timeout=60, max_workers=1, max_retries=3)
    
    result = evaluate(
        dataset=eval_dataset,
        metrics=[Faithfulness()],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=custom_run_config
    )
    result.to_pandas().to_csv(EVAL_RESULTS_PATH, index=False)
    return result

def generate_markdown_report(result):
    print("4. Đang xuất báo cáo...")
    df = result.to_pandas()
    metrics = df.select_dtypes(include='number').mean()
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Báo cáo Đánh giá RAG Pipeline\n\n")
        for m, s in metrics.items():
            f.write(f"- **{m}**: {s:.4f}\n")
    print("✅ Đã hoàn tất! Kiểm tra file trong thư mục outputs.")
    print(metrics)

if __name__ == "__main__":
    try:
        qa_df = generate_golden_dataset()
        qa_df = run_rag_pipeline_for_eval(qa_df)
        eval_result = evaluate_with_ragas(qa_df)
        generate_markdown_report(eval_result)
    except Exception as e:
        import traceback
        traceback.print_exc()