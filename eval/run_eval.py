"""
RAG Evaluation Runner
=====================
運行端到端評估：從向量數據庫檢索 + 生成答案 + 評估質素

Usage:
    python eval/run_eval.py --dataset eval/eval_dataset.json --output eval/results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 添加項目根目錄到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from eval.evaluation import RAGEvaluator, format_eval_report


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"


def load_vectorstore():
    """載入 Persistent ChromaDB"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return Chroma(client=client, embedding_function=embeddings)


def generate_answer(question: str, vectorstore, llm) -> tuple:
    """
    模擬 RAG pipeline：檢索 + 生成
    返回 (generated_answer, retrieved_contexts)
    """
    # 1. 檢索
    docs = vectorstore.similarity_search(question, k=4)
    contexts = [d.page_content for d in docs]

    # 2. 生成
    context_text = "\n\n".join(contexts)
    prompt = f"""\
    你是一個專家助手。請根據以下上下文回答問題。
    如果上下文沒有相關資訊，請說「根據提供的資料，我無法回答這個問題」。

    上下文:
    {context_text}

    問題: {question}
    答案:
    """

    response = llm.invoke(prompt)
    return response.content, contexts


def run_evaluation(dataset_path: str, output_path: str):
    """運行完整評估流程"""

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ 錯誤: 請設置 GROQ_API_KEY 環境變數")
        sys.exit(1)

    # 載入數據集
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"📊 載入 {len(dataset)} 條評估問題")

    # 初始化組件
    print("🔧 初始化 LLM 和 Vectorstore...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.3
    )

    # 檢查是否有已索引的文檔
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collections = client.list_collections()

    if not collections:
        print("⚠️ 警告: ChromaDB 中沒有找到任何集合。請先上傳並索引 PDF 文檔。")
        print("   你可以運行 Streamlit app 上傳 PDF，或者使用 eval/sample_docs/ 中的測試文檔。")
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(client=client, embedding_function=embeddings)

    # 生成答案
    print("🤖 正在生成答案...")
    evaluated_dataset = []
    for i, item in enumerate(dataset, 1):
        print(f"   [{i}/{len(dataset)}] {item['question'][:50]}...")
        answer, contexts = generate_answer(item["question"], vectorstore, llm)
        evaluated_dataset.append({
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "retrieved_contexts": contexts,
            "generated_answer": answer
        })

    # 運行評估
    print("📈 正在評估答案質素...")
    evaluator = RAGEvaluator(groq_api_key=api_key)
    results = evaluator.evaluate_batch(evaluated_dataset)

    # 保存結果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 輸出報告
    report = format_eval_report(results)

    # 同時保存 Markdown 報告
    md_path = output_path.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 50)
    print(report)
    print("=" * 50)
    print(f"\n✅ 評估完成！")
    print(f"   JSON 結果: {output_path}")
    print(f"   Markdown 報告: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation Runner")
    parser.add_argument(
        "--dataset",
        default="eval/eval_dataset.json",
        help="評估數據集路徑"
    )
    parser.add_argument(
        "--output",
        default="eval/results.json",
        help="輸出結果路徑"
    )
    args = parser.parse_args()

    run_evaluation(args.dataset, args.output)
