import pytest
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma

# 模擬一個簡單的 Embedding，用於 CI 環境測試 ChromaDB 結構
# 這樣你就不需要在 CI 跑真實的 Embedding 模型（省錢又省時）
class MockEmbedding:
    def embed_documents(self, texts):
        return [[0.1] * 1536 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 1536

def test_environment_variables():
    """1. 檢查核心環境變數"""
    # 確保 GitHub Secrets 或 Streamlit Secrets 有傳入 Groq Key
    assert "GROQ_API_KEY" in os.environ, "缺少 GROQ_API_KEY 環境變數"

def test_chromadb_initialization():
    """2. 測試 ChromaDB 結構是否能正常建立"""
    try:
        # 使用 In-memory 模式進行測試，不產生實際檔案
        vectorstore = Chroma(
            collection_name="test_collection",
            embedding_function=MockEmbedding(),
            persist_directory=None
        )
        assert vectorstore is not None
    except Exception as e:
        pytest.fail(f"ChromaDB 初始化失敗: {e}")

def test_groq_connectivity():
    """3. 測試 Groq API 連線狀況"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("gsk_test"):
        pytest.skip("無有效的 Groq API Key，跳過連線測試")

    try:
        # 測試最輕量的模型確保通訊正常
        llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
            timeout=10
        )
        response = llm.invoke("Hello")
        assert response.content != ""
        assert isinstance(response.content, str)
    except Exception as e:
        pytest.fail(f"Groq API 呼叫失敗: {e}")

def test_pdf_loader_import():
    """4. 檢查 PDF 解析庫是否安裝正確"""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        # 確保 class 可以被實例化
        loader = PyPDFLoader
        assert loader is not None
    except ImportError:
        pytest.fail("找不到 PyPDFLoader，請檢查 requirements.txt 是否包含 pypdf")

def test_rag_logic_structure():
    """5. 模擬 RAG 檢索邏輯（選填：如果你有抽離 function）"""
    # 這裡可以測試你 app.py 入面處理文件的邏輯是否會噴 Error
    # 例如：檢查 text_splitter 是否正常運作
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text = "這是一段測試文本。" * 100
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = splitter.split_text(text)
    assert len(docs) > 1