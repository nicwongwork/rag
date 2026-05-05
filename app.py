import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LlamaIndex 核心
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.deepseek import DeepSeek
#from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.llms.ollama import Ollama

# LlamaCloud / LlamaParse
from llama_cloud import AsyncLlamaCloud

# ====== 配置設定 ======
# 建議將 API Key 放在環境變數，這裡暫時根據你的要求寫死
LLAMA_CLOUD_API_KEY = "llx-WMVJZj75Fk3adn4oytyzq2eeiOhOjOe1eo9XLL9sQ0CTdm0L"
os.environ["DEEPSEEK_API_KEY"] = 'sk-3fde1399fa244f1c8e0db5b3c34df846'
DATA_DIR = Path("./data/pdfs")
INDEX_DIR = Path("./storage")

# 初始化全局設定
#Settings.embed_model = DeepSeek(model="deepseek-chat")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.llm = DeepSeek(
    model="deepseek-chat",
    temperature=0.1,
    timeout=600.0,
)
# 增大 Chunk 以確保題目完整性
Settings.chunk_size = 1024
Settings.chunk_overlap = 100

# 初始化 LlamaCloud Client
client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)

# ====== 全局狀態 ======
rag_index: Optional[VectorStoreIndex] = None
chat_sessions: Dict[str, object] = {}

# ====== Pydantic 模型 ======
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[str]

# ====== 核心功能：Agentic 解析與索引構建 ======

async def parse_pdf_agentic(file_path: Path) -> str:
    """調用 LlamaCloud Agentic Tier 進行高品質 Markdown 解析"""
    print(f"[Parser] 正在將 {file_path.name} 送往 Agentic Tier 解析...")

    with open(file_path, "rb") as f:
        # 1. 上傳檔案
        file_obj = await client.files.create(file=f, purpose="parse")

    # 2. 觸發 Agentic 解析 (專門處理複雜表格與佈局)
    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        version="2026-04-09",
        expand=["markdown_full"],
    )

    return result.markdown_full

async def build_index_async() -> VectorStoreIndex:
    """非同步構建索引"""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        raise RuntimeError(f"請將 PDF 放入 {DATA_DIR} 資料夾後重新啟動")

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"在 {DATA_DIR} 找不到任何 PDF 檔案")

    all_docs = []
    for pdf_path in pdf_files:
        try:
            # 獲取 Markdown 內容
            markdown_text = await parse_pdf_agentic(pdf_path)
            print(markdown_text[:1000]) # 印前 1000 個字
            # 手動封裝成 LlamaIndex Document
            doc = Document(
                text=markdown_text,
                metadata={
                    "file_name": pdf_path.name,
                    "parsed_by": "LlamaCloud-Agentic"
                }
            )
            all_docs.append(doc)
        except Exception as e:
            print(f"[Error] 解析 {pdf_path.name} 失敗: {e}")

    print(f"[RAG] 成功解析 {len(all_docs)} 份文件，正在建立向量索引...")

    # 建立並持久化
    index = VectorStoreIndex.from_documents(all_docs, show_progress=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))

    print("[RAG] 索引構建完成並已儲存至磁碟")
    return index

def load_index() -> VectorStoreIndex:
    """載入現有索引"""
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    index = load_index_from_storage(storage_context)
    print("[RAG] 已從儲存空間載入現有索引")
    return index

# ====== FastAPI Lifespan ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_index

    # 啟動時自動檢查是否需要建 Index
    try:
        if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
            rag_index = load_index()
        else:
            rag_index = await build_index_async()
    except Exception as e:
        print(f"[RAG] 初始化失敗: {e}")
        rag_index = None

    yield
    print("[RAG] 服務正在關閉")

app = FastAPI(
    title="Advanced PDF RAG (Agentic)",
    lifespan=lifespan
)

# ====== Chat Engine 管理 ======
def get_chat_engine(session_id: str):
    if rag_index is None:
        raise RuntimeError("Index 尚未初始化")

    if session_id not in chat_sessions:
        # 使用 condense_plus_context 模式，適合處理多輪對話與複雜文檔
        memory = ChatMemoryBuffer.from_defaults(token_limit=4000, llm=Settings.llm)
        chat_sessions[session_id] = rag_index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            similarity_top_k=3,
            llm=Settings.llm,
            system_prompt=(
                "你是一個精確的試卷分析助教。用戶會詢問特定題目的內容。"
                "【操作指南】"
                "1. 必須嚴格掃描 Markdown 內容中的標題（如 #, ##, ###）和編號清單。"
                "2. 試卷通常分為『甲部』、『乙部』、『丙部』、『丁部』。當用戶問『甲部第1條』，請先定位到『甲部』標題，再尋找其下方標號為『1.』或『1』的內容。"
                "3. 請完整複述題目原文，不要進行改寫或總結。"
                "4. 如果文檔中有多個第1條，請根據上下文判斷哪一個屬於『甲部』。"
                "5. 若找不到，請回答：『在文檔中找不到甲部第1條。』不要嘗試編造。"
                "請務必使用繁體中文回答。"
            )
        )
    return chat_sessions[session_id]

# ====== API 端點 ======

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_ready": rag_index is not None,
        "active_sessions": len(chat_sessions)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if rag_index is None:
        raise HTTPException(status_code=503, detail="Index 尚未就緒")

    session_id = req.session_id or str(uuid.uuid4())
    chat_engine = get_chat_engine(session_id)

    try:
        # 對於試卷類問題，可以在這裡對 query 做簡單增強
        # 例如若 user 只輸入數字，可以補全為「請問第 X 題內容是什麼？」
        resp = chat_engine.chat(req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"對話出錯: {e}")

    sources = list(set([
        node.metadata.get("file_name", "未知來源")
        for node in resp.source_nodes
    ]))

    return ChatResponse(
        session_id=session_id,
        answer=str(resp),
        sources=sources,
    )

@app.delete("/chat/{session_id}")
async def delete_session(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": "Session 清除成功"}
    raise HTTPException(status_code=404, detail="Session 不存在")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)