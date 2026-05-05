import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# 引入配置物件 (確保你本地有 setting.py)
from setting import Settings

# LlamaIndex 核心
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_cloud import AsyncLlamaCloud

# ====== 1. 配置設定 ======
# 這裡建議 API Key 還是放在環境變數，若要寫死請確保安全
LLAMA_CLOUD_API_KEY = "llx-WMVJZj75Fk3adn4oytyzq2eeiOhOjOe1eo9XLL9sQ0CTdm0L"
DATA_DIR = Path("./data/pdfs")
INDEX_DIR = Path("./storage")

# 初始化 LlamaCloud Client
client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)

# ====== 2. 全局狀態 ======
rag_index: Optional[VectorStoreIndex] = None
chat_sessions: Dict[str, object] = {}

# ====== 3. Pydantic 模型 ======
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[str]

# ====== 4. 核心功能：Agentic 解析與索引構建 ======

async def parse_pdf_agentic(file_path: Path) -> str:
    """調用 LlamaCloud Agentic Tier 進行高品質 Markdown 解析"""
    print(f"[Parser] 正在將 {file_path.name} 送往 Agentic Tier 解析...")

    try:
        with open(file_path, "rb") as f:
            # 1. 上傳檔案
            file_obj = await client.files.create(file=f, purpose="parse")

        # 2. 觸發 Agentic 解析
        result = await client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",
            version="2026-04-09",
            expand=["markdown_full"],
        )

        # 轉為字串避免物件導向的驗證錯誤
        return str(result.markdown_full)
    except Exception as e:
        print(f"[Parser Error] {e}")
        raise e

async def build_index_async() -> VectorStoreIndex:
    """非同步構建索引"""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return None

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[Warn] {DATA_DIR} 內沒有 PDF")
        return None

    all_docs = []
    for pdf_path in pdf_files:
        try:
            markdown_text = await parse_pdf_agentic(pdf_path)
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

    if not all_docs:
        return None

    print(f"[RAG] 成功解析 {len(all_docs)} 份文件，正在建立向量索引...")
    index = VectorStoreIndex.from_documents(all_docs, show_progress=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    return index

# ====== 5. FastAPI 生命週期 ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_index
    try:
        if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
            print("[RAG] 正在從本地儲存載入索引...")
            storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
            rag_index = load_index_from_storage(storage_context)
        else:
            rag_index = await build_index_async()
    except Exception as e:
        print(f"[RAG] 初始化失敗: {e}")
        rag_index = None
    yield

app = FastAPI(title="DeepSeek RAG System", lifespan=lifespan)

# ====== 6. Chat Engine 管理 ======
def get_chat_engine(session_id: str):
    if rag_index is None:
        raise RuntimeError("Index 未就緒")

    if session_id not in chat_sessions:
        # 必須明確傳入 llm=Settings.llm 避免尋找預設的 OpenAI
        memory = ChatMemoryBuffer.from_defaults(token_limit=4000, llm=Settings.llm)

        chat_sessions[session_id] = rag_index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            llm=Settings.llm,
            system_prompt=(
                "你是一個精確的試卷分析助教。用戶會詢問特定題目的內容。\n"
                "【操作指南】\n"
                "1. 必須嚴格掃描 Markdown 中的標題（#）和編號。\n"
                "2. 先定位學科部分（如甲部、乙部），再尋找下方對應題目。\n"
                "3. 完整複述題目原文，不要改寫。\n"
                "4. 找不到請直說，不要編造。\n"
                "請務必使用繁體中文回答。"
            )
        )
    return chat_sessions[session_id]

# ====== 7. API 端點 ======

@app.get("/health")
async def health():
    return {"status": "ok", "index_ready": rag_index is not None}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest = Body(...)):
    if rag_index is None:
        raise HTTPException(status_code=503, detail="Index 尚未就緒")

    session_id = req.session_id or str(uuid.uuid4())

    try:
        chat_engine = get_chat_engine(session_id)
        resp = chat_engine.chat(req.message)

        sources = list(set([
            node.metadata.get("file_name", "未知來源")
            for node in resp.source_nodes
        ]))

        return ChatResponse(
            session_id=session_id,
            answer=str(resp),
            sources=sources,
        )
    except Exception as e:
        print(f"[Chat Error] {e}")
        raise HTTPException(status_code=500, detail=f"對話出錯: {str(e)}")

@app.delete("/chat/{session_id}")
async def delete_session(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": "Success"}
    raise HTTPException(status_code=404, detail="Not Found")

if __name__ == "__main__":
    import uvicorn
    # 啟動服務
    uvicorn.run(app, host="0.0.0.0", port=8000)