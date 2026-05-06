import os
import shutil
import tempfile
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv

# 1. 載入環境變數
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})

def load_vectorstore():
    """初始化或讀取 Vectorstore"""
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())

@st.cache_resource
def get_llm(api_key: str):
    # 使用 Groq 的 Llama 模型
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.3)

def main():
    st.set_page_config(page_title="Source from PDF", layout="wide")

    # 初始化 Session State
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = set()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Sidebar 介面
    with st.sidebar:
        st.title("📚 管理中心")

        # 功能：徹底清空數據庫
        if st.button("🔥 徹底清空所有書籍記憶"):
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
            st.session_state["processed_files"] = set()
            st.session_state["messages"] = []
            # 強制清除 Streamlit 緩存，令 Vectorstore 重新載入
            st.cache_resource.clear()
            st.success("數據庫已完全抹除！")
            st.rerun()

        st.divider()

        # 檔案上傳
        uploaded_file = st.file_uploader("上傳 PDF 講義", type=["pdf"])

        # 獲取當前 Vectorstore 實例
        vs = load_vectorstore()

        if uploaded_file:
            if uploaded_file.name not in st.session_state["processed_files"]:
                with st.spinner("正在建立索引..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()

                    # 關鍵修正：將 metadata 中的 source 由臨時路徑改為原始檔名
                    for doc in documents:
                        doc.metadata["source"] = uploaded_file.name

                    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                    vs.add_documents(splitter.split_documents(documents))

                    st.session_state["processed_files"].add(uploaded_file.name)
                    os.remove(tmp_path)
                    st.rerun()

        # 書籍列表與過濾
        st.divider()
        all_data = vs.get()
        # 從 DB 獲取唯一的來源名稱
        existing_files = sorted(list(set([m["source"] for m in all_data["metadatas"] if "source" in m])))

        selected_file = st.selectbox(
            "🎯 切換閱讀中的書籍：",
            ["全部書籍"] + existing_files,
            index=0
        )

        if existing_files:
            st.caption(f"目前數據庫內共有 {len(existing_files)} 本書")

    # 主聊天介面
    st.title("🎓 Study Assistant")

    # API Key 與 LLM 初始化
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("找不到 GROQ_API_KEY，請檢查環境變數設定。")
        return

    try:
        llm = get_llm(api_key)
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        return

    # 顯示歷史對話
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 對話輸入
    if user_input := st.chat_input("問下書入面嘅內容..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                # 設定檢索參數與過濾器
                search_kwargs = {"k": 4}
                if selected_file != "全部書籍":
                    search_kwargs["filter"] = {"source": selected_file}

                # 進行向量搜尋
                docs = vs.similarity_search(user_input, **search_kwargs)

                if not docs:
                    answer = "喺選定嘅書籍入面搵唔到相關資料，或者數據庫係空嘅。"
                else:
                    context = "\n\n".join([d.page_content for d in docs])
                    prompt = f"Context:\n{context}\n\nQuestion: {user_input}\n\nPlease answer in Traditional Chinese (Hong Kong)."

                    response = llm.invoke(prompt)
                    answer = response.content

                st.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"發生錯誤: {e}")

if __name__ == "__main__":
    main()