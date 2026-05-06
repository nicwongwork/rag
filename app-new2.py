import os
import shutil
import tempfile
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv

# 載入環境變數
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
    """初始化或讀取 Vectorstore[cite: 1]"""
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())

@st.cache_resource
def get_llm(api_key: str):
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.3)[cite: 1]

def main():
    st.set_page_config(page_title="Source from PDF", layout="wide")[cite: 1]

    # 初始化 Session State
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = set()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Sidebar
    with st.sidebar:
        st.title("📚 管理中心")

        # 修正：徹底清空記憶
        if st.button("🔥 徹底清空所有書籍記憶"):
            if os.path.exists(CHROMA_DIR):
                # 1. 刪除實體檔案
                shutil.rmtree(CHROMA_DIR)
            # 2. 清空 Session 狀態
            st.session_state["processed_files"] = set()
            st.session_state["messages"] = []
            # 3. 強制清除 Streamlit 的資源緩存，令 Vectorstore 重新初始化
            st.cache_resource.clear()
            st.success("數據庫已完全抹除！")
            st.rerun()

        st.divider()

        # 上傳檔案
        uploaded_file = st.file_uploader("上傳 PDF", type=["pdf"])[cite: 1]

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

                    # 修正：強制將 Metadata 的 source 改回原始檔名[cite: 1]
                    for doc in documents:
                        doc.metadata["source"] = uploaded_file.name

                    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)[cite: 1]
                    vs.add_documents(splitter.split_documents(documents))

                    st.session_state["processed_files"].add(uploaded_file.name)
                    os.remove(tmp_path)
                    st.rerun()

        # 顯示書籍列表與選擇
        st.divider()
        # 修正：直接從 DB 獲取最新的 Metadata
        all_data = vs.get()
        existing_files = sorted(list(set([m["source"] for m in all_data["metadatas"] if "source" in m])))

        selected_file = st.selectbox(
            "🎯 切換閱讀中的書籍：",
            ["全部書籍"] + existing_files,
            index=0
        )

    # 主 UI
    st.title("🎓 Study Assistant")

    # API Key 與 LLM 初始化[cite: 1]
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY Missing")
        return
    llm = get_llm(api_key)

    # 聊天邏輯[cite: 1]
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("問下書入面嘅嘢..."):[cite: 1]
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # 檢索過濾器
            search_kwargs = {"k": 4}
            if selected_file != "全部書籍":
                search_kwargs["filter"] = {"source": selected_file}

            docs = vs.similarity_search(user_input, **search_kwargs)[cite: 1]

            if not docs:
                ans = "喺選定嘅書籍入面搵唔到相關資料。"
            else:
                context = "\n\n".join([d.page_content for d in docs])
                prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer in Traditional Chinese:"
                ans = llm.invoke(prompt).content[cite: 1]

            st.markdown(ans)
            st.session_state["messages"].append({"role": "assistant", "content": ans})

if __name__ == "__main__":
    main()