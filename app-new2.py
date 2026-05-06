import os
import shutil
import tempfile
import streamlit as st
import chromadb

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
CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 恢復你原本嘅 Premium UI CSS
PREMIUM_STYLE = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1c24 100%);
    }
    h1 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -1px;
    }
    .stChatMessage {
        background-color: #333333;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #30363d;
        background-color: #21262d;
        color: #c9d1d9;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #30363d;
        border-color: #8b949e;
    }
</style>
"""

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})

def load_vectorstore():
    """初始化 Vectorstore，使用 PersistentClient 避開 Tenant 報錯"""
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR)

    persistent_client = chromadb.PersistentClient(path=CHROMA_DIR)

    return Chroma(
        client=persistent_client,
        embedding_function=get_embeddings()
    )

@st.cache_resource
def get_llm(api_key: str):
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.3)

def main():
    # 設定 Page 並注入 CSS Style
    st.set_page_config(page_title="Source from PDF", page_icon="📚", layout="wide")
    st.markdown(PREMIUM_STYLE, unsafe_allow_html=True)

    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = set()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    with st.sidebar:
        st.title("📚 管理中心")

        if st.button("🔥 徹底清空所有書籍記憶"):
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
            st.session_state["processed_files"] = set()
            st.session_state["messages"] = []
            st.cache_resource.clear()
            st.success("數據庫已抹除！")
            st.rerun()

        st.divider()
        uploaded_file = st.file_uploader("上傳 PDF 講義", type=["pdf"])

        vs = load_vectorstore()

        if uploaded_file:
            if uploaded_file.name not in st.session_state["processed_files"]:
                with st.spinner("正在建立索引..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()

                    # 修正檔名顯示邏輯
                    for doc in documents:
                        doc.metadata["source"] = uploaded_file.name

                    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                    vs.add_documents(splitter.split_documents(documents))

                    st.session_state["processed_files"].add(uploaded_file.name)
                    os.remove(tmp_path)
                    st.rerun()

        st.divider()
        # 獲取現有書籍名單
        all_data = vs.get()
        existing_files = sorted(list(set([m["source"] for m in all_data["metadatas"] if "source" in m])))

        selected_file = st.selectbox(
            "🎯 切換閱讀中的書籍：",
            ["全部書籍"] + existing_files,
            index=0
        )

    st.title("🎓 Study Assistant")
    st.markdown("*SOURCE TO YOUR STUDIES*")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("請在 Secrets 加入 GROQ_API_KEY")
        return

    llm = get_llm(api_key)

    # 聊天氣泡顯示
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("問下書入面嘅內容..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                search_kwargs = {"k": 4}
                if selected_file != "全部書籍":
                    search_kwargs["filter"] = {"source": selected_file}

                docs = vs.similarity_search(user_input, **search_kwargs)

                if not docs:
                    answer = "搵唔到相關資料。"
                else:
                    context = "\n\n".join([d.page_content for d in docs])
                    prompt = f"Context:\n{context}\n\nQuestion: {user_input}\n\nAnswer in Traditional Chinese:"
                    answer = llm.invoke(prompt).content

                st.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"發生錯誤: {e}")

if __name__ == "__main__":
    main()