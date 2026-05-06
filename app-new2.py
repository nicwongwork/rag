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
APP_TITLE = "Source from PDF"
APP_SUBTITLE = "SOURCE TO YOUR STUDIES"

PREMIUM_STYLE = """
<style>
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #1a1c24 100%); color: white; }
    .stChatMessage { background-color: #333333; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; }
</style>
"""

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})

def load_vectorstore() -> Chroma:
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())

@st.cache_resource
def get_llm(api_key: str):
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.3)

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
    st.markdown(PREMIUM_STYLE, unsafe_allow_html=True)

    # 初始化 session state
    if "messages" not in st.session_state: st.session_state["messages"] = []
    if "processed_files" not in st.session_state: st.session_state["processed_files"] = set()

    # Sidebar
    with st.sidebar:
        st.title(f"🔍 {APP_TITLE}")

        # 功能 1：清空對話
        if st.button("💬 Reset Conversation"):
            st.session_state["messages"] = []
            st.rerun()

        # 功能 2：徹底清空數據庫
        if st.button("🔥 Clear All Memory (DB)"):
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR) # 刪除整個資料夾
            st.session_state["processed_files"] = set()
            st.cache_resource.clear() # 清除 Embedding 緩存
            st.success("Database wiped clean!")
            st.rerun()

        st.divider()

        # Knowledge Base
        st.subheader("📚 Knowledge Base")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        vectorstore = load_vectorstore()

        # 處理文件上傳
        if uploaded_file:
            if uploaded_file.name not in st.session_state["processed_files"]:
                with st.spinner("Indexing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    # 強制加入 source 標籤
                    for doc in documents:
                        doc.metadata["source"] = uploaded_file.name

                    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                    vectorstore.add_documents(splitter.split_documents(documents))
                    st.session_state["processed_files"].add(uploaded_file.name)
                    os.remove(tmp_path)
                    st.rerun()

        # 功能 3：選擇當前要使用的 PDF
        st.divider()
        # 從 Vectorstore 獲取所有現有的 source (如果你沒刪除過，這裡會顯示所有歷史 PDF)
        all_metadata = vectorstore.get().get("metadatas", [])
        existing_files = list(set([m["source"] for m in all_metadata if "source" in m]))

        selected_file = st.selectbox(
            "🎯 Select PDF to query:",
            ["All Files"] + existing_files,
            index=0
        )

        if existing_files:
            st.info(f"Currently indexing {len(existing_files)} files.")

    # Main UI
    st.title(f"🎓 {APP_TITLE}")

    # API Key 檢查
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.warning("Please check your GROQ_API_KEY.")
        return
    llm = get_llm(api_key)

    # 顯示對話
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if user_input := st.chat_input("Ask about your studies..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                # 功能 4：根據選擇過濾檢索結果
                search_kwargs = {}
                if selected_file != "All Files":
                    search_kwargs["filter"] = {"source": selected_file}

                docs = vectorstore.similarity_search(user_input, k=4, **search_kwargs)

                if not docs:
                    answer = "I can't find anything about that in the selected material."
                else:
                    context = "\n\n".join([d.page_content for d in docs])
                    prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer in Traditional Chinese:"
                    response = llm.invoke(prompt)
                    answer = response.content

                st.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()