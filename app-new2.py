import os
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

# ====== 配置設定 ======
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Premium UI 樣式
PREMIUM_STYLE = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1c24 100%);
        color: #ffffff;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
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

# ====== 核心功能 ======

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})

@st.cache_resource
def load_vectorstore():
    """使用 EphemeralClient (InMemory)，徹底避開 Streamlit Cloud 嘅硬碟寫入報錯"""
    client = chromadb.EphemeralClient()
    return Chroma(client=client, embedding_function=get_embeddings())

@st.cache_resource
def get_llm(api_key: str):
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.3)

# ====== 主介面 ======

def main():
    st.set_page_config(page_title="Source from PDF", page_icon="📚", layout="wide")
    st.markdown(PREMIUM_STYLE, unsafe_allow_html=True)

    # 初始化 Session State
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = set()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 獲取 InMemory Vectorstore
    vs = load_vectorstore()

    # --- 左側邊欄 ---
    with st.sidebar:
        st.title("📚 管理中心")

        # 記憶體重置按鈕 (保留一鍵清空功能)
        if st.button("🔥 徹底清空所有書籍"):
            st.session_state["processed_files"] = set()
            st.session_state["messages"] = []
            st.cache_resource.clear()
            st.success("所有記憶已重置！")
            st.rerun()

        st.divider()
        st.subheader("📥 上傳書籍")
        uploaded_file = st.file_uploader("上傳 PDF 講義", type=["pdf"], label_visibility="collapsed")

        if uploaded_file:
            if uploaded_file.name not in st.session_state["processed_files"]:
                with st.spinner("正在建立索引 (InMemory)..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()

                    for doc in documents:
                        doc.metadata["source"] = uploaded_file.name

                    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                    vs.add_documents(splitter.split_documents(documents))

                    st.session_state["processed_files"].add(uploaded_file.name)
                    os.remove(tmp_path)
                    st.rerun()

        # 獲取目前記憶體內有嘅書單
        all_data = vs.get()
        existing_files = sorted(list(set([m["source"] for m in all_data["metadatas"] if "source" in m])))

        st.divider()
        st.subheader("🗑️ 刪除單一書籍")
        if existing_files:
            # 讓用戶選擇要刪除的書
            file_to_delete = st.selectbox("選擇要移除嘅講義：", existing_files, key="delete_box")

            if st.button(f"刪除 {file_to_delete}"):
                # 1. 根據 source 名稱搵出所有對應嘅數據 ID
                target_data = vs.get(where={"source": file_to_delete})
                ids_to_delete = target_data.get("ids", [])

                if ids_to_delete:
                    # 2. 叫 ChromaDB 刪除呢堆 ID
                    vs.delete(ids=ids_to_delete)

                # 3. 喺 Session State 入面除名
                if file_to_delete in st.session_state["processed_files"]:
                    st.session_state["processed_files"].remove(file_to_delete)

                st.success(f"已成功刪除 {file_to_delete}！")
                st.rerun()
        else:
            st.info("目前數據庫是空的")

        st.divider()
        st.subheader("🎯 閱讀設定")
        selected_file = st.selectbox(
            "切換提問範圍：",
            ["全部書籍"] + existing_files,
            index=0
        )
        if existing_files:
            st.caption(f"目前共收錄 {len(existing_files)} 本書")

    # --- 主聊天區域 ---
    st.title("🎓 Study Assistant")
    st.markdown("*SOURCE TO YOUR STUDIES*")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("找不到 GROQ_API_KEY，請檢查環境變數或 Streamlit Secrets。")
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

    # 接收用戶輸入
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
                    answer = "喺目前嘅講義入面搵唔到相關資料。"
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