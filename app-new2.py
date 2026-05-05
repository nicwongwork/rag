import os
import sys
import tempfile
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
APP_TITLE = "Source from PDF"
APP_SUBTITLE = "SOURCE TO YOUR STUDIES"

# Custom CSS for Premium UI
PREMIUM_STYLE = """
<style>
    .main {
        background-color: #0e1117;
    }
    .stMain {
        }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1c24 100%);
    }
    .sidebar .sidebar-content {
        background-color: #1a1c24;
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
    .stChatInputContainer {
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .css-1offfwp {
        background-color: #238636 !important;
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

PROMPT_TEMPLATE = (
    "You are a sophisticated Study Assistant. Use the provided context to answer the student's question accurately. "
    "If the answer isn't in the context, politely state that you don't know based on the available materials. "
    "\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

@st.cache_resource
def load_vectorstore() -> Chroma:
    "Loading vectorstore with HuggingFace embeddings. If it doesn't exist, it will be created on the fly when documents are added."
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectorstore

@st.cache_resource
def get_llm(api_key: str) -> ChatGroq:
    # Using Llama 3.3 70B via Groq for lightning-fast RAG
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.3,
    )
    #llm = ChatDeepSeek(
    #    model="deepseek-chat",
    #    temperature=0.3,
    #    api_base='https://api.deepseek.com/v1',
    #    api_key=api_key,
    #    timeout=60.0
    #)
    return llm

def build_context(chunks) -> str:
    return "\n\n".join(chunk.page_content for chunk in chunks)

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
    st.markdown(PREMIUM_STYLE, unsafe_allow_html=True)

    # Sidebar Header
    with st.sidebar:
        st.title(f"🔍 {APP_TITLE}")
        st.markdown(f"**{APP_SUBTITLE}**")
        st.divider()

        # Tools
        if st.button("🗑️ Reset Conversation"):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()

        # Knowledge Base Management
        st.subheader("📚 Knowledge Base")
        uploaded_file = st.file_uploader("Upload course material (PDF)", type=["pdf"])

        if "processed_files" not in st.session_state:
            st.session_state["processed_files"] = set()

        # Initialize vectorstore
        try:
            vectorstore = load_vectorstore()
        except Exception as exc:
            st.error(f"Engine Error: {exc}")
            return

        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state["processed_files"]:
                with st.spinner("Analyzing and indexing document..."):
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_path = tmp_file.name

                        loader = PyPDFLoader(tmp_path)
                        documents = loader.load()

                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=700,
                            chunk_overlap=100,
                        )
                        splits = splitter.split_documents(documents)
                        vectorstore.add_documents(splits)

                        st.session_state["processed_files"].add(uploaded_file.name)
                        st.success("Document added to knowledge base.")
                    except Exception as exc:
                        st.error(f"Indexing Error: {exc}")
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.remove(tmp_path)
            else:
                st.info(f"'{uploaded_file.name}' is indexed.")

    # Main UI
    st.title(f"🎓 {APP_TITLE}")
    st.markdown(f"*{APP_SUBTITLE}*")

    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # API Key Handling
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.warning("⚠️ Backend connection not established. Please check your configuration.")
        return

    try:
        llm = get_llm(api_key)
    except Exception as exc:
        st.error(f"Intelligence Engine Error: {exc}")
        return

    # Chat Display
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    user_input = st.chat_input("Ask anything about your studies...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🔍 Analyzing documents...")

            try:
                # Retrieve relevant context
                docs = vectorstore.similarity_search(user_input, k=4)

                if not docs:
                    answer = "I couldn't find any relevant information in your current study materials."
                else:
                    context = build_context(docs)
                    filled_prompt = PROMPT_TEMPLATE.format(context=context, question=user_input)

                    response = llm.invoke(filled_prompt)
                    answer = response.content

                placeholder.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})

            except Exception as exc:
                placeholder.markdown(f"⚠️ Service interruption: {exc}")

if __name__ == "__main__":
    main()