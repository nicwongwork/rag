import os
import tempfile
import streamlit as st
import chromadb

from pathlib import Path
from dotenv import load_dotenv

# 1. Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ====== Configuration ======
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Premium UI Style
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
    """Use EphemeralClient (InMemory) to completely avoid Streamlit Cloud's hard disk write errors"""
    client = chromadb.EphemeralClient()
    return Chroma(client=client, embedding_function=get_embeddings())

@st.cache_resource
def get_llm(api_key: str):
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.3)

# ====== 主介面 ======

def main():
    st.set_page_config(page_title="Source from PDF", page_icon="📚", layout="wide")
    st.markdown(PREMIUM_STYLE, unsafe_allow_html=True)

    # Session State
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = set()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # retrieve InMemory Vectorstore
    vs = load_vectorstore()

    # --- Sidebar ---
    with st.sidebar:
        st.title("📚 Management Center")

        # Memory reset button (keep the one-click clear feature)
        if st.button("🔥 Clear All Books"):
            st.session_state["processed_files"] = set()
            st.session_state["messages"] = []
            st.cache_resource.clear()
            st.success("All memory has been reset!")
            st.rerun()

        st.divider()
        st.subheader("📥 Upload Books")
        uploaded_file = st.file_uploader("Upload PDF Lecture Notes", type=["pdf"], label_visibility="collapsed")

        if uploaded_file:
            if uploaded_file.name not in st.session_state["processed_files"]:
                with st.spinner("Indexing in progress (InMemory)..."):
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

        # Retrieve the list of books currently in memory
        all_data = vs.get()
        existing_files = sorted(list(set([m["source"] for m in all_data["metadatas"] if "source" in m])))

        st.divider()
        st.subheader("🗑️ Delete a Single Book")
        if existing_files:
            # Let the user choose which book to delete
            file_to_delete = st.selectbox("Select a lecture note to remove:", existing_files, key="delete_box")

            if st.button(f"Delete {file_to_delete}"):
                # 1. Find all data IDs corresponding to the source name
                target_data = vs.get(where={"source": file_to_delete})
                ids_to_delete = target_data.get("ids", [])

                if ids_to_delete:
                    # 2. Instruct ChromaDB to delete these IDs
                    vs.delete(ids=ids_to_delete)

                # 3. Remove from Session State
                if file_to_delete in st.session_state["processed_files"]:
                    st.session_state["processed_files"].remove(file_to_delete)

                st.success(f"Successfully deleted {file_to_delete}!")
                st.rerun()
        else:
            st.info("The database is currently empty")

        st.divider()
        st.subheader("🎯 Reading Settings")
        selected_file = st.selectbox(
            "Switch question scope:",
            ["All Books"] + existing_files,
            index=0
        )
        if existing_files:
            st.caption(f"Currently, {len(existing_files)} books are included")

    # --- Main Chat Area ---
    st.title("🎓 Study Assistant")
    st.markdown("*SOURCE TO YOUR STUDIES*")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found, please check your environment variables or Streamlit Secrets.")
        return

    try:
        llm = get_llm(api_key)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Receive user input
    if user_input := st.chat_input("Ask about the content in the books..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                search_kwargs = {"k": 4}
                if selected_file != "All Books":
                    search_kwargs["filter"] = {"source": selected_file}

                docs = vs.similarity_search(user_input, **search_kwargs)

                if not docs:
                    answer = "Cannot find relevant information in the current lecture notes."
                else:
                    context = "\n\n".join([d.page_content for d in docs])
                    prompt = f"""
                    You are an expert Book Explainer. Your task is to explain and answer the user's question based EXCLUSIVELY on the provided Context.

                    CRITICAL RULES:
                    1. Answer ONLY using the provided Context below. Do NOT use any outside knowledge, prior training data, or assumptions.
                    2. If the answer cannot be explicitly found in the Context, respond EXACTLY with: "I don't know, uploaded PDF doesn't provide the information"
                    3. Do NOT guess or extrapolate.
                    4. Please answer in Traditional Chinese (Hong Kong).

                    Context:
                    {context}

                    Question: {user_input}
                    Answer:
                    """

                    response = llm.invoke(prompt)
                    answer = response.content

                st.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()