# index_builder.py
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

DATA_DIR = Path("./data/pdfs")
INDEX_DIR = Path("./storage")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def build_index():
    docs = SimpleDirectoryReader(str(DATA_DIR), recursive=True).load_data()
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    return index

def load_index():
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    index = load_index_from_storage(storage_context)
    return index
