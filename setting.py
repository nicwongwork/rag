import os
from llama_index.core import Settings
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

# 1. 設定 API Key
api_key = os.environ.get("DEEPSEEK_API_KEY")

# 2. 配置 LLM (使用 DeepSeek)
Settings.llm = DeepSeek(
    model="deepseek-chat-2",
    temperature=0.1,
    api_key=api_key,
    timeout=60.0
)

# 3. 配置 Embedding (使用本地 HuggingFace，不依賴 OpenAI)
# 這樣就不會再彈出 OpenAI 的連線錯誤
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 4. 配置文本切分 (針對試卷，建議 chunk_size 稍微加大以保留上下文)
Settings.transformations = [
    SentenceSplitter(chunk_size=1024, chunk_overlap=100)
]