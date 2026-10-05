import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os

# ===== 配置 =====
load_dotenv(Path(__file__).parent / ".env")
api_key = os.getenv("ZHIPU_API_KEY")
if not api_key:
    raise ValueError("没有读到 ZHIPU_API_KEY，检查 .env 文件")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SAVE_DIR = "./embedding_data"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MODEL_NAME = "glm-4-flash"

# ===== 全局加载一次（避免每次请求都重新加载）=====
_embed_model = None
_vectors = None
_chunks = None

def init():
    """启动时加载模型和数据，只加载一次"""
    global _embed_model, _vectors, _chunks
    save_dir = Path(SAVE_DIR)
    _vectors = np.load(save_dir / "vectors.npy")
    with open(save_dir / "metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    _chunks = data["chunks"]
    _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def retrieve_top_k(query, top_k=3):
    query_emb = _embed_model.encode([query], normalize_embeddings=True)
    similarities = cosine_similarity(query_emb, _vectors)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [{"text": _chunks[i], "score": float(similarities[i])} for i in top_indices]

def generate_answer(query, context_chunks):
    context = "\n---\n".join([c["text"] for c in context_chunks])
    prompt = f"""请根据以下参考信息回答用户的问题，不要编造内容：
参考信息：
{context}

用户问题：{query}
"""
    client = OpenAI(api_key=api_key, base_url=ZHIPU_BASE_URL)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

def ask(query, top_k=3):
    """对外暴露的统一入口"""
    contexts = retrieve_top_k(query, top_k)
    answer = generate_answer(query, contexts)
    return {"answer": answer, "sources": contexts}