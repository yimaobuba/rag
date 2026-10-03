import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os

# ====================== 1. 配置信息，改成你自己的 ======================
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
SAVE_DIR = "./embedding_data"
# API配置（这里以OpenAI为例，其他模型只改base_url和api_key就行）
# 1. 加载 .env 文件（会自动找项目根目录下的 .env）
load_dotenv()
# 2. 从环境变量读出密钥
api_key = os.getenv("ZHIPU_API_KEY")
if not api_key:
    raise ValueError("没有读到 ZHIPU_API_KEY，检查 .env 文件")
OPENAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"  # 国内模型比如豆包改成对应地址
MODEL_NAME = "glm-4-flash"  # 改成你调用的模型名

# ====================== 2. 加载本地存储的向量和文本块 ======================
def load_embedding_data(save_dir):
    save_dir = __import__('pathlib').Path(save_dir)
    vectors = np.load(save_dir / "vectors.npy")
    with open(save_dir / "metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return vectors, data["chunks"], data["metadata"]

# ====================== 3. 根据问题检索最相关的文本块 ======================
def retrieve_top_k(query, model, vectors, chunks, top_k=3):
    query_emb = model.encode([query])
    similarities = cosine_similarity(query_emb, vectors)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# ====================== 4. 调用大模型API生成回答 ======================
def generate_answer(query, context_chunks):
    # 把检索到的上下文拼接到prompt里
    context = "\n---\n".join(context_chunks)
    prompt = f"""请根据以下参考信息回答用户的问题，不要编造内容：
参考信息：
{context}

用户问题：{query}
"""
    # 初始化客户端并调用
    client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # 温度调小一点，回答更贴合参考信息
    )
    return response.choices[0].message.content

# ====================== 5. 主流程 ======================
if __name__ == "__main__":
    # 加载数据和模型
    vectors, chunks, metadata = load_embedding_data(SAVE_DIR)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # 输入你的问题
    user_query = input("请输入你的问题：")
    
    # 检索相关上下文
    top_context = retrieve_top_k(user_query, embed_model, vectors, chunks)
    print(f"\n已检索到{len(top_context)}条相关上下文，正在调用API生成回答...\n")
    
    # 生成并打印回答
    final_answer = generate_answer(user_query, top_context)
    print("最终回答：\n" + final_answer)