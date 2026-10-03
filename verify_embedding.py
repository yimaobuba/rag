import numpy as np
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ==============================
# 1. 加载数据
# ==============================
save_dir = Path("./embedding_data")
vectors = np.load(save_dir / "vectors.npy")
with open(save_dir / "metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data["chunks"]
metadata = data["metadata"]

print(f"加载了 {len(vectors)} 个向量，每个向量维度为 {vectors.shape[1]}")

# ==============================
# 2. 查询示例
# ==============================
query_text = "如何保持专心？"  # 替换为你想查询的内容

# 使用相同的模型进行嵌入
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
query_vector = model.encode([query_text])

# ==============================
# 3. 计算余弦相似度
# ==============================
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(query_vector, vectors)[0]
top_k_indices = similarities.argsort()[::-1][:3]  # 取最相似的3个

print("\n最相似的文本块：")
for idx in top_k_indices:
    print(f"相似度: {similarities[idx]:.4f}")
    print(f"chunk_id: {metadata[idx]['chunk_id']}")
    print(f"文本: {chunks[idx]}")
    print("-" * 50)