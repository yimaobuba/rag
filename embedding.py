from sentence_transformers import SentenceTransformer
from text_splitter import chunks  # 从text_splitter导入chunks
import numpy as np
import json
from pathlib import Path

# 加载预训练模型（支持中文）
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 生成向量
vectors = model.encode(chunks)
print(vectors.shape)  # 输出每个文本块的向量维度

# 在这里定义基础metadata，和你的chunks一一对应
# 只需要给每个chunk分配一个id就可以先跑通流程
metadata = [{"chunk_id": idx} for idx, _ in enumerate(chunks)]

# 保存向量和元数据
save_dir = Path("./embedding_data")
save_dir.mkdir(parents=True, exist_ok=True)

# 保存向量
np.save(save_dir / "vectors.npy", np.array(vectors))
# 保存文本和元数据
data = {"chunks": chunks, "metadata": metadata}
with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("存储完成！文件保存在embedding_data文件夹下")