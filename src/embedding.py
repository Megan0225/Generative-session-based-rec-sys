import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: 加载数据
csv_file = "data/products_train_UK.csv"
data = pd.read_csv(csv_file)

# 检查是否包含必要字段
if 'title' not in data.columns or 'descrip' not in data.columns or 'id' not in data.columns:
    raise ValueError("CSV 文件必须包含 'title'、'descrip' 和 'id' 字段")

# 创建新字段，将 title 和 descrip 拼接
data['title_descrip'] = data['title'].astype(str) + " " + data['descrip'].astype(str)

# Step 2: 加载 Sentence-BERT 模型
model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cuda')  # 默认truncate=True模型输入限制128token

# Step 3: 对 desc 字段生成 embedding
title_descriptions = data['title_descrip'].tolist()
embeddings = model.encode(title_descriptions, show_progress_bar=True)

# 保存嵌入和 ID 到 .npy 文件
embedded_file_path = "data/"
np.save(embedded_file_path + "id.npy", data['id'].values)  # 保存 ID
np.save(embedded_file_path + "td_embedding.npy", embeddings)  # 保存嵌入
print("ID 和嵌入已保存为 .npy 文件")