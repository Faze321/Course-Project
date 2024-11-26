from os import getenv
from openai import OpenAI
import numpy as np
import faiss


client = OpenAI(
    api_key=getenv("DASHSCOPE_SHARE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 向量化
def Embedding(file_path,dim=1024):
    data_str = []
    data_emb = []
    with open(file_path,"r",encoding="utf-8") as f:
        for data in f:
            data_str.append(data)

    for i in range(0,len(data_str)):
        completion = client.embeddings.create(
            model="text-embedding-v3",
            input=data_str[i],
            dimensions=dim,
            encoding_format="float"
        )
        data_emb.append(completion.data[0].embedding)

    return data_emb

# 写入faiss库，并导入到本地文件
def StoreEmbeddingData(EmbededData,dim):
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(EmbededData).astype(np.float32))
    faiss.write_index(faiss_index,"faiss.index")


if __name__ == "__main__":
    path = "./运动鞋店铺知识库.txt"
    dim = 1024
    data_emb = Embedding(path,dim)
    StoreEmbeddingData(data_emb,dim)