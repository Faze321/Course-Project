from os import getenv
from openai import OpenAI
import numpy as np
import faiss
import peewee as pw


client = OpenAI(
    api_key=getenv("DASHSCOPE_SHARE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

db = pw.MySQLDatabase(database='ai_course', user='root', password=getenv("SQL_PWD"), host='localhost')
class AIContext(pw.Model):
    id = pw.AutoField()
    text = pw.TextField()
    class Meta:
        database = db
        table_name = 'ai_context'

# 向量化
def Embedding(dim=1024):
    data_emb = []
    raw_count = AIContext.select().count()

    for i in range(0,raw_count):
        completion = client.embeddings.create(
            model="text-embedding-v3",
            input=AIContext.get(AIContext.id == i+1).text,
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
    dim = 1024
    db.connect()
    data_emb = Embedding(dim)
    StoreEmbeddingData(data_emb,dim)
    db.close()