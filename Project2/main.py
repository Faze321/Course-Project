from os import getenv
from openai import OpenAI
import faiss
import numpy as np
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

# 判断是否结束对话
def IsExit(user_input):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "Your task is to determine if the user wants to end this conversation, if so return 'T', if not return 'F'"},
            {"role": "user", "content": user_input},
        ],
    )

    return completion.choices[0].message.content == "T"


def GetResponse(dim=1024,faiss_path="./Project2/faiss.index"):
    faiss_index = faiss.read_index(faiss_path)
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
    db.connect()
    while True:
        user_input = input("\nUser:")
        if IsExit(user_input):
            print("感谢您的咨询，再见")
            # print("Thank you for your inquiry. Goodbye.")
            db.close()
            break
        user_embedding = client.embeddings.create(
                model="text-embedding-v3",
                input=user_input,
                dimensions=dim,
                encoding_format="float"
        )
        embedded_user_input = user_embedding.data[0].embedding
        _, Index = faiss_index.search(np.array([embedded_user_input]).astype(np.float32), k=2)
        # print(Index)

        messages.append({"role": "assistant", "content": AIContext.get(AIContext.id == Index[0][0]+1).text})
        messages.append({"role": "assistant", "content": AIContext.get(AIContext.id == Index[0][1]+1).text})
        messages.append({"role": "user", "content": user_input})
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            stream=True
        )
        full_reply = ""
        for chunk in completion:
            print(chunk.choices[0].delta.content, end="", flush=True)
            full_reply += chunk.choices[0].delta.content
        messages.append({"role": "system", "content": full_reply})

if __name__ == "__main__":
    dim = 1024
    GetResponse(dim)




