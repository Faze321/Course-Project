from os import getenv
from openai import OpenAI
import faiss
import numpy as np

client = OpenAI(
    api_key=getenv("DASHSCOPE_SHARE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

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


def GetResponse(offline_database, dim=1024,faiss_path="./faiss.index"):
    faiss_index = faiss.read_index(faiss_path)
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
    while True:
        user_input = input("\nUser:")
        if IsExit(user_input):
            print("感谢您的咨询，再见")
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

        data_str = []
        relative_data = []
        # 读取离线数据库
        with open(offline_database,"r",encoding="utf-8") as f:
            for data in f:
                data_str.append(data)
        for i in Index[0]:
            relative_data.append(data_str[i])
        
        messages.append({"role": "assistant", "content": relative_data[0]})
        messages.append({"role": "assistant", "content": relative_data[1]})
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
    path = "./运动鞋店铺知识库.txt"
    dim = 1024
    GetResponse(path,dim)




