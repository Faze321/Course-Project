import os
from openai import OpenAI
try:
    client = OpenAI(
    api_key=os.getenv("DASHSCOPE_SHARE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # 判断是否结束对话
    def IsExit(UserInput):
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "Your task is to determine if the user wants to end this conversation, if so return 'T', if not return 'F'"},
                {"role": "user", "content": UserInput},
            ],
        )
        return completion.choices[0].message.content == "T"
    # 对话
    def Response():

        messages = [
            {"role": "system","content": "You are a helpful assistant."},
        ]
        while True:
            user_input = input("\nUser: ")
            messages.append({"role": "user", "content": user_input})
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                stream=True,
            )
            if IsExit(user_input):
                print("感谢您的咨询，再见")
                break
            full_reply = ""
            for chunk in completion:
                each_reply = chunk.choices[0].delta.content
                print(each_reply, end="", flush=True)
                full_reply += each_reply
            messages.append({"role": "system", "content": full_reply})


    if __name__ == "__main__":
        Response()


except Exception as e:
    print(f"错误信息:{e}")