from openai import OpenAI
import os

# 将 "SERVER_IP" 替换为服务器的实际 IP 或主机名
SERVER_IP = "localhost"  # 示例：替换为你的服务器 IP
model="/topsmodels/data-llm/qwen/Qwen2-0.5B"

client = OpenAI(
    base_url=f"http://{SERVER_IP}:8089/v1",  # 使用服务器 IP 而非 localhost
    api_key="token-abc123"
)

# 文本补全
completion = client.completions.create(
    model=model,
    prompt="Python是一种",
    max_tokens=50
)
print(completion.choices[0].text)

# 聊天补全
chat_completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手"},
        {"role": "user", "content": "请解释Python中的多线程是什么?"}
    ]
)
print(chat_completion.choices[0].message.content)
