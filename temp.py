import os
from openai import OpenAI

# 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
# 初始化Openai客户端，从环境变量中读取您的API Key
client = OpenAI(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
    # 从环境变量中获取您的 API Key
    api_key="c01a174f-bd1e-4407-93fd-946377d67d54"
)

# # Non-streaming:
# print("----- standard request -----")
# completion = client.chat.completions.create(
#     model="bot-20250704000214-252z2",  # bot-20250704000214-252z2 为您当前的智能体的ID，注意此处与Chat API存在差异。差异对比详见 SDK使用指南
#     messages=[
#         {"role": "system", "content": "你是DeepSeek，是一个 AI 人工智能助手"},
#         {"role": "user", "content": "今天特斯拉的股价是多少？"},
#     ],
# )
# print(completion.choices[0].message.content)
# if hasattr(completion, "references"):
#     print(completion.references)
# if hasattr(completion.choices[0].message, "reasoning_content"):
#     print(completion.choices[0].message.reasoning_content)  # 对于R1模型，输出reasoning content

# Streaming:
ticker = "TSLA"
curr_date = "2025-07-09"
print("----- streaming request -----")
stream = client.chat.completions.create(
    model="bot-20250704000214-252z2",  # bot-20250704000214-252z2 为您当前的智能体的ID，注意此处与Chat API存在差异。差异对比详见 SDK使用指南
    messages=[
        {"role": "system", "content": "你是DeepSeek，是一个 AI 人工智能助手"},
        # {"role": "user", "content": f"分析一下，股票 {ticker} 从 20250709 向前 7 天期间市场情绪情况"},
        # {"role": "user", "content": f"最近一个月，关于股票 {ticker} 的基本面（Fundamental）讨论"},
        {"role": "user", "content": f"前7 天至今{curr_date}期间发布的全球或宏观经济新闻吗，请确保仅获取该时间段内发布的、有助于交易决策的信息。"},
    ],
    stream=True,
)
res = ""
for chunk in stream:
    # print(chunk, end="")
    # if hasattr(chunk, "references"):
    #     print(chunk.references, end="")
    # if not chunk.choices:
    #     continue
    if chunk.choices[0].delta.content:
        res += chunk.choices[0].delta.content
        print(chunk.choices[0].delta.content, end="")
    # elif hasattr(chunk.choices[0].delta, "reasoning_content"):
    #     print(chunk.choices[0].delta.reasoning_content, end="")
print("######")
print(res)