from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

# tiktoken仅做粗略估算
def count_token(text: str, model="gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


if __name__ == "__main__":
    user_input = input("\n请输入你的问题：")

    print(f"\n输入文本：{user_input}")
    print(f"【本地粗略估算输入token】：{count_token(user_input)}")

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        temperature=0,
        top_p=0.95,
        max_tokens=1000,
        messages=[
        {"role": "user", "content": user_input}
    ]
    )

    answer = response.choices[0].message.content
    print("\n========模型返回回答========")
    print(answer)

    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    # DeepSeek‑v4‑flash 价格换算
    price_in_per_m_input = 0.14   # USD / 1M tokens 输入
    price_in_per_m_output = 0.28  # USD / 1M tokens 输出
    rate = 7.2

    cost_input = prompt_tokens / 1_000_000 * price_in_per_m_input * rate
    cost_output = completion_tokens / 1_000_000 * price_in_per_m_output * rate
    cost_total = cost_input + cost_output

    print("\n========API真实Token与费用估算（仅供参考）========")
    print(f"输入prompt_tokens：{prompt_tokens}")
    print(f"输出completion_tokens：{completion_tokens}")
    print(f"总total_tokens：{total_tokens}")
    print(f"✅本次预估花费：{cost_total:.4f} 元")