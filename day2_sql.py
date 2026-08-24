from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

def clean_sql(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```sql"):
        text = text[6:].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


# ============【离线知识库：相当于向量库里存的各个表的chunk】============
schema_store = [
    {
        "table_name": "bank_loan",
        "keywords": ["贷款","放款","逾期","pd","lgd","ead","loan_amount"],
        "schema_text": """
表名：bank_loan
字段：
loan_id 贷款ID，主键
cust_id 客户ID
loan_amount 贷款金额
pd 违约概率
lgd 违约损失率
ead 违约风险暴露
issue_date 放款日期 'yyyy‑mm‑dd'
status 贷款状态，取值:active/overdue/closed
"""
    },
    {
        "table_name": "bank_customer",
        "keywords": ["客户","年龄","城市","收入","income","age","city"],
        "schema_text": """
表名：bank_customer
字段：
cust_id 客户ID主键
age 客户年龄
city 所在城市
income 年收入
"""
    }
]

# ============【模拟检索函数！替代向量数据库相似度查询】============
def mock_retrieve(user_query: str):
    """
    输入用户问题，模拟向量检索，返回匹配到的表结构文本
    真实RAG这里会做embedding+向量库搜索；这里简化：关键词匹配
    """
    hit_schemas = []
    for item in schema_store:
        for kw in item["keywords"]:
            if kw in user_query:
                hit_schemas.append(item["schema_text"])
                break
    # 把检索命中的多个表拼接成一大段字符串返回
    return "\n".join(hit_schemas)


system_prompt = """
你是专业银行数据分析师，只输出标准SQL，严格遵守下面全部硬性规则：
1. 只能使用提供给你的表和字段，严禁编造不存在的表名、字段；
2. 只返回纯SQL文本，禁止任何解释、markdown、说明文字；
3. 日期字段issue_date格式为'yyyy‑mm‑dd'；
4. 统计类需求，正确使用group by；
5. 如果需要多表，正确写join关联cust_id；
6.【最重要强制规则】：如果用户需求模糊、关键词语义不明确、不知道要计算哪些指标，**禁止生成SQL！！！**
必须原样输出：【信息不足，无法生成SQL】，不要自行脑补指标、不要猜用户意图。
"""


if __name__ == "__main__":
    user_question = input("\n【RAG模拟】请输入业务问题：")

    retrieved_schema = mock_retrieve(user_question)
    print(f"\n>>>【模拟检索返回的表结构】\n{retrieved_schema}")

    # =========新增前置兜底：检索为空直接拦截，不调用LLM=========
    if not retrieved_schema.strip():
        print("\n⚠️没有检索到和问题相关的数据表，无法生成SQL，请修改你的业务问题！")
    else:
        messages = [
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": f"""
【检索得到可用表结构】
{retrieved_schema}

用户业务问题：{user_question}
"""}
        ]

        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            temperature=0,
            max_tokens=1000,
            stream=False,
            messages=messages
        )

        raw_out = resp.choices[0].message.content.strip()
        sql_out = clean_sql(raw_out)

        if sql_out == "【信息不足，无法生成SQL】":
            print("\n⚠️需求模糊，请补充指标信息")
        else:
            print("\n====最终生成SQL====")
            print(sql_out)

        pt = resp.usage.prompt_tokens
        ct = resp.usage.completion_tokens
        cost = (pt/1e6*0.14 + ct/1e6*0.28)*7.2
        print(f"\nprompt_tokens:{pt}, completion_tokens:{ct}，预估花费 {cost:.4f}")