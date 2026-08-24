from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
client_llm = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

def clean_sql(raw_text: str) -> str:
    """剥离sql markdown标记"""
    text = raw_text.strip()
    if text.startswith("```sql"):
        text = text[6:].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

# ---------------------- 1. 离线准备：业务表schema知识库 ----------------------
schema_list = [
    {
        "table_name": "bank_loan",
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

# ---------------------- 2. 初始化Chroma向量库，加载Embedding模型 ----------------------
# 使用本地开源embedding模型
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
# 持久化到本地文件夹 ./chroma_db，程序关闭数据不会丢失
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# 创建集合，相当于一张向量表
collection = chroma_client.get_or_create_collection(
    name="bank_table_schema",
    embedding_function=embedding_func
)

# 把schema写入向量库；先清空旧数据，避免重复插入
collection.delete(where={"table":{"$ne":""}})
for item in schema_list:
    collection.add(
        documents=[item["schema_text"]],
        metadatas=[{"table": item["table_name"]}],
        ids=[item["table_name"]]
    )

# ---------------------- 3. 真实向量检索函数 ----------------------
def vector_retrieve(user_query: str, top_n=2, distance_threshold=0.7) -> str:
    """
    带距离阈值过滤的向量检索
    :param user_query: 用户问题
    :param top_n: 最多返回几条
    :param distance_threshold: 距离阈值，大于该值直接丢弃
    :return: 拼接好的schema字符串；无符合条件返回空字符串
    """
    res = collection.query(
        query_texts=[user_query],
        n_results=top_n
    )
    hit_docs = res["documents"][0]
    hit_distances = res["distances"][0]  # 获取每条对应的距离

    valid_docs = []
    for doc, dist in zip(hit_docs, hit_distances):
        print(f"[DEBUG向量距离] dist={dist:.4f}")
        if dist <= distance_threshold:
            valid_docs.append(doc)

    if len(valid_docs) == 0:
        return ""
    return "\n".join(valid_docs)


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
    user_question = input("\n【真实向量RAG】请输入业务问题：")

    retrieved_schema = vector_retrieve(user_question)
    print(f"\n>>>【向量检索返回的表结构】\n{retrieved_schema}")

    # 前置兜底：检索为空直接拦截，不调用LLM
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

        resp = client_llm.chat.completions.create(
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