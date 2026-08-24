import os
import json
import logging
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import requests
import time
import re

def clean_sql_markdown(text:str) -> str:
    # 去掉 ```sql 和 ``` 标记
    text = re.sub(r"^```sql\s*","",text,flags=re.MULTILINE)
    text = re.sub(r"```\s*$","",text,flags=re.MULTILINE)
    return text.strip()

# ========== 1、加载环境变量 ==========
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD"))
TOP_N = int(os.getenv("TOP_N"))

# 加载json业务配置
with open("config.json", "r", encoding="utf-8") as f:
    app_config = json.load(f)

# ========== 2、日志配置 ==========
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        # 删掉 StreamHandler()，不再把日志打印到终端屏幕
    ]
)
logger = logging.getLogger(__name__)
# ========== 3、从data/table_schema.json加载表schema ==========
def load_table_schemas(json_path: str = "data/table_schema.json") -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ========== 4、初始化Chroma向量库 ==========
def init_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name="bank_table_schema",
        embedding_function=embedding_func
    )

    # 如果集合为空，把schema写入向量库
    exist_count = collection.count()
    if exist_count == 0:
        schema_list = load_table_schemas()
        ids = []
        docs = []
        metas = []
        for item in schema_list:
            ids.append(item["table_name"])
            docs.append(item["schema_content"])
            metas.append({"table_name": item["table_name"]})
        collection.add(
            documents=docs,
            ids=ids,
            metadatas=metas
        )
        logger.info(f"向量库初始化完成，入库 {len(schema_list)} 张表schema")
    return collection

collection = init_chroma()

# ========== 5、带距离阈值的向量检索 ==========
def vector_retrieve(user_query: str, top_n: int, distance_threshold: float) -> str:
    res = collection.query(
        query_texts=[user_query],
        n_results=top_n
    )
    hit_docs = res["documents"][0]
    hit_distances = res["distances"][0]

    valid_docs = []
    for doc, dist in zip(hit_docs, hit_distances):
        logger.info(f"[DEBUG向量距离] dist={dist:.4f}")
        print(f"[DEBUG向量距离] dist={dist:.4f}")
        if dist <= distance_threshold:
            valid_docs.append(doc)

    if len(valid_docs) == 0:
        return ""
    return "\n".join(valid_docs)

# ========== 6、SQL安全校验，拦截高危操作 ==========
DANGER_KEYWORDS = {"drop", "alter", "delete", "truncate", "rename", "grant"}
def sql_safety_check(sql_text: str) -> tuple[bool, str]:
    lower_sql = sql_text.lower()
    for kw in DANGER_KEYWORDS:
        if kw in lower_sql:
            return False, f"安全拦截：检测到高危SQL关键字 [{kw}]，禁止执行"
    return True, "ok"

# ========== 7、调用DeepSeek LLM ==========
SYSTEM_PROMPT = """
你是数据分析助手，根据给到的数据表schema，把用户自然语言转为SQL。
规则：
1. 只允许SELECT查询语句；禁止drop、alter、delete等修改语句。
2. 如果用户需求模糊、缺少指标、检索到的表不匹配业务，直接输出：⚠️需求模糊，请补充指标信息。
3. 只输出最终结果，不要多余解释。
"""

def call_llm(user_question: str, context_schema: str):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【可用数据表schema】\n{context_schema}\n\n用户问题：{user_question}"}
    ]
    payload = {
        "model": app_config["llm_model"],
        "messages": messages,
        "temperature": 0.1
    }

    start_time = time.time()
    resp = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    elapsed = time.time() - start_time

    result = resp.json()
    content = result["choices"][0]["message"]["content"]

    # 获取token统计
    prompt_tokens = result["usage"]["prompt_tokens"]
    completion_tokens = result["usage"]["completion_tokens"]
    total_tokens = result["usage"]["total_tokens"]

    # DeepSeek‑chat公开价格（仅估算，非正式账单）
    input_price = 0.0014 / 1000
    output_price = 0.0028 / 1000
    est_cost = prompt_tokens * input_price + completion_tokens * output_price

    # 控制台打印耗时、token、估算费用
    print(f"\n----- LLM调用统计 -----")
    print(f"接口耗时：{elapsed:.2f} s")
    print(f"输入token：{prompt_tokens}，输出token：{completion_tokens}，总token：{total_tokens}")
    print(f"估算费用：¥{est_cost:.6f}")
    print(f"------------------------")

    logger.info(f"LLM调用｜耗时:{elapsed:.2f}s｜prompt_tokens:{prompt_tokens}｜completion_tokens:{completion_tokens}｜est_cost:{est_cost:.6f}")
    return content

# ========== 主控制台循环入口 ==========
def main():
    print("===== RAG‑SQL 银行自然语言转SQL工具 =====")
    while True:
        user_input = input("\n【真实向量RAG】请输入业务问题：")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("程序退出")
            break
        logger.info(f"用户输入：{user_input}")

        retrieved_schema = vector_retrieve(
            user_query=user_input,
            top_n=TOP_N,
            distance_threshold=DISTANCE_THRESHOLD
        )

        print("\n>>>【向量检索返回的表结构】")
        print(retrieved_schema if retrieved_schema else "无")

        if not retrieved_schema.strip():
            print("⚠️没有检索到和问题相关的数据表，无法生成SQL，请修改你的业务问题！")
            logger.info("向量层拦截：未找到匹配schema")
            continue

        llm_output = call_llm(user_input, retrieved_schema)
        llm_output = clean_sql_markdown(llm_output) #清理markdown代码块
        safety_ok, safety_msg = sql_safety_check(llm_output)
        if not safety_ok:
            print(f"\n{safety_msg}")
            logger.warning(f"SQL安全拦截：{safety_msg}")
            continue

        print(f"\n【LLM输出结果】\n{llm_output}")
        logger.info(f"LLM输出：{llm_output}")

if __name__ == "__main__":
    main()