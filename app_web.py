import os
import json
import time
import re
import logging
import sys
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import requests
import streamlit as st

# ========== 1、区分云端/本地：只有Streamlit Cloud才读取st.secrets ==========
is_cloud = "STREAMLIT_SERVER" in os.environ

if is_cloud:
    # 云端：读取平台Secrets
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
    DEEPSEEK_BASE_URL = st.secrets["DEEPSEEK_BASE_URL"]
    # 云端兜底默认参数
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.7"))
    TOP_N = int(os.getenv("TOP_N", "2"))
else:
    # 本地开发：只读取 .env，完全不触碰 st.secrets，避免报错
    load_dotenv()
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD"))
    TOP_N = int(os.getenv("TOP_N",3))

with open("config.json", "r", encoding="utf-8") as f:
    app_config = json.load(f)

# ========== 2、日志配置（保留之前修改好的） ==========
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

if is_cloud:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(console_handler)
else:
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/app_web.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
# ========== 3、加载表schema ==========
def load_table_schemas(json_path: str = "data/table_schema.json") -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ========== 4、初始化Chroma（缓存，避免单次会话重复重建） ==========
# ========== 4、初始化Chroma（内存模式，适配Streamlit云端） ==========
@st.cache_resource
def init_chroma():
    # 云端使用内存模式，不读写磁盘（免费实例磁盘会丢失）
    client = chromadb.Client()

    # 强制走国内镜像下载embedding模型，避免huggingface网络损坏
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    embedding_func = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="bank_table_schema",
        embedding_function=embedding_func
    )

    # 调试：打印当前集合文档数量
    doc_count = collection.count()
    print(f"【DEBUG】向量库文档总数量: {doc_count}")

    # 清空旧数据，防止损坏的embedding残留
    collection.delete()

    # 重新灌入全部表schema
    if collection.count() == 0:
        schema_list = load_table_schemas()
        collection.add(
            documents=[item["schema_content"] for item in schema_list],
            ids=[item["table_name"] for item in schema_list],
            metadatas=[{"table_name": item["table_name"]} for item in schema_list]
        )
        logger.info(f"向量库初始化完成，入库 {len(schema_list)} 张表")
        print(f"【DEBUG】重建完成，入库 {len(schema_list)} 张表")

    return collection

# 执行初始化，拿到全局collection对象
collection = init_chroma()

# ========== 5、向量检索（带距离阈值，返回详情供页面展示） ==========
def vector_retrieve(user_query: str, top_n: int, distance_threshold: float):
    res = collection.query(query_texts=[user_query], n_results=top_n)
    hit_docs = res["documents"][0]
    hit_distances = res["distances"][0]
    valid_docs = []
    detail_list = []
    for doc, dist in zip(hit_docs, hit_distances):
        passed = dist <= distance_threshold
        detail_list.append({"distance": round(dist, 4), "passed": passed, "doc": doc})
        if passed:
            valid_docs.append(doc)
    return "\n".join(valid_docs), detail_list

# ========== 6、SQL安全校验 ==========
DANGER_KEYWORDS = {"drop", "alter", "delete", "truncate", "rename", "grant"}
def sql_safety_check(sql_text: str):
    lower_sql = sql_text.lower()
    for kw in DANGER_KEYWORDS:
        if kw in lower_sql:
            return False, f"安全拦截：检测到高危SQL关键字 [{kw}]，禁止执行"
    return True, "ok"

# ========== 7、清理markdown代码块 ==========
def clean_sql_markdown(text: str) -> str:
    text = re.sub(r"^```sql\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()

# ========== 8、调用LLM ==========
SYSTEM_PROMPT = """
你是数据分析助手，根据给到的数据表schema，把用户自然语言转为SQL。
规则：
1. 只允许SELECT查询语句；禁止drop、alter、delete等修改语句。
2. 如果用户需求模糊、缺少指标、检索到的表不匹配业务，直接输出：⚠️需求模糊，请补充指标信息。
3. 只输出最终结果，不要多余解释。
"""

def call_llm(user_question: str, context_schema: str):
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【可用数据表schema】\n{context_schema}\n\n用户问题：{user_question}"}
    ]
    payload = {"model": app_config["llm_model"], "messages": messages, "temperature": 0.1}
    start = time.time()
    resp = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    elapsed = time.time() - start
    result = resp.json()
    content = result["choices"][0]["message"]["content"]
    usage = result["usage"]
    input_price = 0.0014 / 1000
    output_price = 0.0028 / 1000
    est_cost = usage["prompt_tokens"] * input_price + usage["completion_tokens"] * output_price
    stats = {
        "elapsed": round(elapsed, 2),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "est_cost": round(est_cost, 6)
    }
    logger.info(f"LLM调用｜{stats}｜输出:{content}")
    return content, stats

# ========== 9、Streamlit 页面 ==========
st.set_page_config(page_title="RAG-SQL 银行自然语言转SQL", layout="wide")
st.title("🏦 RAG-SQL 银行自然语言转SQL工具")
st.caption("基于 Chroma 向量检索 + DeepSeek 大模型，支持语义召回表结构、距离阈值过滤、SQL安全校验")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 当前配置")
    st.write(f"Embedding模型：`{EMBEDDING_MODEL}`")
    st.write(f"距离阈值：`{DISTANCE_THRESHOLD}`")
    st.write(f"Top-N召回：`{TOP_N}`")
    st.write(f"LLM模型：`{app_config['llm_model']}`")
    st.divider()
    st.subheader("📋 测试用例")
    st.code("统计放贷每个月总金额", language="text")
    st.code("各个城市逾期贷款金额", language="text")
    st.code("统计客户的数据", language="text")
    st.code("今天吃什么饭", language="text")

# 主区域
user_input = st.text_area("💬 请输入你的业务问题", height=80, placeholder="例如：统计2025年每个月的贷款总金额")
col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🚀 生成SQL", type="primary", use_container_width=True)
with col2:
    clear = st.button("🔄 清空", use_container_width=True)

if clear:
    st.rerun()

if submit and user_input.strip():
    logger.info(f"用户输入：{user_input}")

    # 第一步：向量检索
    with st.spinner("正在进行向量检索..."):
        retrieved_schema, detail_list = vector_retrieve(user_input, TOP_N, DISTANCE_THRESHOLD)

    st.subheader("🔍 向量检索结果")
    for i, d in enumerate(detail_list, 1):
        status = "✅ 通过阈值" if d["passed"] else "❌ 低于阈值，已过滤"
        st.write(f"**候选 {i}**：distance = `{d['distance']}` → {status}")
        with st.expander("查看该候选schema内容", expanded=False):
            st.code(d["doc"], language="sql")

    # 向量层拦截
    if not retrieved_schema.strip():
        st.error("⚠️ 没有检索到和问题相关的数据表，无法生成SQL，请修改你的业务问题！")
        logger.info("向量层拦截：未找到匹配schema")
    else:
        st.success(f"✅ 成功召回有效schema，已送入大模型")

        # 第二步：调用LLM
        with st.spinner("大模型正在生成SQL..."):
            raw_output, stats = call_llm(user_input, retrieved_schema)
        clean_output = clean_sql_markdown(raw_output)

        # 第三步：SQL安全校验
        safety_ok, safety_msg = sql_safety_check(clean_output)

        st.subheader("🤖 大模型输出")
        if not safety_ok:
            st.error(safety_msg)
            logger.warning(f"SQL安全拦截：{safety_msg}")
        elif "⚠️需求模糊" in clean_output:
            st.warning(clean_output)
        else:
            st.code(clean_output, language="sql")
            st.download_button("📥 下载SQL文件", clean_output, file_name="query.sql", mime="text/plain")

        # 调用统计
        st.subheader("📊 调用统计")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("接口耗时", f"{stats['elapsed']}s")
        c2.metric("输入token", stats["prompt_tokens"])
        c3.metric("输出token", stats["completion_tokens"])
        c4.metric("总token", stats["total_tokens"])
        c5.metric("估算费用", f"¥{stats['est_cost']}")