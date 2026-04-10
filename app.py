# ====================== 汽车评分RAG智能体 ======================
import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Literal
from operator import add

# LangChain 核心
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import pypdf

# ====================== 页面配置 ======================
st.set_page_config(page_title="汽车评分RAG智能体", page_icon="🚗", layout="wide")

# ====================== 配置与全局变量 ======================
load_dotenv()
base_dir = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(base_dir, "data", "品牌汽车大数据评分研究_毕业论文.pdf")
CSV_PATH = os.path.join(base_dir, "data", "综合评分结果_AHP熵权.csv")
FAISS_DB_PATH = os.path.join(base_dir, "faiss_index")
# 安全的环境变量读取方式
YOUR_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 加个校验，防止没配置环境变量
if not YOUR_API_KEY:
    raise ValueError("请在.env文件中配置DASHSCOPE_API_KEY！")


# ====================== 车型查询工具 ======================
def get_top5_cars():
    """获取综合排名前5的车型数据"""
    if 'car_df' not in st.session_state or st.session_state.car_df is None:
        return "暂无车型数据"
    top5 = st.session_state.car_df.head(5)[["排名", "车型名称", "综合得分", "产品力", "创新力"]].to_string(index=False)
    return f"综合排名Top5车型数据：\n{top5}"


def query_car_by_name(car_name: str):
    """根据车型名称查询得分数据"""
    if 'car_df' not in st.session_state or st.session_state.car_df is None:
        return "暂无车型数据"
    res = st.session_state.car_df[st.session_state.car_df["车型名称"].str.contains(car_name, na=False, case=False)]
    if res.empty:
        return f"未查询到「{car_name}」的相关数据"
    return res[["车型名称", "综合得分", "排名", "产品力", "创新力"]].to_string(index=False)


# ====================== 状态定义 ======================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    route: Literal["retrieve", "direct", "tool_call"]
    context: str


# ====================== 初始化：只在第一次运行时执行 ======================
@st.cache_resource(show_spinner="正在加载系统资源...")
def init_system():
    """初始化所有资源，只运行一次"""
    # 1. 加载CSV数据
    car_df = None
    if os.path.exists(CSV_PATH):
        car_df = pd.read_csv(CSV_PATH)
        if len(car_df.columns) >= 7:
            car_df.columns = ["车型名称", "产品力", "市场表现", "用户口碑", "创新力", "综合得分", "排名"] + list(
                car_df.columns[7:])

    # 2. 初始化Embedding
    embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=YOUR_API_KEY)

    # 3. 加载/构建向量库
    def load_pdf():
        docs = []
        reader = pypdf.PdfReader(PDF_PATH)
        for i, page in enumerate(reader.pages):
            txt = page.extract_text()
            if txt and txt.strip():
                for j in range(0, len(txt), 1000):
                    chunk = txt[j:j + 1000]
                    docs.append(Document(page_content=chunk, metadata={"page": i + 1, "source": f"论文第{i + 1}页"}))
        return docs

    if os.path.exists(FAISS_DB_PATH):
        vs = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = load_pdf()
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(FAISS_DB_PATH)

    # 4. 初始化LLM
    llm = ChatTongyi(model="qwen-turbo", temperature=0.1, dashscope_api_key=YOUR_API_KEY)

    return car_df, vs, llm


# ====================== 智能路由 ======================
def get_route(query: str):
    """根据用户问题，强制分配路由，100%稳定"""
    q = query.lower()
    # 车型数据关键词（只要包含这些词，就走工具调用）
    car_keywords = ["分", "得分", "排名", "最高", "第一", "top", "对比", "比亚迪", "特斯拉", "蔚来", "理想", "车型",
                    "汽车"]
    # 论文内容关键词
    paper_keywords = ["论文", "研究", "方法", "模型", "结论", "指标", "AHP", "熵权", "权重", "体系"]

    if any(k in q for k in car_keywords):
        return "tool_call"
    elif any(k in q for k in paper_keywords):
        return "retrieve"
    else:
        return "direct"


# ====================== 主界面 ======================
def main():
    # 1. 侧边栏设置
    st.sidebar.title("⚙️ 系统设置")
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 模型设置")
    model_choice = st.sidebar.selectbox("选择大模型", ["qwen-turbo", "qwen-plus", "qwen-max"], index=0)
    st.sidebar.subheader("🔍 检索设置")
    top_k = st.sidebar.slider("检索TopK", 1, 20, 4)

    # 2. 初始化系统（只运行一次）
    if 'car_df' not in st.session_state:
        with st.spinner("🚀 正在初始化系统，请稍候..."):
            car_df, vs, llm = init_system()
            st.session_state.car_df = car_df
            st.session_state.vs = vs
            st.session_state.llm = llm

    # 3. 主标题
    st.title("🚗 汽车品牌大数据评分RAG智能体")
    st.caption("基于LangChain + Streamlit 的可视化对话系统")

    # 4. 数据看板
    st.markdown("---")
    st.subheader("📊 车型数据看板")
    if st.session_state.car_df is not None:
        df = st.session_state.car_df
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🏆 Top10 车型综合排名")
            st.dataframe(df.head(10)[["排名", "车型名称", "综合得分", "产品力", "创新力"]], width='stretch')

        with col2:
            st.markdown("### 📈 综合得分对比")
            st.bar_chart(df.head(10).set_index("车型名称")["综合得分"])

    # 5. 对话区域
    st.markdown("---")
    st.subheader("💬 智能对话")

    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史对话
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    if prompt := st.chat_input("问我关于汽车评分或论文的问题..."):
        # 显示用户问题
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # 1. 智能路由
            route = get_route(prompt)
            context = ""

            # 2. 执行对应逻辑
            if route == "tool_call":
                # 车型工具调用：直接获取数据
                if any(k in prompt.lower() for k in ["最高", "第一", "top5", "前五", "排名前5"]):
                    context = get_top5_cars()
                else:
                    # 提取车型名称，查询对应数据
                    car_names = ["特斯拉", "比亚迪", "蔚来", "理想", "奔驰", "宝马", "福特"]
                    for car in car_names:
                        if car in prompt:
                            context = query_car_by_name(car)
                            break
                    if not context:
                        context = get_top5_cars()

            elif route == "retrieve":
                # 论文检索
                docs = st.session_state.vs.similarity_search(prompt, k=top_k)
                context = ""
                for i, d in enumerate(docs):
                    context += f"[{i + 1}] {d.metadata['source']}\n{d.page_content}\n\n"

            else:
                # 闲聊
                context = "你是汽车评分智能助手，友好回应用户的打招呼和闲聊。"

            # 3. 选择Prompt
            if route == "direct":
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "{context}"),
                    ("human", "{query}")
                ])
            else:
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", """你是《品牌汽车大数据评分研究》的专业智能助手。
请严格基于提供的参考内容回答，禁止编造数据。
回答要求：专业、严谨、简洁。
参考内容：{context}"""),
                    ("human", "{query}")
                ])

            # 4. 流式输出
            chain = prompt_template | st.session_state.llm | StrOutputParser()
            for chunk in chain.stream({"context": context, "query": prompt}):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

            # 保存到历史
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 6. 底部说明
    st.markdown("---")
    st.caption("💡 支持功能：论文内容检索、车型评分查询、Top排名查询、多车型对比")


if __name__ == "__main__":
    main()