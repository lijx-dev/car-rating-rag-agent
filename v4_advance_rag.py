# ====================== 汽车评分RAG高级完整版 ======================
import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Literal
from operator import add

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

# ====================== 配置 ======================
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
class CarQueryInput(BaseModel):
    car_name: str = Field(..., description="要查询的车型名称，如'特斯拉model3'、'比亚迪汉'")
    query_type: Literal["basic", "full", "rank", "compare"] = Field(default="basic",
                                                                    description="查询类型：basic=基础得分，full=全量数据，rank=排名，compare=多车型对比")
    compare_cars: list[str] = Field(default_factory=list, description="对比车型列表，仅compare类型需要")


def query_car_rating(car_name: str, query_type: str = "basic", compare_cars: list = None):
    """
    汽车评分查询工具，用于查询车型的综合得分、排名、各维度得分及多车型对比
    :param car_name: 要查询的车型名称
    :param query_type: 查询类型，可选basic/full/rank/compare
    :param compare_cars: 对比车型列表，仅compare类型使用
    :return: 车型评分数据字符串
    """
    if car_df is None:
        return "暂无车型数据"
    compare_cars = compare_cars or []
    all_cars = [car_name] + compare_cars
    res = car_df.copy()
    # 多车型模糊匹配
    filter_mask = pd.Series([False] * len(res))
    for car in all_cars:
        filter_mask = filter_mask | res["车型名称"].str.contains(car, na=False, case=False)
    res = res[filter_mask]
    if res.empty:
        return f"未查询到{all_cars}的相关数据"
    # 根据查询类型返回对应数据
    if query_type == "basic":
        return res[["车型名称", "综合得分", "排名", "产品力", "创新力"]].to_string(index=False)
    elif query_type == "full":
        return res.to_string(index=False)
    elif query_type == "rank":
        top5 = car_df.head(5)[["排名", "车型名称", "综合得分"]].to_string(index=False)
        return f"综合排名Top5：\n{top5}\n\n你查询的车型：\n{res[['车型名称', '综合得分', '排名']].to_string(index=False)}"
    elif query_type == "compare":
        return f"多车型对比数据：\n{res[['车型名称', '综合得分', '排名', '产品力', '创新力']].to_string(index=False)}"
    return res[["车型名称", "综合得分", "排名"]].to_string(index=False)


car_tool = StructuredTool.from_function(
    func=query_car_rating,
    name="query_car_rating",
    description="汽车评分专用工具，用于查询车型的精准评分、排名、各维度得分、多车型对比",
    args_schema=CarQueryInput
)
tools = [car_tool]


# ====================== 状态定义 ======================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    route: Literal["retrieve", "direct", "tool_call"]
    context: str


# ====================== 加载数据 ======================
car_df = None
if os.path.exists(CSV_PATH):
    car_df = pd.read_csv(CSV_PATH)
    # 适配CSV列名，兼容不同格式
    if len(car_df.columns) >= 7:
        car_df.columns = ["车型名称", "产品力", "市场表现", "用户口碑", "创新力", "综合得分", "排名"] + list(
            car_df.columns[7:])
    print("✅ CSV车型数据加载成功")

# ====================== PDF读取 ======================
embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=YOUR_API_KEY)


def load_pdf_simple():
    """加载PDF并按1000字符分块，保留页码元数据"""
    docs = []
    reader = pypdf.PdfReader(PDF_PATH)
    for i, page in enumerate(reader.pages):
        txt = page.extract_text()
        if txt and txt.strip():
            # 按1000字符分块，避免过长
            for j in range(0, len(txt), 1000):
                chunk = txt[j:j + 1000]
                docs.append(Document(
                    page_content=chunk,
                    metadata={"page": i + 1, "source": f"论文第{i + 1}页"}
                ))
    print(f"✅ PDF加载完成，共{len(docs)}个文本块")
    return docs


def get_retriever():
    """获取FAISS向量检索器，本地存在则加载，否则重新构建"""
    if os.path.exists(FAISS_DB_PATH):
        print("📦 加载本地向量库")
        return FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = load_pdf_simple()
    print("🔨 构建FAISS向量库")
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(FAISS_DB_PATH)
    return vs


vs = get_retriever()

# ====================== LLM初始化 ======================
llm = ChatTongyi(model="qwen-turbo", temperature=0.1, dashscope_api_key=YOUR_API_KEY)
llm_tools = llm.bind_tools(tools)


# ====================== 智能路由节点 ======================
def route_question(state: AgentState):
    """
    智能路由节点，根据用户问题类型分配处理分支
    - tool_call：车型数据查询
    - retrieve：论文内容检索
    - direct：闲聊/打招呼
    """
    q = state["query"].lower()
    # 车型数据关键词
    if any(k in q for k in ["分", "排名", "对比", "比亚迪", "特斯拉", "蔚来", "理想", "车型"]):
        return {"route": "tool_call"}
    # 论文内容关键词
    if any(k in q for k in ["论文", "研究", "方法", "模型", "结论", "指标", "AHP", "熵权"]):
        return {"route": "retrieve"}
    # 闲聊场景
    return {"route": "direct"}


# ====================== 检索/工具/生成节点 ======================
def retrieve(state: AgentState):
    """论文检索节点：执行向量检索，生成上下文"""
    docs = vs.similarity_search(state["query"], k=4)
    ctx = ""
    for i, d in enumerate(docs):
        ctx += f"[{i + 1}] {d.metadata['source']}\n{d.page_content}\n\n"
    return {"context": ctx}


def call_tool(state: AgentState):
    """工具调用节点：调用车型评分工具，获取精准数据"""
    r = llm_tools.invoke([HumanMessage(content=state["query"])])
    for tc in r.tool_calls:
        out = car_tool.invoke(tc["args"])
        return {"context": f"车型精准数据：\n{out}"}
    return {"context": "未查询到相关车型数据"}


def gen_answer(state: AgentState):
    """回答生成节点：准备生成最终回答"""
    return {"context": state["context"]}


def direct_ans(state: AgentState):
    """直接回答节点：处理闲聊场景"""
    return {"context": "闲聊场景，无需检索"}


# ====================== 构建LangGraph工作流 ======================
g = StateGraph(AgentState)
# 添加节点
g.add_node("route", route_question)
g.add_node("retrieve", retrieve)
g.add_node("tool", call_tool)
g.add_node("generate", gen_answer)
g.add_node("direct", direct_ans)

# 构建边
g.add_edge(START, "route")
# 条件路由
g.add_conditional_edges(
    "route",
    lambda s: s["route"],
    {
        "retrieve": "retrieve",
        "tool_call": "tool",
        "direct": "direct"
    }
)
# 后续流程
g.add_edge("retrieve", "generate")
g.add_edge("tool", "generate")
g.add_edge("generate", END)
g.add_edge("direct", END)

# 编译智能体
app = g.compile()

# ====================== 主程序：流式对话 ======================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚗 汽车品牌大数据评分智能助手（高级完整版）")
    print("=" * 80)
    print("✨ 支持功能：论文精准检索 | 车型评分查询 | 多车型对比 | 智能路由")
    print("💡 输入 'quit' 退出\n")

    while True:
        inp = input("你: ")
        if inp.lower() == "quit":
            print("\n👋 感谢使用，再见！")
            break

        # 调用智能体
        final_state = app.invoke({
            "query": inp,
            "messages": [HumanMessage(content=inp)]
        })

        # 流式输出回答
        print("AI: ", end="", flush=True)
        route = final_state["route"]
        context = final_state["context"]

        # 选择对应Prompt
        if route == "direct":
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是《品牌汽车大数据评分研究》的智能助手，友好回应用户的打招呼和闲聊。"),
                ("human", "{query}")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是《品牌汽车大数据评分研究》毕业论文的专业智能助手。
请严格基于提供的参考内容回答，禁止编造数据。
回答要求：专业、严谨、简洁，数据类问题标注来源，论文类问题标注页码。
参考内容：{context}"""),
                ("human", "{query}")
            ])

        # 流式打印
        chain = prompt | llm | StrOutputParser()
        for chunk in chain.stream({"context": context, "query": inp}):
            print(chunk, end="", flush=True)
        print("\n")