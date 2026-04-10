# -------------------------- 1. 基础配置与导入 --------------------------
import os
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Literal
from operator import add

# LangChain 核心
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# 通义千问
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

# LangGraph (新增)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# -------------------------- 2. 初始化环境 --------------------------
load_dotenv()
base_dir = os.path.dirname(os.path.abspath(__file__))

# 配置文件路径
PDF_PATH = os.path.join(base_dir, "data", "品牌汽车大数据评分研究_毕业论文.pdf")
CSV_PATH = os.path.join(base_dir, "data", "综合评分结果_AHP熵权.csv")
FAISS_DB_PATH = os.path.join(base_dir, "faiss_index")

# 你的通义千问 API Key
# 安全的环境变量读取方式
YOUR_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 加个校验，防止没配置环境变量
if not YOUR_API_KEY:
    raise ValueError("请在.env文件中配置DASHSCOPE_API_KEY！")


# -------------------------- 3. 定义 State (对应 TypeScript 的 AgentStateAnnotation) --------------------------
class AgentState(TypedDict):
    """智能体的状态定义"""
    messages: Annotated[Sequence[BaseMessage], add_messages]  # 对话历史
    query: str  # 用户当前问题
    route: Literal["retrieve", "direct"]  # 路由判断结果
    documents: list  # 检索到的文档


# -------------------------- 4. 定义路由 Schema (对应 TypeScript 的 z.object) --------------------------
class RouteQuery(BaseModel):
    """路由判断的结构化输出"""
    route: Literal["retrieve", "direct"] = Field(
        ...,
        description="判断用户的问题是否需要检索论文知识库：retrieve=需要检索，direct=不需要检索（直接回答/闲聊）"
    )


# -------------------------- 5. 加载 CSV 数据与向量库  --------------------------
# 5.1 加载 CSV
car_df = None
if os.path.exists(CSV_PATH):
    car_df = pd.read_csv(CSV_PATH)
    car_df.columns = [
        "车型名称", "产品力", "市场表现", "用户口碑", "创新力",
        "综合得分", "排名", "w_AHP_产品力", "w_熵权_产品力", "w_组合_产品力",
        "w_AHP_市场表现", "w_熵权_市场表现", "w_组合_市场表现",
        "w_AHP_用户口碑", "w_熵权_用户口碑", "w_组合_用户口碑",
        "w_AHP_创新力", "w_熵权_创新力", "w_组合_创新力"
    ]
    print("✅ CSV 车型数据加载成功")


# 5.2 构建/加载向量库
def build_or_load_vectorstore():
    if os.path.exists(FAISS_DB_PATH):
        print("正在加载本地向量库...")
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=YOUR_API_KEY
        )
        return FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    print("正在构建向量库（首次运行较慢）...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=YOUR_API_KEY
    )
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(FAISS_DB_PATH)
    print("✅ 向量库构建完成")
    return vectorstore


vectorstore = build_or_load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------- 6. 初始化大模型 --------------------------
llm = ChatTongyi(
    model="qwen-turbo",
    temperature=0.1,
    dashscope_api_key=YOUR_API_KEY
)

# -------------------------- 7. 定义 Prompt (对应 TypeScript 的 ROUTER_SYSTEM_PROMPT 和 RESPONSE_SYSTEM_PROMPT) --------------------------
# 7.1 路由判断 Prompt
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能路由助手。请判断用户的问题是否需要检索《品牌汽车大数据评分研究》毕业论文知识库。
    以下情况**不需要**检索（direct）：
    - 闲聊、打招呼（如"你好"、"谢谢"）
    - 简单的常识问题
    - 问题明确不涉及论文内容

    以下情况**需要**检索（retrieve）：
    - 询问论文的研究方法、模型、结论
    - 询问汽车评分、排名、指标体系
    - 任何可能涉及论文内容的专业问题

    用户问题：{query}
    """),
])

# 7.2 回答生成 Prompt
RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是《品牌汽车大数据评分研究》毕业论文的专业智能助手。
    请根据以下参考资料回答用户的问题。

    参考资料：
    {context}

    核心结论（如果涉及，请优先提及）：
    - 创新力是新时代汽车品牌的首要竞争力
    - 新能源车型已全面超越传统燃油车型
    - 多维均衡发展是品牌持久竞争力的根本保证

    如果答案不在参考资料中，请明确告知，不要编造。
    """),
    ("placeholder", "{messages}"),
])


# -------------------------- 8. 定义 LangGraph 节点 (对应 TypeScript 的各个 Node) --------------------------

# 节点 1: checkQueryType (智能路由判断)
def check_query_type(state: AgentState, config: RunnableConfig):
    """对应 TypeScript 的 checkQueryType"""
    print(f"🤔 正在判断问题类型: {state['query']}")

    # 绑定结构化输出
    structured_llm = llm.with_structured_output(RouteQuery)

    # 调用路由判断
    prompt_value = ROUTER_PROMPT.invoke({"query": state["query"]})
    result = structured_llm.invoke(prompt_value)

    return {"route": result.route}


# 节点 2: retrieveDocuments (检索文档)
def retrieve_documents(state: AgentState, config: RunnableConfig):
    """对应 TypeScript 的 retrieveDocuments"""
    print(f"🔍 正在检索论文知识库...")
    docs = retriever.invoke(state["query"])
    return {"documents": docs}


# 节点 3: generateResponse (根据检索内容生成回答)
def generate_response(state: AgentState, config: RunnableConfig):
    """对应 TypeScript 的 generateResponse"""
    print(f"💡 正在生成回答...")

    # 格式化文档
    context = "\n\n".join([doc.page_content for doc in state["documents"]])

    # 调用大模型
    chain = RESPONSE_PROMPT | llm | StrOutputParser()
    response = chain.invoke({
        "context": context,
        "messages": state["messages"]
    })

    return {"messages": [AIMessage(content=response)]}


# 节点 4: directAnswer
def direct_answer(state: AgentState, config: RunnableConfig):
    """对应 TypeScript 的 answerQueryDirectly"""
    print(f"💬 直接回答（无需检索）...")

    # 简单的直接回答逻辑
    direct_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是汽车评分助手。如果是打招呼或闲聊，请友好回应；如果是其他问题，请告知用户你主要回答论文相关内容。"),
        ("placeholder", "{messages}")
    ])

    chain = direct_prompt | llm | StrOutputParser()
    response = chain.invoke({"messages": state["messages"]})

    return {"messages": [AIMessage(content=response)]}


# -------------------------- 9. 定义条件边 (对应 TypeScript 的 routeQuery) --------------------------
def route_query(state: AgentState) -> Literal["retrieveDocuments", "directAnswer"]:
    """对应 TypeScript 的 routeQuery"""
    if state["route"] == "retrieve":
        return "retrieveDocuments"
    else:
        return "directAnswer"


# -------------------------- 10. 构建 StateGraph (对应 TypeScript 的 builder) --------------------------
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("checkQueryType", check_query_type)
workflow.add_node("retrieveDocuments", retrieve_documents)
workflow.add_node("generateResponse", generate_response)
workflow.add_node("directAnswer", direct_answer)

# 构建边
workflow.add_edge(START, "checkQueryType")
workflow.add_conditional_edges(
    "checkQueryType",
    route_query,
    {
        "retrieveDocuments": "retrieveDocuments",
        "directAnswer": "directAnswer"
    }
)
workflow.add_edge("retrieveDocuments", "generateResponse")
workflow.add_edge("generateResponse", END)
workflow.add_edge("directAnswer", END)

# 编译图
app = workflow.compile()

# -------------------------- 11. 主程序：命令行对话 --------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚗 汽车品牌大数据评分智能助手 (LangGraph 高级版)")
    print("=" * 60)
    print("提示：直接输入问题即可，输入 'quit' 退出\n")

    while True:
        user_input = input("你: ")
        if user_input.lower() == 'quit':
            break

        # 调用 LangGraph
        output = app.invoke({
            "query": user_input,
            "messages": [HumanMessage(content=user_input)]
        })

        # 打印最后一条 AI 消息
        print(f"AI: {output['messages'][-1].content}\n")