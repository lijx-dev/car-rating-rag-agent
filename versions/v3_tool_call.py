# -------------------------- 1. 基础配置与导入 --------------------------
import os
import json
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
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document

# 通义千问
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

# LangGraph
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# -------------------------- 2. 初始化环境 --------------------------
load_dotenv()
base_dir = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(base_dir, "data", "品牌汽车大数据评分研究_毕业论文.pdf")
CSV_PATH = os.path.join(base_dir, "data", "综合评分结果_AHP熵权.csv")
FAISS_DB_PATH = os.path.join(base_dir, "faiss_index")

YOUR_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not YOUR_API_KEY:
    raise ValueError("请在.env文件中配置DASHSCOPE_API_KEY！")

# -------------------------- 3. 定义车型查询工具 --------------------------
class CarQueryInput(BaseModel):
    car_name: str = Field(..., description="要查询的车型名称，比如'特斯拉model3'、'比亚迪汉'")
    query_type: Literal["basic", "full", "rank", "compare"] = Field(
        default="basic",
        description="查询类型：basic=基础得分排名，full=全量数据，rank=排名查询，compare=多车型对比"
    )
    compare_cars: list[str] = Field(default_factory=list, description="对比车型列表")

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

def query_car_rating(car_name: str, query_type: str = "basic", compare_cars: list = None):
    if car_df is None:
        return "暂无车型数据"
    compare_cars = compare_cars or []
    all_query_cars = [car_name] + compare_cars
    result_df = car_df.copy()
    filter_mask = pd.Series([False] * len(result_df))
    for car in all_query_cars:
        filter_mask = filter_mask | result_df["车型名称"].str.contains(car, na=False, case=False)
    result_df = result_df[filter_mask]
    if result_df.empty:
        return f"未查询到{all_query_cars}的相关数据"
    if query_type == "basic":
        return result_df[["车型名称", "综合得分", "排名", "产品力", "市场表现", "用户口碑", "创新力"]].to_string(index=False)
    elif query_type == "full":
        return result_df.to_string(index=False)
    elif query_type == "rank":
        top5 = car_df.head(5)[["排名", "车型名称", "综合得分"]].to_string(index=False)
        return f"综合排名Top5：\n{top5}\n\n你查询的车型数据：\n{result_df[['车型名称', '综合得分', '排名']].to_string(index=False)}"
    elif query_type == "compare":
        return f"多车型对比数据：\n{result_df[['车型名称', '综合得分', '排名', '产品力', '创新力']].to_string(index=False)}"
    else:
        return result_df[["车型名称", "综合得分", "排名"]].to_string(index=False)

car_query_tool = StructuredTool.from_function(
    func=query_car_rating,
    name="query_car_rating",
    description="汽车评分专用工具，用于查询车型的精准评分、排名、各维度得分、多车型对比",
    args_schema=CarQueryInput,
    return_direct=False
)
tools = [car_query_tool]

# -------------------------- 4. 定义 Agent 状态 --------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    route: Literal["retrieve", "direct", "tool_call"]
    documents: list
    tool_result: str
    context: str

# -------------------------- 5. 结构化分块 + 向量检索 --------------------------
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=YOUR_API_KEY
)

def load_and_split_pdf(pdf_path):
    print("📄 正在加载并结构化拆分毕业论文...")
    loader = PyPDFLoader(pdf_path)
    raw_documents = loader.load()
    print(f"✅ PDF加载完成，共 {len(raw_documents)} 页")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", "。", ".", " "],
        is_separator_regex=False
    )
    split_docs = text_splitter.split_documents(raw_documents)
    for doc in split_docs:
        page_num = doc.metadata.get("page", 0) + 1
        doc.metadata["source_info"] = f"【第{page_num}页】"
    print(f"✅ 结构化分块完成，共生成 {len(split_docs)} 个文本块")
    return split_docs

def build_or_load_retriever():
    if os.path.exists(FAISS_DB_PATH):
        print("📦 正在加载本地向量库...")
        vectorstore = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        split_docs = []
        try:
            with open(os.path.join(FAISS_DB_PATH, "split_docs.json"), "r", encoding="utf-8") as f:
                split_docs_json = json.load(f)
                split_docs = [Document(**doc) for doc in split_docs_json]
        except:
            split_docs = []
    else:
        split_docs = load_and_split_pdf(PDF_PATH)
        print("🔨 正在构建FAISS向量库...")
        vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
        vectorstore.save_local(FAISS_DB_PATH)
        split_docs_json = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in split_docs]
        with open(os.path.join(FAISS_DB_PATH, "split_docs.json"), "w", encoding="utf-8") as f:
            json.dump(split_docs_json, f, ensure_ascii=False, indent=2)

    def stable_retrieve(query: str):
        docs = vectorstore.similarity_search(query, k=4)
        print(f"🔍 检索完成，找到 {len(docs)} 条相关内容")
        return docs
    print("✅ 检索器构建完成")
    return stable_retrieve

retrieve_func = build_or_load_retriever()

# -------------------------- 6. 初始化大模型 --------------------------
llm = ChatTongyi(
    model="qwen-turbo",
    temperature=0.1,
    dashscope_api_key=YOUR_API_KEY
)
llm_with_tools = llm.bind_tools(tools)

# -------------------------- 7. Prompt --------------------------
RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是《品牌汽车大数据评分研究》毕业论文的专业智能助手。
请严格基于提供的参考内容回答，禁止编造数据和内容。
【参考内容】
{context}
【核心结论】
- 创新力是新时代汽车品牌的首要竞争力
- 新能源车型已全面超越传统燃油车型的综合竞争力
- 多维均衡发展是品牌持久竞争力的根本保证
回答要求：专业、严谨、简洁。
"""),
    ("placeholder", "{messages}"),
])

# -------------------------- 8. LangGraph 节点 --------------------------
def check_query_type(state: AgentState, config: RunnableConfig):
    print(f"\n🤔 正在判断问题类型: {state['query']}")
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是智能路由助手，只返回JSON，不要多余文字。
规则：
1. 问车型分数/排名/对比 → {"route":"tool_call"}
2. 问论文内容/方法/模型 → {"route":"retrieve"}
3. 闲聊 → {"route":"direct"}
用户问题：{query}
"""),
    ])
    chain = router_prompt | llm | StrOutputParser()
    response_str = chain.invoke({"query": state["query"]})
    try:
        clean_str = response_str.strip().strip("```json").strip("```").strip()
        result = json.loads(clean_str)
        route = result.get("route", "direct")
    except:
        route = "direct"
    print(f"✅ 路由结果: {route}")
    return {"route": route}

def retrieve_documents(state: AgentState, config: RunnableConfig):
    print(f"🔍 正在检索论文知识库...")
    docs = retrieve_func(state["query"])
    context = ""
    for idx, doc in enumerate(docs):
        source_info = doc.metadata.get("source_info", "未知来源")
        context += f"[{idx+1}] {source_info}\n{doc.page_content}\n\n"
    return {"documents": docs, "context": context}

def call_car_tool(state: AgentState, config: RunnableConfig):
    print(f"🛠️  正在调用车型评分工具...")
    tool_call_response = llm_with_tools.invoke([HumanMessage(content=state["query"])])
    for tool_call in tool_call_response.tool_calls:
        selected_tool = {t.name:t for t in tools}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        context = f"车型精准数据：\n{tool_output}"
        return {"tool_result": tool_output, "context": context}
    return {"tool_result": "未查询到相关车型数据", "context": "未查询到相关车型数据"}

def generate_response(state: AgentState, config: RunnableConfig):
    return {"context": state.get("context", "")}

def direct_answer(state: AgentState, config: RunnableConfig):
    return {"context": "闲聊场景，无需检索"}

# -------------------------- 9. 构建工作流 --------------------------
workflow = StateGraph(AgentState)
workflow.add_node("checkQueryType", check_query_type)
workflow.add_node("retrieveDocuments", retrieve_documents)
workflow.add_node("callCarTool", call_car_tool)
workflow.add_node("generateResponse", generate_response)
workflow.add_node("directAnswer", direct_answer)

workflow.add_edge(START, "checkQueryType")
workflow.add_conditional_edges(
    "checkQueryType",
    lambda s:s["route"],
    {
        "retrieve": "retrieveDocuments",
        "tool_call": "callCarTool",
        "direct": "directAnswer"
    }
)
workflow.add_edge("retrieveDocuments", "generateResponse")
workflow.add_edge("callCarTool", "generateResponse")
workflow.add_edge("generateResponse", END)
workflow.add_edge("directAnswer", END)
app = workflow.compile()

# -------------------------- 10. 主程序 --------------------------
if __name__ == "__main__":
    print("\n" + "="*75)
    print("🚗 汽车品牌大数据评分智能助手 (稳定兼容版)")
    print("="*75)
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'quit':
            print("\n👋 感谢使用，再见！")
            break
        final_state = app.invoke({
            "query": user_input,
            "messages": [HumanMessage(content=user_input)]
        })
        print("AI: ", end="", flush=True)
        context = final_state.get("context", "")
        route = final_state.get("route", "direct")

        if route == "direct":
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是友好的汽车评分助手，闲聊请正常回复。"),
                ("human", user_input)
            ])
        else:
            prompt = RESPONSE_PROMPT

        chain = prompt | llm | StrOutputParser()
        for chunk in chain.stream({"context": context, "messages": [HumanMessage(content=user_input)]}):
            print(chunk, end="", flush=True)
        print("\n")