# -------------------------- 1. 基础配置与导入 --------------------------
import os
import pandas as pd
from dotenv import load_dotenv

# LangChain 核心组件
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 通义千问专用
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

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

# -------------------------- 3. 加载 CSV 结构化数据 --------------------------
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


# -------------------------- 4. 核心逻辑：构建向量库 --------------------------
def build_or_load_vectorstore():
    if os.path.exists(FAISS_DB_PATH):
        print("正在加载本地向量库...")
        # 加载时也用通义千问的 Embedding
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=YOUR_API_KEY
        )
        return FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    print("正在构建向量库（首次运行较慢，请耐心等待）...")

    # 1. 加载 PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"PDF 加载完成，共 {len(documents)} 页")

    # 2. 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " "]
    )
    splits = text_splitter.split_documents(documents)
    print(f"文本分块完成，共 {len(splits)} 个块")

    # 3. 构建 FAISS 向量库 (这里也统一用通义千问)
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=YOUR_API_KEY
    )

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(FAISS_DB_PATH)
    print("✅ 向量库构建完成并保存")
    return vectorstore


# -------------------------- 5. 初始化所有组件 --------------------------
# 1. 先拿到 vectorstore
vectorstore = build_or_load_vectorstore()

# 2. 再构建 retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. 设计 Prompt
system_prompt = """
你是《品牌汽车大数据评分研究》毕业论文的专业智能助手。
请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：
{question}

如果问题涉及具体车型的分数、排名，请优先结合以下数据规则回答：
- 创新力是新时代汽车品牌的首要竞争力
- 新能源车型已全面超越传统燃油车型
- 多维均衡发展是品牌持久竞争力的根本保证

如果答案不在参考资料中，请明确告知，不要编造。
"""
prompt = ChatPromptTemplate.from_template(system_prompt)

# 4. 初始化通义千问大模型 (显式传入Key)
llm = ChatTongyi(
    model="qwen-turbo",
    temperature=0.1,
    dashscope_api_key=YOUR_API_KEY
)


# 5. 辅助函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 6. 构建 RAG 链
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


# -------------------------- 6. CSV 数据查询辅助函数 --------------------------
def check_car_data(question):
    if car_df is None:
        return None

    for idx, row in car_df.iterrows():
        car_name = str(row["车型名称"])
        if car_name in question:
            return f"""
            【{car_name}】 精准数据：
            • 综合得分：{row['综合得分']}
            • 总排名：第 {int(row['排名'])} 名
            • 产品力：{row['产品力']}
            • 市场表现：{row['市场表现']}
            • 用户口碑：{row['用户口碑']}
            • 创新力：{row['创新力']}
            """

    if "排名" in question and ("前" in question or "1" in question or "2" in question or "3" in question):
        top_str = car_df.head(5)[["排名", "车型名称", "综合得分"]].to_string(index=False)
        return f"【综合排名 Top 5】：\n{top_str}"

    return None


# -------------------------- 7. 主程序：命令行对话 --------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚗 汽车品牌大数据评分智能助手")
    print("=" * 60)
    print("提示：直接输入问题即可，输入 'quit' 退出\n")

    while True:
        user_input = input("你: ")
        if user_input.lower() == 'quit':
            break

        data_info = check_car_data(user_input)

        if data_info:
            final_input = f"{user_input}\n\n补充数据：{data_info}"
            print("AI: ", end="", flush=True)
            response = rag_chain.invoke(final_input)
            print(response)
        else:
            print("AI: ", end="", flush=True)
            response = rag_chain.invoke(user_input)
            print(response)