# 🚗 汽车品牌大数据评分RAG智能体
> 基于LangChain + LangGraph + Streamlit 的生产级RAG对话智能体，作为AI产品经理/AI工程师作品集项目。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

## 🔍 项目概览
本项目是本科毕业论文《品牌汽车大数据评分研究》的产品化落地成果，面向购车消费者、汽车行业分析师、学术研究者打造一站式 AI 对话助手。
融合AHP - 熵权法组合赋权模型、多源汽车数据与LangChain+LangGraph RAG 智能体，解决购车信息不对称、论文检索繁琐、行业数据分析效率低三大核心痛点。
独立完成全流程：需求文档 → 原型设计 → 代码开发 → 在线部署 → 产品落地


## ✨ 核心亮点
学术成果产品化：将统计学毕业论文转化为可交互 AI 产品，体现研究 + 落地能力
数据分析师硬核能力：10 万 + 条数据处理、AHP - 熵权法建模、数据可视化、多源数据融合
AI 产品全栈能力：RAG 智能体、智能路由、工具调用、流式输出、对话交互设计
完整作品集闭环：PRD 需求文档 + 高保真原型 + 可运行代码 + 在线 Demo
垂直领域落地：汽车行业垂直场景，贴合 AI 应用落地真实需求

## 🛠 技术栈
模块	  技术 / 工具
交互前端	Streamlit
AI 核心	LangChain、LangGraph、通义千问 (Qwen)
向量存储	FAISS
数据处理	Python、Pandas、NumPy
文档解析	PyPDF
部署平台	Streamlit Community Cloud
模型算法	AHP - 熵权法、混合检索、重排序


## ✨ 项目功能
1. 📊 汽车数据可视化看板
Top10 车型综合排名实时展示
综合得分柱状图对比
产品力 / 创新力 / 市场表现 / 用户口碑四大维度得分可视化
2. 💬 智能对话交互
车型评分 / 排名精准查询（支持特斯拉、比亚迪、理想等 24 款主流车型）
汽车行业知识智能问答（购车技巧、车型对比、行业趋势）
流式打字机输出，提升交互体验
对话历史持久化记录
3. 🔍 RAG 行业知识库检索
基于公开汽车评测、行业报告构建向量库
相似度检索 + 来源标注（文档名称）
支持自定义检索 TopK 参数
4. 🧠 智能路由系统
意图自动识别：车型数据→工具调用 / 行业问题→RAG 检索 / 闲聊→友好回复
结构化数据精准查询，无编造、无 hallucination

## 📄 产品需求文档（核心摘要）
产品定位：汽车垂直领域 AI 购车助手 + 行业知识智能检索工具
目标用户：购车消费者、汽车行业从业者、汽车爱好者
核心指标：数据准确率 100%、检索精度≥90%、系统响应≤5 秒
业务规则：基于 AHP - 熵权法评分体系，严格遵循模型计算逻辑
---

## 🚀 快速部署
### 1. 克隆项目
```bash
git clone https://github.com/wembyinspurs/car-rating-rag-agent.git
cd car-rating-rag-agent
```

### 2. 安装依赖
pip install -r requirements.txt

### 3.配置环境变量：复制 .env.example 为 .env，填入你的 API Key：
DASHSCOPE_API_KEY=your-api-key-here

### 4，运行项目
streamlit run app.py
然后打开浏览器访问：http://localhost:8501

## 📸 项目截图

### 1. 主界面概览
![主界面](./docs/prototype/main_interface.png)

### 2. 数据可视化看板
![数据看板](./docs/prototype/data_dashboard.png)

### 3. 智能对话交互
![对话示例1](./docs/prototype/chat_1.png)
![对话示例2](./docs/prototype/chat_2.png)
![对话示例3](./docs/prototype/chat_3.png)

## 📄 产品文档
- [完整产品需求文档 PRD.md](sslocal://flow/file_open?url=.%2Fdocs%2FPRD.md&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)

## 项目结构

car-rating-rag-agent/
├── .idea/
├── data/
├── docs/
│   ├── PRD.md
│   └── prototype/
│       ├── main_interface.png
│       ├── data_dashboard.png
│       ├── chat_1.png
│       ├── chat_2.png
│       └── chat_3.png
├── versions/
│   ├── v1_basic_rag.py
│   ├── v2_langgraph_agent.py
│   ├── v3_tool_call.py
│   ├── v4_advance_rag.py
│   └── main.py
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── app.py
└── requirements.txt

### 作者
Junxian Li
1564536767@qq.com
