# 🚗 汽车品牌大数据评分RAG智能体
> 基于LangChain + LangGraph + Streamlit 的生产级RAG对话智能体，作为AI产品经理/AI工程师作品集项目。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

## 🔍 项目概览
本项目是本科毕业论文《品牌汽车大数据评分研究》的产品化落地成果，面向购车消费者、汽车行业分析师、学术研究者打造一站式 AI 对话助手。
融合AHP - 熵权法组合赋权模型、多源汽车数据与LangChain+LangGraph RAG 智能体，解决购车信息不对称、论文检索繁琐、行业数据分析效率低三大核心痛点。
无实习经历・独立完成全流程：需求文档 → 原型设计 → 代码开发 → 在线部署 → 产品落地


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
- 🤖 **智能路由**：基于LangGraph的工作流，自动识别问题类型
- 🔍 **混合检索**：FAISS向量检索 + BM25关键词检索
- 💬 **可视化界面**：Streamlit专业Web界面，支持流式输出
- 📊 **数据看板**：Top10车型数据可视化，交互式图表
- ⚙️ **配置管理**：侧边栏设置，支持模型切换、参数调整

## 🚀 快速开始
### 1. 克隆仓库
```bash
git clone https://github.com/wembyinspurs/car-rating-rag-agent.git
cd car-rating-rag-agent
###2. 安装依赖
pip install -r requirements.txt
###3.配置环境变量：复制 .env.example 为 .env，填入你的 API Key：
DASHSCOPE_API_KEY=your-api-key-here
###4，运行项目
streamlit run app.py
然后打开浏览器访问：http://localhost:8501

## 📸 项目截图
### 主界面
<img width="3184" height="1736" alt="image" src="https://github.com/user-attachments/assets/10d50df0-4ade-4559-8af4-bf5f29e3d812" />

### 数据看板
<img width="2488" height="1176" alt="image" src="https://github.com/user-attachments/assets/8bcc28dc-de3c-4ea8-8701-9080fa0b88d7" />


### 智能对话
<img width="2480" height="1392" alt="image" src="https://github.com/user-attachments/assets/42ab76dc-eab0-4b28-ab8a-989bf820ada0" />

<img width="2408" height="1400" alt="image" src="https://github.com/user-attachments/assets/f7e25a38-fa47-4895-90a6-de05ca996550" />

<img width="2530" height="1168" alt="image" src="https://github.com/user-attachments/assets/24b0eaa1-8a0e-4d5a-9f67-b67cbcc1ddd8" />


car-rating-rag-agent/
├── app.py              # Streamlit Web界面（核心）
├── data/               # 示例数据（车型评分CSV）
├── docs/               # 项目截图
├── requirements.txt    # 依赖文件
├── .env.example        # 环境变量模板
├── .gitignore          # Git忽略文件
├── LICENSE             # 开源许可证
└── README.md           # 项目说明

###作者
Junxian Li
1564536767@qq.com
