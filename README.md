# 🔍 深研 (DeepResearch)：基于私有文献的逻辑检索引擎

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.30+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)

## 📌 项目定位
本项目是一款为“高语义密度”文献（如哲学、历史、战略研究）设计的 RAG (检索增强生成) 工具。通过重构 LCEL 架构，解决了通用大模型在垂直领域存在的上下文断裂与幻觉问题。

## ✨ 核心功能
- **LCEL 现代架构**：弃用陈旧的 Chains 模块，采用最新的 LangChain 表达式语言实现数据流高度透明化。
- **颗粒度级溯源**：实时展示检索来源（文档名 + 精准页码），并输出相关性分数（Relevance Score）。
- **交互式折叠报告**：利用 HTML5 语义化组件实现“渐进式披露”排版，降低用户认知负荷。
- **SMTP 反馈闭环**：内置 Bad Case 提报系统，自动捕获失败上下文并发送至开发者邮箱，实现数据飞轮。

## 🛠️ 技术栈
- **Frontend**: Streamlit (富文本渲染 + 状态管理)
- **Orchestration**: LangChain (LCEL + VectorStore)
- **Model**: DeepSeek-V3 (兼容 OpenAI API)
- **Vector DB**: Chroma (内存模式)
- **Embedding**: HuggingFace (all-MiniLM-L6-v2)

## 🚀 快速启动
1. 克隆仓库：`git clone https://github.com/你的用户名/DeepResearch.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 配置秘密：在 `.streamlit/secrets.toml` 中填入你的 API Key 和 SMTP 授权码。
4. 运行：`streamlit run app.py`

线上 Demo 首次运行需加载精排模型，请耐心等待 5 分钟左右
