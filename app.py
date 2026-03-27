"""
深研：基于私有文献的逻辑检索引擎
基于 Streamlit + LangChain + DeepSeek 构建的企业级 RAG 应用
支持多文档上传、向量检索、逻辑问答与用户反馈系统
"""

import re
import streamlit as st
import os
import sys
import tempfile
import traceback
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# --- 2024/2025 现代企业级架构 (LCEL) 导入 ---
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ============================================================================
# 页面配置与初始化
# ============================================================================

st.set_page_config(
    page_title="深研：基于私有文献的逻辑检索引擎",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = 0
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "processing_status" not in st.session_state:
    st.session_state.processing_status = "等待上传文档"

# ============================================================================
# 核心功能模块
# ============================================================================

def extract_text_from_pdfs(uploaded_files: List) -> Tuple[List[Document], List[Dict]]:
    """
    从上传的PDF文件中提取文本并分割
    
    Args:
        uploaded_files: 上传的PDF文件列表
        
    Returns:
        Tuple[List[Document], List[Dict]]: 文档块列表和元数据列表
    """
    documents = []
    metadatas = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # 更新进度状态
            progress = (i / len(uploaded_files))
            progress_bar.progress(progress)
            status_text.text(f"正在处理: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # 加载PDF文档
            loader = PyPDFLoader(tmp_path)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', doc.page_content)
                content = re.sub(r'\s+', ' ', content)
                doc.page_content = content.strip()
            for doc in pdf_docs:
                doc.metadata["source_document"] = uploaded_file.name
            
            # 为每个文档添加元数据
            for doc in pdf_docs:
                doc.metadata["source_document"] = uploaded_file.name
                if "page" not in doc.metadata:
                    doc.metadata["page"] = 1
            
            documents.extend(pdf_docs)
            
            # 清理临时文件
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"处理文件 {uploaded_file.name} 时出错: {str(e)}")
            continue
    
    # 文本分割
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        # 为每个分割块创建元数据
        for doc in split_docs:
            metadata = {
                "source": doc.metadata.get("source_document", "unknown"),
                "page": doc.metadata.get("page", 1),
                "chunk_id": len(metadatas)
            }
            metadatas.append(metadata)
        
        progress_bar.progress(1.0)
        status_text.text(f"处理完成！共分割 {len(split_docs)} 个文本块")
        st.success(f"成功处理 {len(uploaded_files)} 个文档，生成 {len(split_docs)} 个文本块")
    
    return split_docs if documents else [], metadatas

@st.cache_resource
def build_vector_store(_texts: List[str], _metadatas: List[Dict]) -> Chroma:
    """
    构建向量数据库并缓存
    
    Args:
        _texts: 文本列表
        _metadatas: 元数据列表
        
    Returns:
        Chroma: 向量数据库实例
    """
    try:
        # 初始化嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 构建向量数据库
        vector_store = Chroma.from_texts(
            texts=_texts,
            embedding=embeddings,
            metadatas=_metadatas,
            collection_name="deep_research_docs"
        )
        
        return vector_store
        
    except Exception as e:
        st.error(f"构建向量数据库时出错: {str(e)}")
        raise

def initialize_rag_chain(vector_store: Chroma):
    """
    初始化现代 RAG 检索链 (基于 LCEL 架构)
    """
    try:
        api_key = st.secrets["deepseek"]["api_key"]
        
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.1,
            max_tokens=2000
        )
        
        system_prompt = """
        你是一个严谨的分析师与逻辑专家。你必须将分析结果封装在 HTML 的折叠标签中，以确保报告的整洁与专业。

        ### 📝 输出规范（必须严格执行）：
        1. **核心摘要**：
           在所有折叠框之前，先用一句话总结核心结论。
        
        2. **折叠框格式**：
           每个主要章节必须使用以下格式：
           <details>
           <summary><b> [这里填入大标题，例如：📌 核心逻辑链路] </b></summary>
           <br>
           [这里填入详细的分析内容，必须包含精准引用，例如：[文献名，第 X 页]]
           <br>
           </details>
        
        3. **必填章节**：
           - <details><summary><b> 🔗 逻辑链路 </b></summary>...</details>
           - <details><summary><b> 🔹 分项深度解析 </b></summary>...</details>
           - <details><summary><b> 💡 战略建议与结论 </b></summary>...</details>          
        
        4. **强制要求**：
           - 每个 <details> 标签之间必须空两行。
           - 严禁输出任何纯文本段落（除了开头的摘要）。
            - 折叠框内的内容要保持精炼，使用列表（-）进行排版。        
        
        ### 检索到的原始文献上下文：
        {context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        
        # 定义文档格式化函数
        def format_docs(docs):
            formatted = []
            for i, doc in enumerate(docs):
                # 将文件名和页码直接标在每一段的开头
                source_name = doc.metadata.get("source", "未知文档")
                page_num = doc.metadata.get("page", "N/A")
                content = f"--- [来自文档：{source_name} | 第 {page_num} 页] ---\n{doc.page_content}"
                formatted.append(content)
            return "\n\n".join(formatted)
        
        # 构建核心生成链
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 构建包含溯源信息的并行链 (LCEL 灵魂所在)
        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "input": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)
        
        return rag_chain_with_source
        
    except Exception as e:
        st.error(f"初始化 RAG 链时出错: {str(e)}")
        return None

def get_rag_response(question: str, retrieval_chain) -> Tuple[str, List[Dict]]:
    """
    获取RAG回答和检索溯源信息
    
    Args:
        question: 用户问题
        retrieval_chain: 检索链实例
        
    Returns:
        Tuple[str, List[Dict]]: 回答内容和检索溯源信息
    """
    try:
        # 增加API调用计数
        st.session_state.api_call_count += 1
        
        # 获取回答
        response = retrieval_chain.invoke(question)
        
        # 提取回答内容
        answer = response.get("answer", "抱歉，未能生成回答。")
        
        # 提取检索到的文档和元数据
        source_docs = response.get("context", [])
        sources_info = []
        
        for i, doc in enumerate(source_docs):
            if isinstance(doc, Document):
                metadata = doc.metadata
                source_info = {
                    "text": doc.page_content,
                    "source": metadata.get("source", "未知文档"),
                    "page": metadata.get("page", 1),
                    "score": metadata.get("score", 0.0),
                    "index": i
                }
                sources_info.append(source_info)
        
        # 按相关性分数排序
        sources_info.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return answer, sources_info
        
    except Exception as e:
        st.error(f"获取回答时出错: {str(e)}")
        return "抱歉，处理您的问题时出现错误。", []

def send_feedback_email(user_question: str, ai_response: str, feedback_text: str, sources: List[Dict]) -> bool:
    """
    发送用户反馈邮件
    
    Args:
        user_question: 用户问题
        ai_response: AI回答
        feedback_text: 用户反馈
        sources: 检索溯源信息
        
    Returns:
        bool: 发送是否成功
    """
    try:
        # 从secrets获取邮件配置
        smtp_server = st.secrets["email"]["smtp_server"]
        smtp_port = int(st.secrets["email"]["smtp_port"])
        sender = st.secrets["email"]["sender"]
        password = st.secrets["email"]["password"]
        receiver = st.secrets["email"]["receiver"]
        
        # 准备邮件内容
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 格式化溯源信息
        sources_text = ""
        for source in sources[:3]:  # 只取前3个最相关的
            sources_text += f"- {source['text']}\n  来源: {source['source']} [页码: {source['page']}] [分数: {source.get('score', 0):.2f}]\n\n"
        
        # 构建邮件正文
        email_body = f"""
反馈时间：{timestamp}
用户会话ID：{st.session_state.session_id}

=== 问题与回答 ===
用户问题：{user_question}
AI原始回答：{ai_response}

=== 用户改进建议 ===
{feedback_text}

=== 检索溯源信息 ===
{sources_text}

=== 系统信息 ===
API调用次数：{st.session_state.api_call_count}
处理文档数：{len(st.session_state.uploaded_files)}
        """
        
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = "[深研反馈] 来自用户的 Bad Case 提报"
        
        msg.attach(MIMEText(email_body, 'plain'))
        
        # 发送邮件
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, password)
            server.send_message(msg)
        
        return True
        
    except Exception as e:
        st.error(f"发送邮件时出错: {str(e)}")
        return False

# ============================================================================
# 界面构建
# ============================================================================

def build_sidebar():
    """构建侧边栏"""
    with st.sidebar:
        st.title("📚 文档管理")
        
        # 文件上传
        uploaded_files = st.file_uploader(
            "上传PDF文献（支持多选）",
            type="pdf",
            accept_multiple_files=True,
            help="请上传需要分析的PDF文档，支持批量上传"
        )
        
        if uploaded_files and uploaded_files != st.session_state.uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            with st.spinner("正在处理文档..."):
                # 提取文本
                documents, metadatas = extract_text_from_pdfs(uploaded_files)
                
                if documents:
                    # 构建向量数据库
                    texts = [doc.page_content for doc in documents]
                    vector_store = build_vector_store(texts, metadatas)
                    st.session_state.vector_store = vector_store
                    st.session_state.processing_status = "文档处理完成"
        
        # 系统状态监控
        st.divider()
        st.subheader("📊 系统状态")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("API调用次数", st.session_state.api_call_count)
        with col2:
            st.metric("会话ID", st.session_state.session_id)
        
        st.info(f"处理状态: {st.session_state.processing_status}")
        
        if st.session_state.vector_store:
            collection_info = st.session_state.vector_store._collection.count()
            st.success(f"向量数据库: {collection_info} 个文本块已就绪")
        
        # 使用说明
        st.divider()
        st.subheader("💡 使用说明")
        st.info("""
        1. 上传PDF文档（支持多选）
        2. 等待文档处理完成
        3. 在下方输入问题
        4. 查看AI回答和检索溯源
        5. 如有问题可提交反馈
        """)

def build_main_content():
    """构建主内容区域"""
    st.title("🔍 深研：基于私有文献的逻辑检索引擎")
    st.markdown("基于 DeepSeek 大模型的私有文献智能问答系统")
    
    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            
            # 如果是AI回答，显示溯源信息
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("🔍 知识库检索溯源", expanded=False):
                    for source in message["sources"]:
                        score = source.get("score", 0)
                        score_text = f"**[相关性分数: {score:.2f}]**" if score > 0 else ""
                        st.markdown(f"""
                        **[{source['source']}] [页码: {source['page']}]** {score_text}
                        
                        {source['text']}
                        """)
                        st.divider()
    
    # 用户输入
    if prompt := st.chat_input("请输入关于文献的问题..."):
        # 检查向量数据库是否就绪
        if not st.session_state.vector_store:
            st.error("请先上传并处理PDF文档")
            return
            
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 显示AI回答
        with st.chat_message("assistant"):
            with st.spinner("正在检索文献并生成回答..."):
                # 初始化RAG链
                retrieval_chain = initialize_rag_chain(st.session_state.vector_store)
                
                if retrieval_chain:
                    # 获取回答
                    answer, sources = get_rag_response(prompt, retrieval_chain)
                    
                    # 流式显示回答
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in answer.split():
                        full_response += chunk + " "
                        response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    
                    response_placeholder.markdown(full_response, unsafe_allow_html=True)
                    
                    # 添加AI消息到历史
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
                else:
                    st.error("RAG链初始化失败，请检查配置")

def build_feedback_section():
    """构建反馈系统"""
    with st.expander("💬 建议反馈与 Bad Case 提报", expanded=False):
        st.markdown("如果您对回答不满意或发现错误，请告诉我们")
        
        feedback_text = st.text_area(
            "您的反馈建议",
            placeholder="请描述您遇到的问题或提供改进建议...",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_feedback = st.button("📧 提交反馈", type="secondary")
        
        if submit_feedback and feedback_text:
            if st.session_state.messages:
                # 获取最近的问题和回答
                recent_messages = st.session_state.messages[-2:] if len(st.session_state.messages) >= 2 else []
                user_question = ""
                ai_response = ""
                sources = []
                
                for msg in recent_messages:
                    if msg["role"] == "user":
                        user_question = msg["content"]
                    elif msg["role"] == "assistant":
                        ai_response = msg["content"]
                        sources = msg.get("sources", [])
                
                # 发送反馈邮件
                with st.spinner("正在发送反馈..."):
                    success = send_feedback_email(
                        user_question, 
                        ai_response, 
                        feedback_text, 
                        sources
                    )
                    
                    if success:
                        st.success("反馈已发送！感谢您的宝贵意见。")
                        st.balloons()
                    else:
                        st.error("发送失败，请检查邮件配置或稍后重试")
            else:
                st.warning("请先进行对话后再提交反馈")

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    try:
        # 构建界面
        build_sidebar()
        build_main_content()
        build_feedback_section()
        
        # 页脚信息
        st.divider()
        st.caption("© 2024 深研逻辑检索引擎 | 基于 Streamlit + LangChain + DeepSeek 构建")
        
    except Exception as e:
        st.error(f"应用运行时出错: {str(e)}")
        st.error("详细错误信息:")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()