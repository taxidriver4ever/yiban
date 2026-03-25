import os

from langchain_classic.retrievers import ContextualCompressionRetriever
from mcp.server.fastmcp import FastMCP
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_ollama import OllamaEmbeddings, ChatOllama

mcp = FastMCP("知识引擎", log_level="ERROR")
# 名字你也可以自己取

current_dir = os.path.dirname(os.path.abspath(__file__))
vector_db_path = os.path.join(current_dir, "..", "faiss_index")
embeddings = OllamaEmbeddings(model="qwen3-embedding:4b") # 换模型改这里
vectorstore = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
chat_ollama = ChatOllama(model="qwen2.5:7b") # 换模型改这里

compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12") # 换模型改这里
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

async def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@mcp.tool()
async def query_specific_details(question: str) -> str:
    """
    【精密细节考据专家】
    专门用于检索知识库中极其具体的微观事实。
    适用场景：
    1. 具体的招式名称、公式数值、特定角色的台词。
    2. 物品的成分、具体的地理位置、历史事件的精确时间点。
    3. 任何需要“翻书”确认的硬核细节。
    当你需要提供证据支持，或者回答“是什么”、“在哪”、“具体如何”时使用。

    Args:
        question: 需要考据的具体细节问题。
    """
    # 提示词可以帮我改一下

    rag_prompt = ChatPromptTemplate.from_template("""
        你是一个严谨的资料考据专家。请完全基于【参考片段】回答，不要加入任何片段之外的固有印象。
        【参考片段】：{context}
        【考据问题】：{question}
        """)
    # 提示词可以帮我改一下

    rag_chain = (
            RunnableParallel({
                "context": compression_retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })
            | rag_prompt
            | chat_ollama
            | StrOutputParser()
    )

    response = await rag_chain.ainvoke(question)
    return f"[专家解析]\n{response}"


@mcp.tool()
async def summarize_concept_or_arc(topic: str) -> str:
    """
    【宏观逻辑总结大师】
    专注于对知识库中的人物关系、剧情主线或复杂概念进行纵向梳理。
    适用场景：
    1. 人物或事物的全周期演变过程（从开始到结束）。
    2. 多个实体之间的复杂因果关系分析。
    3. 针对某个主题（如：武学体系、历史背景）的综述。
    当用户问“为什么”、“总结一下”、“评价”、“关系如何”时优先调用。

    Args:
        topic: 需要宏观综述的主题或概念。
    """
    # 提示词可以帮我改一下

    summary_retriever = vectorstore.as_retriever(search_kwargs={"k": 20}) # 可以比20更高
    docs = await summary_retriever.ainvoke(topic)
    context = await format_docs(docs)

    prompt = f"""
        你是一个擅长提炼核心逻辑的分析专家。请根据以下提供的多条背景资料，
        对“{topic}”进行条理清晰、深入浅出的宏观总结。

        【背景资料】：
        {context}
        """
    # 提示词可以帮我改一下

    response = await chat_ollama.ainvoke(prompt)
    return response.content

if __name__ == "__main__":
    mcp.run(transport="stdio")