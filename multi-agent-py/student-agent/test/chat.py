import os

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_ollama import OllamaEmbeddings, ChatOllama
import asyncio

# 环境与路径
current_dir = os.path.dirname(os.path.abspath(__file__))
vector_db_path = os.path.join(current_dir, "..", "faiss_index")
embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")
vectorstore = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
chat_ollama = ChatOllama(model="qwen2.5:7b")

intent_classifier_prompt = PromptTemplate.from_template("""
你是一个路由调度员。请判断用户的提问属于哪种类型：

1. 如果是日常打招呼、闲聊、问你是谁、要求写代码/翻译等通用任务，返回 'GENERAL'。
2. 如果是询问关于《倚天屠龙记》、武功、剧情、张无忌等具体武侠内容的，返回 'RAG'。

只返回这两个单词之一，不要解释。

用户问题：{question}
""")

rag_prompt = ChatPromptTemplate.from_template("""
你是一个严谨的武侠研究专家。请严格基于提供的【片段内容】回答，禁止引用片段之外的传闻或自行发明设定。

【片段内容】：
{context}

【研究问题】：{question}

回答要求：
1. 必须分点陈述，每一点都要引用片段中的具体情节或招式。
2. 如果片段中没有提到某种武功，绝对不要出现在答案中。
3. 严禁混淆人物关系和道具用途（如药物、内功等）。
""")

async def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12")

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

rag_chain = (
        RunnableParallel(
            {"context": compression_retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()})
        | rag_prompt
        | chat_ollama
        | StrOutputParser()
)

general_chat_chain = (
        ChatPromptTemplate.from_template("你是一个全能的 AI 助手，请回答：{question}")
        | chat_ollama
        | StrOutputParser()
)

async def route_query(input_data):

    classifier = intent_classifier_prompt | chat_ollama | StrOutputParser()
    intent = await classifier.ainvoke({"question": input_data["question"]})

    print(f"\n[路由日志] 识别意图为: {intent.strip().upper()}")

    query_text = input_data["question"]

    if "RAG" in intent.upper():
        return rag_chain.astream(query_text)
    else:
        return general_chat_chain.astream(query_text)

async def main():

    # 测试 1: 正常回答
    question = "你好，请问你是谁？你能帮我写一个 Java 的冒泡排序吗？"

    # 测试 2: 知识库回答
    # question = "张无忌在面对阿三时，除了云手还用了什么？"

    result = await route_query({"question":question})

    async for chunk in result:
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
