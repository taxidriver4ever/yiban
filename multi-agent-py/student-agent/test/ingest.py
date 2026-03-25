import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# 配置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data")
vector_db_path = os.path.join(current_dir, "..", "faiss_index")


def ingest_docs():
    # 1. 定义不同后缀对应的加载器
    print(f"正在扫描 {data_path} 下的多种格式文档...")

    loaders = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".pdf": PyPDFLoader
    }

    documents = []
    # 2. 遍历文件夹，根据后缀选择加载器
    for file in os.listdir(data_path):
        # 生成完整文件路径
        file_path = os.path.join(data_path, file)
        ext = os.path.splitext(file)[1].lower()
        if ext in loaders:
            print(f"正在加载: {file}")
            try:
                loader = loaders[ext](file_path)
                # 对于某些 Loader 可能需要特定编码，PDF 则不需要
                if ext == ".txt" or ext == ".md":
                    loader.encoding = 'utf-8'
                documents.extend(loader.load())
            except Exception as e:
                print(f"加载 {file} 失败: {e}")

    print(f"✅ 总计加载了 {len(documents)} 个文档对象。")

    # 3. 递归切分（PDF 和 TXT 此时都已变成统一的 Document 对象）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )

    docs = text_splitter.split_documents(documents)

    # 4. 初始化 Embedding 并保存
    embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(vector_db_path)
    print("✅ 混合格式知识库已更新！")


if __name__ == "__main__":
    ingest_docs()