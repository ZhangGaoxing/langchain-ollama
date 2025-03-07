from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

prompt_template = """
    你是徐州工业职业技术学院的专业助手，能够根据知识库中的新闻稿，尽可能简洁的回答问题。如果你不知道答案，就说不知道\n
    {context}"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context"])

#model = OllamaLLM(model="qwen2.5:7b")
model = OllamaLLM(model="deepseek-r1:1.5b")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(embedding_function=embeddings, persist_directory='./embeddings')
qa = RetrievalQA.from_llm(llm=model, retriever=db.as_retriever(), prompt=prompt)

def chat_with_system(question):
    response = qa.invoke({"query": question})
    print(f"AI: {response['result']}")
    return response

# 示例对话
if __name__ == "__main__":
    qa.invoke({"query": "2025年徐州工业职业技术学院发生了什么事情"})

    for chunk in qa.stream({"query": "2025年徐州工业职业技术学院发生了什么事情"}):
        print(chunk, end="", flush=True)
