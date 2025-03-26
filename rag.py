from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model = OllamaLLM(model="qwen2.5:0.5b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(embedding_function=embeddings, persist_directory='./embeddings')
retriever=db.as_retriever(search_kwargs={"k": 3})

template = """你是徐州工业职业技术学院的专业助手，能够根据知识库中的新闻稿回答问题：
{context}
问题：{question}
回答应简洁且准确，避免编造信息。
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

def chat_with_system(question):
    response = chain.invoke(question)
    print(f"AI: {response}")
    return response

# 示例对话
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chat_with_system(user_input)
