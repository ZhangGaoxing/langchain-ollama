# import uuid
# from langchain_ollama.llms import OllamaLLM
# from langchain_ollama.embeddings import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_retrieval_chain, create_history_aware_retriever
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.messages import AIMessage, HumanMessage

# llm = OllamaLLM(model="qwen2.5:7b")
# embeddings = OllamaEmbeddings(model="nomic-embed-text")
# db = Chroma(embedding_function=embeddings, persist_directory='./embeddings')
# retriever = db.as_retriever()

# # 创建包含聊天历史支持的检索器
# contextualize_q_system_prompt = (
#     "根据聊天历史和最新的用户问题，尽可能简洁的回答问题。"
# )
# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )

# # 创建问答链
# system_prompt = (
#     "你是徐州工业职业技术学院的专业助手，能够根据知识库中的新闻稿，尽可能简洁的回答问题。如果你不知道答案，就说不知道。{context}"
# )
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# # 管理聊天历史
# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# store = {}

# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )

# def chat_with_system(question, session_id):
#     response = conversational_rag_chain.invoke({"input": question}, config={"configurable": {"session_id": session_id}})
#     print(f"AI: {response['answer']}")
#     return response

# # 示例对话
# if __name__ == "__main__":
#     session_id = uuid.uuid4()
#     chat_history = get_session_history(session_id)

#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             break
#         response = chat_with_system(user_input, session_id)

#     print("\n聊天历史:")
#     for message in chat_history.messages:
#         if isinstance(message, AIMessage):
#             prefix = "AI"
#         else:
#             prefix = "User"
#         print(f"{prefix}: {message.content}\n")

import uuid
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from psycopg import Connection
from langgraph.checkpoint.postgres import PostgresSaver

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

DB_URI = "postgresql://postgres:%40Passw0rd@localhost:5432/llm"

llm = ChatOllama(model="qwen2.5:1.5b")
# checkpointer = MemorySaver()
conn = Connection.connect(DB_URI)
checkpointer = PostgresSaver(conn)
checkpointer.setup()

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


graph = graph_builder.compile(checkpointer=checkpointer)

# qa_async_png = graph.get_graph().draw_mermaid_png()
# with open("chatbot.png", "wb") as f:
#     f.write(qa_async_png)

def stream_graph_updates(user_input: str, config: dict):
    events = graph.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values")
    for event in events:
        last_event = event
    print("AI: ", last_event["messages"][-1].content)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        stream_graph_updates(user_input, config)

    print("\nHistory: ")
    # for message in graph.get_state(config).values["messages"]:
    for message in checkpointer.get(config)["channel_values"]["messages"]:
        if isinstance(message, AIMessage):
            prefix = "AI"
        else:
            prefix = "User"
        print(f"{prefix}: {message.content}")

    conn.close()
