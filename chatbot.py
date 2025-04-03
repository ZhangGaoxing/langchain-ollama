import os
import uuid
from typing_extensions import TypedDict
from typing import List
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.messages import AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

os.environ['TAVILY_API_KEY'] = "tvly-dev-xxx"

llm = ChatOllama(model="qwen2.5:1.5b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(embedding_function=embeddings, persist_directory='./embeddings')
retriever=db.as_retriever(search_kwargs={"k": 3})
web_search_tool = TavilySearchResults(k=3)

### Implement the Router
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \n
    You are an expert at routing a user question to a vectorstore or web search. Use the vectorstore 
    for questions on Xuzhou College of Industrial Technology. You do not need to be stringent with 
    the keywords in the question related to these topics. Otherwise, use web-search. \n
    Give a binary choice 'web_search' or 'vectorstore' based on the question. 
    Return a JSON with a single key 'datasource' and no premable or explaination. Question to route: {question} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question"],
)
question_router = prompt | llm | JsonOutputParser()

### Implement the Generate Chain
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \n
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer 
    the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and 
    keep the answer concise. \n
    <|eot_id|><|start_header_id|>user<|end_header_id|> \n
    Question: {question} \n
    Context: {documents} \n
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "documents"],
)

rag_chain = prompt | llm | StrOutputParser()

### Implement the Retrieval Grader
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \n
    You are a grader assessing relevance of a retrieved document to a user question. If the document 
    contains keywords related to the user question, grade it as relevant. It does not need to be a stringent 
    test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. 
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination. \n
    <|eot_id|><|start_header_id|>user<|end_header_id|> \n
    Here is the retrieved document: \n {document} \n
    Here is the user question: {question} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)
retrieval_grader = prompt | llm | JsonOutputParser()

### Implement the hallucination grader
prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> \n
    You are a grader assessing whether an answer is grounded in a set of facts. \n
    Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in a set of facts. 
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. \n
    <|eot_id|><|start_header_id|>user<|end_header_id|> \n
    Here are the facts: \n {documents} \n
    Here is the answer: {generation} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)
hallucination_grader = prompt | llm | JsonOutputParser()

### Implement the Answer Grader
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \n
    You are a grader assessing whether an answer is useful to resolve a question. \n
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. 
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. \n
    <|eot_id|><|start_header_id|>user<|end_header_id|> \n
    Here is the answer: \n {generation} \n
    Here is the question: {question} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)
answer_grader = prompt | llm | JsonOutputParser()

class GraphState(TypedDict):
    question : str
    generation : str
    web_search : str
    documents : List[str]

def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents from vectorstore

    Args:
        state (GraphState): The current graph state

    Returns:
        state (GraphState): New key added to state, documents, that contains retrieved documents
    """

    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state: GraphState) -> GraphState:
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (GraphState): The current graph state

    Returns:
        state (GraphState): New key added to state, generation, that contains LLM generation
    """

    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"documents": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state: GraphState) -> GraphState:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (GraphState): The current graph state

    Returns:
        state (GraphState): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state: GraphState) -> GraphState:
    """
    Web search based based on the question

    Args:
        state (GraphState): The current graph state

    Returns:
        state (GraphState): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

def route_question(state: GraphState) -> str:
    """
    Route question to web search or RAG.

    Args:
        state (GraphState): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source['datasource'] == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    
def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (GraphState): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (GraphState): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search) # web search
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("grade_documents", grade_documents) # grade documents
workflow.add_node("generate", generate) # generatae

workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

app = workflow.compile(checkpointer=MemorySaver())

def stream_graph_updates(user_input: str, config: dict):
    events = app.stream({"question": user_input, "generation": "", "web_search": "no", "documents": []}, config, stream_mode="values")
    for event in events:
        last_event = event
    print("AI: ", last_event["generation"])

if __name__ == "__main__":
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        stream_graph_updates(user_input, config)

    print("\nHistory: ")
    for message in app.get_state(config).values["messages"]:
        if isinstance(message, AIMessage):
            prefix = "AI"
        else:
            prefix = "User"
        print(f"{prefix}: {message.content}")
