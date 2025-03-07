from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma

documents_path = './documents'
embeddings_path = './embeddings'
text_loader_kwargs={'encoding': 'utf-8'}

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    metadata["author"] = record.get("author")
    metadata["date"] = record.get("date")
    return metadata

documents = JSONLoader(file_path=f'{documents_path}/articles.jsonl', jq_schema='.', content_key='content', metadata_func=metadata_func, json_lines=True).load()
# documents = DirectoryLoader(documents_path, glob='**/*.jsonl', show_progress=True, use_multithreading=True, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs).load()
# text_splitter = RecursiveCharacterTextSplitter(
#     separators=[
#         "\n\n",
#         "\n",
#         " ",
#         ".",
#         ",",
#         "\u200B",  # Zero-width space
#         "\uff0c",  # Fullwidth comma
#         "\u3001",  # Ideographic comma
#         "\uff0e",  # Fullwidth full stop
#         "\u3002",  # Ideographic full stop
#         "",
#     ], chunk_size=1000, chunk_overlap=20, add_start_index=True)
# split_docs = text_splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(documents, embeddings, persist_directory=embeddings_path)
