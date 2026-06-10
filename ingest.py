
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

docs = TextLoader("data/company_info.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
db = FAISS.from_documents(chunks, OpenAIEmbeddings())
db.save_local("vectorstore")
print("Done")
