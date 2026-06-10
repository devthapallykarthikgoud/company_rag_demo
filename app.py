
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

db = FAISS.load_local(
    "vectorstore",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(model="gpt-4o-mini")

while True:
    q = input("Question: ")
    if q.lower() == "exit":
        break

    docs = db.similarity_search(q, k=3)
    context = "\n".join(d.page_content for d in docs)
    answer = llm.invoke(f"Context:\n{context}\n\nQuestion:{q}")
    print(answer.content)
