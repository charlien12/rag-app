#https://docs.langchain.com/oss/python/integrations/document_loaders/web_base
#https://docs.langchain.com/oss/python/integrations/text_embedding/google_generative_ai
# --- Setup & loading ---
from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(web_paths=[
    "https://docs.langchain.com/oss/python/langchain/overview"
])
docs = loader.load()

# --- Chunking ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)
print(f"Chunks: {len(texts)}")

# --- Embeddings + Vector store (indexing only) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# Prefer the env var GOOGLE_API_KEY or pass google_api_key explicitly
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("API_KEY")  # <-- correct param name
)

# Persist is optional but recommended
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="chroma_db"
)
vectorstore.persist()

# Debug peek (optional)
print(vectorstore._collection.get())

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- Prompt from Hub ---
from langchain_classic import hub
prompt = hub.pull("rlm/rag-prompt")  # returns a ChatPromptTemplate

# --- LCEL pieces ---
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Use a GENERATION model for answering
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",        # or "gemini-2.0-flash"/"gemini-2.5-flash" if available
    google_api_key=os.getenv("API_KEY"),
    temperature=0.1,
)

# --- RAG chain (retriever -> format -> prompt -> LLM -> text) ---
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
print("ask the question")
ask_prompt=input().strip()
# --- Run ---
answer = rag_chain.invoke(ask_prompt)
print("\n=== Answer ===\n", answer)
