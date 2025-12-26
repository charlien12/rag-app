import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic import hub

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="RAG Web Q&A",
    layout="wide"
)

st.title("🔗 Web-based RAG Question Answering")
st.write("Paste a website URL, index it, and ask questions using Gemini + LangChain")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --------------------------------------------------
# SIDEBAR – WEBSITE INGESTION
# --------------------------------------------------
st.sidebar.header("📥 Website Indexing")

url = st.sidebar.text_input(
    "Enter Website URL",
    placeholder="https://docs.langchain.com/oss/python/langchain/overview"
)

index_button = st.sidebar.button("Index Website")

# --------------------------------------------------
# INDEXING LOGIC
# --------------------------------------------------
if index_button and url:
    with st.spinner("Loading and indexing website..."):

        # Load website
        loader = WebBaseLoader(web_paths=[url])
        docs = loader.load()

        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(docs)

        # Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        # Vector Store
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
        vectorstore.persist()

        st.session_state.vectorstore = vectorstore

    st.sidebar.success(f"Indexed {len(texts)} chunks successfully!")

# --------------------------------------------------
# QUESTION ANSWERING UI
# --------------------------------------------------
st.header("💬 Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="What is LangChain?"
)

ask_button = st.button("Get Answer")

# --------------------------------------------------
# RAG CHAIN
# --------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if ask_button:
    if not st.session_state.vectorstore:
        st.warning("Please index a website first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):

            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

            prompt = hub.pull("rlm/rag-prompt")

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.1,
            )

            rag_chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(question)

        st.subheader("✅ Answer")
        st.write(answer)
