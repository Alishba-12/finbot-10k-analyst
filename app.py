# app.py - FinBot: 10-K Analyst RAG Chatbot

import streamlit as st
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from edgar import Company, set_identity

# --- Page Config ---
st.set_page_config(
    page_title="FinBot | 10-K Analyst",
    page_icon="📊",
    layout="wide"
)

# --- Constants ---
PERSIST_DIR = "./chroma_db_finbot"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# --- Setup SEC Identity ---
try:
    identity = st.secrets["SEC_IDENTITY"]
except:
    identity = "your_email@example.com"  # Replace with your email!

set_identity(identity)

# --- Caching ---
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner="Building vector store (this may take a minute)...")
def build_vector_store(ticker: str):
    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        latest = filings.latest()
        
        if not latest:
            st.error(f"No 10-K filing found for {ticker}")
            return None
        
        filing_text = latest.text()
        
        docs = [Document(
            page_content=filing_text,
            metadata={"source": f"{ticker}_10-K", "filing_date": str(latest.filing_date)}
        )]
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=1024,
            chunk_overlap=128,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        
        embeddings = load_embeddings()
        
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        return vectorstore
        
    except Exception as e:
        st.error(f"Error building vector store for {ticker}: {e}")
        return None

def get_rag_response(query: str, vectorstore, top_k: int = 5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant information found.", []
    
    context_parts = []
    for i, doc in enumerate(docs):
        context_parts.append(f"[Source {i+1}]\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    answer = f"""**FinBot Answer:**
Based on the retrieved sections from the 10-K:

{context[:2000]}...

*Note: Retrieval preview only. LLM integration coming next.*
"""
    return answer, docs

# --- UI ---
st.title("📊 FinBot | 10-K Analyst")
st.caption("Ask questions about any public company's annual report (10-K filing)")

with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.text_input("Company Ticker", value="AAPL").upper()
    top_k = st.slider("Chunks to retrieve", 1, 10, 5)
    st.divider()
    
    if st.button("🔄 Build Vector Store", use_container_width=True):
        st.session_state.vectorstore = build_vector_store(ticker)
        st.session_state.ticker = ticker
        st.success(f"Vector store ready for {ticker}!")
    
    st.divider()
    st.markdown("### 📚 About")
    st.markdown("RAG-powered Q&A over SEC 10-K filings.")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the 10-K filing..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
            st.warning("Please click **'Build Vector Store'** in the sidebar first.")
        else:
            with st.spinner("Searching the filing..."):
                answer, sources = get_rag_response(prompt, st.session_state.vectorstore, top_k)
                st.markdown(answer)
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(src.page_content[:500] + "...")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

st.divider()
st.caption("Built with Streamlit, LangChain, ChromaDB, and edgartools")