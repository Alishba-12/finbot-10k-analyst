# app.py - FinBot: 10-K Analyst RAG Chatbot

import streamlit as st
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
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
LLM_MODEL = "openai/gpt-oss-120b"

COMPANIES = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google (Alphabet)": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Nvidia": "NVDA",
    "Meta (Facebook)": "META",
    "Netflix": "NFLX",
    "JPMorgan Chase": "JPM",
    "Visa": "V",
    "Walmart": "WMT",
    "Coca-Cola": "KO",
}

# --- SEC Identity ---
try:
    identity = st.secrets["SEC_IDENTITY"]
except Exception:
    identity = "kalishbakhan456@gmail.com"

set_identity(identity)

# --- Cached Resources ---
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner="Loading LLM...")
def load_llm():
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"]
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
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=1024,
            chunk_overlap=128,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        embeddings = load_embeddings()
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
    except Exception as e:
        st.error(f"Error building vector store for {ticker}: {e}")
        return None

# --- RAG Prompt ---
PROMPT = ChatPromptTemplate.from_template("""You are a financial analyst assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: "I could not find this in the filing."
Cite the source numbers you used (e.g. [Source 2]).

Context:
{context}

Question: {question}

Answer:""")

def get_rag_response(query, vectorstore, llm, top_k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found.", []
    context = "\n\n".join(
        f"[Source {i+1}]\n{d.page_content}" for i, d in enumerate(docs)
    )
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    return answer, docs

# --- UI ---
st.title("📊 FinBot | 10-K Analyst")
st.caption("Ask questions about any public company's annual report (10-K filing)")

with st.sidebar:
    st.header("⚙️ Configuration")

    company_name = st.selectbox(
        "Choose a Company",
        options=list(COMPANIES.keys()),
        index=0
    )
    ticker = COMPANIES[company_name]
    st.caption(f"Selected ticker: **{ticker}**")

    top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)

    st.divider()

    if st.button("🔄 Build Vector Store", use_container_width=True):
        st.session_state.vectorstore = build_vector_store(ticker)
        st.session_state.ticker = ticker
        st.session_state.company_name = company_name
        if st.session_state.vectorstore is not None:
            st.success(f"✅ Vector store ready for {company_name} ({ticker})!")

    if "company_name" in st.session_state:
        st.info(f"📂 Currently loaded: **{st.session_state.company_name}**")

    st.divider()
    st.markdown("### 📚 About")
    st.markdown("""
    FinBot uses **RAG** to answer questions about 10-K filings.

    **How to use:**
    1. Pick a company from the dropdown
    2. Click **Build Vector Store**
    3. Wait ~1 minute
    4. Ask questions in the chat
    """)

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
            st.warning("⚠️ Please click **'Build Vector Store'** in the sidebar first.")
        else:
            with st.spinner("Searching and generating answer..."):
                answer, sources = get_rag_response(
                    prompt,
                    st.session_state.vectorstore,
                    load_llm(),
                    top_k
                )
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
st.caption("Built with Streamlit, LangChain, ChromaDB, Groq, and edgartools")