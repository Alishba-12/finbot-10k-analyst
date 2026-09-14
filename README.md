# 📊 FinBot | 10-K Analyst

> A RAG-powered chatbot that answers questions about SEC 10-K filings — grounded in the actual document, with source citations.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What It Does

**FinBot** is a Retrieval-Augmented Generation (RAG) system that lets you "chat" with any public company's annual report (10-K filing). Instead of manually searching through 100+ pages of financial disclosures, just ask a question in plain English.

### Example Questions
- *"What was Apple's total net sales in the most recent fiscal year?"*
- *"What are the main risk factors for Tesla?"*
- *"How does Microsoft describe its competition?"*
- *"What was Google's research and development expense?"*

---

## ⚡ How It Works
User Question
↓
[Retrieval] → Search vector store for relevant 10-K sections
↓
[Augmentation] → Combine retrieved chunks into context
↓
[Generation] → Produce answer with source citations

### The Pipeline
1. **Fetch** — Downloads the latest 10-K from SEC EDGAR using `edgartools`
2. **Chunk** — Splits the document into 1024-token chunks with 128-token overlap
3. **Embed** — Converts chunks into vectors using `BAAI/bge-small-en-v1.5`
4. **Store** — Persists embeddings in ChromaDB
5. **Retrieve** — Finds the top-k most relevant chunks for your question
6. **Answer** — Returns the answer with source citations

---

## 🚀 How to Run Locally

# Clone the repository
git clone https://github.com/Alishba-12/finbot-10k-analyst.git
cd finbot-10k-analyst

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
Then open your browser to http://localhost:8501.

☁️ Deploy on Streamlit Cloud (Free)
Push this repo to GitHub

Go to share.streamlit.io

Connect your GitHub account

Select this repository

Click Deploy

🛠️ Tech Stack
Component	Tool
UI	Streamlit
RAG Framework	LangChain
Vector Database	ChromaDB
Embedding Model	BAAI/bge-small-en-v1.5
Data Source	SEC EDGAR (via edgartools)
📁 Project Structure
finbot-10k-analyst/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Documentation
└── .gitignore          # Git ignore rules
🎓 What This Project Demonstrates
✅ RAG implementation with real-world documents

✅ Vector embeddings and semantic search

✅ Financial document processing (10-K filings)

✅ Streamlit deployment

✅ End-to-end AI pipeline

⚠️ Limitations
Currently returns retrieved context as the answer (LLM integration coming soon)

Works best with English-language filings

Not financial advice — for informational purposes only

🔮 Roadmap
□ Integrate free LLM for polished answers
□ Support for 10-Q (quarterly) filings
□ Multi-company comparison mode
□ Evaluation metrics with Ragas
□ Cost & latency benchmarks
📄 Data Source
All 10-K filings are sourced from SEC EDGAR — the official U.S. Securities and Exchange Commission database.

👩‍💻 Author
Alishba

https://img.shields.io/badge/GitHub-Alishba--12-blue?style=flat&logo=github

