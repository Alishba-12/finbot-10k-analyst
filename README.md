# 📊 FinBot | 10-K Analyst

> A RAG-powered chatbot that answers questions about SEC 10-K filings, grounded in the source document with citations.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://python.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3-orange.svg)](https://groq.com/)

**Live Demo:** [finbot-10k-analyst.streamlit.app](https://finbot-10k-analyst-4byi7ymuckbpri7yxwghik.streamlit.app/)

---

## What It Does

FinBot lets you "chat" with any public company's annual report (10-K filing). Instead of manually searching through 100+ pages of financial disclosures, you pick a company, ask a question in plain English, and get a synthesized answer with citations back to the source passages.

### Example Questions
- What was Apple's total net sales in the most recent fiscal year?
- What are the main risk factors for Microsoft?
- How much does Nvidia spend on research and development?
- What is Tesla's dividend policy?

---

## How It Works

User Question
|
v
[Retrieval] Search vector store for relevant 10-K sections
|
v
[Augmentation] Combine retrieved chunks into a grounded context
|
v
[Generation] LLM produces an answer restricted to that context
|
v
Answer + Source Citations


### Pipeline
1. **Fetch** — Downloads the latest 10-K from SEC EDGAR using edgartools
2. **Chunk** — Splits the document into 1024-token windows with 128-token overlap, using a token-aware recursive splitter that prioritizes paragraph and sentence boundaries
3. **Embed** — Converts chunks into vectors using BAAI/bge-small-en-v1.5
4. **Store** — Persists embeddings in ChromaDB
5. **Retrieve** — Finds the top-k most semantically relevant chunks for the question
6. **Generate** — Groq-hosted LLM produces an answer restricted to the retrieved context, with source markers

---

## Retrieval Validation

Retrieval quality was measured on a hand-built gold set of 10 question-answer pairs against Apple's 10-K. Each question has a target keyword or phrase, and a question counts as a hit if the correct passage appears in the top-5 retrieved chunks.

| Metric | Result |
|--------|--------|
| Hit Rate @ 5 | [X]/10 ([X]%) |

Manual spot-checks were also performed: for a sample of answers, the retrieved figure was cross-referenced against the original 10-K PDF to confirm it matched.

The test set and script are available in `notebooks/retrieval_eval.ipynb`.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| UI | Streamlit |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| LLM | Groq (openai/gpt-oss-120b) |
| Data Source | SEC EDGAR via edgartools |
| Hosting | Streamlit Community Cloud |

---

## Project Structure

finbot-10k-analyst/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Documentation
├── LICENSE # MIT License
├── notebooks/
│ └── retrieval_eval.ipynb # Retrieval validation script
└── .streamlit/
└── config.toml # Theme and server settings


---

## Coverage

FinBot is ticker-agnostic. It can query any public company that files a Form 10-K with the SEC, which is roughly 6,700 companies per year. The dropdown ships with 12 commonly requested companies for convenience, but the underlying pipeline works for any valid ticker.

End-to-end testing was performed on Apple (AAPL). A larger sweep across tickers is a planned next step.

---

## How to Run Locally

```bash
git clone https://github.com/Alishba-12/finbot-10k-analyst.git
cd finbot-10k-analyst
pip install -r requirements.txt
streamlit run app.py
Then open your browser to http://localhost:8501.

Secrets Required
Create .streamlit/secrets.toml with:
SEC_IDENTITY = "kalishbakhan456@gmail.com"
GROQ_API_KEY = "gsk_......"

The SEC identity is required by EDGAR to identify API users. The Groq key is free at console.groq.com.

Deployment
Deployed on Streamlit Community Cloud (free tier). Push to the main branch and the app redeploys automatically.

What This Project Demonstrates
Retrieval-Augmented Generation over real, messy financial documents

Token-aware chunking with overlap for long structured text

Vector embeddings and semantic search

Context-restricted prompting with citation enforcement

End-to-end deployment with secrets management

Retrieval evaluation using a gold set

Limitations
Only one company can be loaded at a time. Multi-company comparison is not yet supported.

Works best with English-language filings.

Not financial advice. For informational purposes only.

Chunking can still split some financial tables across boundaries, which occasionally affects retrieval on table-heavy questions.

Roadmap
Multi-company comparison mode

10-Q (quarterly) filing support

Hybrid search (BM25 + dense) for better keyword matching

Re-ranking layer to improve top-k precision

Ragas-based evaluation for faithfulness and answer relevance

Cost and latency benchmarks

Data Source
All 10-K filings are sourced from SEC EDGAR, the official U.S. Securities and Exchange Commission database. Filings are public domain.

Author
Alishba

https://img.shields.io/badge/GitHub-Alishba--12-blue?style=flat&logo=github

⭐ If you find this useful, please give it a star.

---

## Two Things to Fill In Before You Push

1. **Hit Rate @ 5** — replace `[X]/10 ([X]%)` with your actual number once you run the validation script
2. **Model name in the badge** — the badge says "Llama 3.3" but the code now uses `openai/gpt-oss-120b`. Either update the badge text to match, or leave the badge as-is and note that the LLM has been updated.

---

## Push Commands

**Where:** 💻 Local Computer, in your project folder

```bash
git add README.md
git commit -m "Update README: LLM integration, retrieval validation, updated model"
git push origin main
