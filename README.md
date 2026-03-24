# ⚡ Gemini-Powered Persistent RAG System

A high-efficiency Retrieval-Augmented Generation (RAG) pipeline built with LangChain, FAISS, and Gemini 3 Flash. This system is engineered to handle local PDF documents while strictly respecting Google AI Studio's free-tier API quotas through persistent vector storage.

- **Smart Persistence:** Automatically saves/loads FAISS indices to disk to bypass daily embedding limits.
- **Quota Optimized:** "One-Hit" batching strategy to stay under 100 RPM and 1,000 RPD.
- **Modern LLM:** Powered by Gemini 3 Flash for fast, grounded reasoning.
- **Error Resilient:** Built-in auto-retry logic for `429 RESOURCE_EXHAUSTED` errors.

## Section: Environment Variables
To run this project, you will need to add the following environment variables to your `.env` file:

`GOOGLE_API_KEY` - Your Gemini API key from [Google AI Studio](https://aistudio.google.com/)

## Section: Installation
Install the project dependencies with pip:

```bash
  python -m venv env 
```
```bash
  .\env\Scripts\activate
```
```bash
  pip install -r requirements.txt
```

### Section: Run Locally
```markdown
1. Ensure your PDF (e.g., `demo1.pdf`) is in the root directory.
2. Run the main script: python rag.py
3. On the first run, the script will generate a faiss_index_store folder. Subsequent runs will load this locally without using embedding quota.
```
---

### Section: Tech Stack
```markdown
Language: Python 3.9+  
Orchestration: LangChain  
LLM: Google Gemini 3 Flash  
Embeddings: Gemini-Embedding-001 (3072 dimensions)  
Vector Database: FAISS (Facebook AI Similarity Search)
```

### Section: FAQ

#### When does my quota reset?
The 1,000 RPD (Requests Per Day) limit resets daily at 12:00 AM PT (**12:30 PM IST**).

#### How do I change the PDF?
Simply replace the PDF file in the directory and **delete** the `faiss_index_store` folder so the script knows to re-index the new data.

#### Does this work offline?
The vector retrieval works offline once the index is created, but the Gemini 3 Flash model requires an internet connection to generate the final answer.
