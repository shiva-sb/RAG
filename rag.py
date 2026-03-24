import os
import re
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS 
from langchain_classic.chains import RetrievalQA

load_dotenv()

# --- SETTINGS ---
INDEX_PATH = "faiss_index_store"
FILE_PATH = "demo1.pdf"
EMBED_MODEL = "models/gemini-embedding-001"

# 1. Initialize Embeddings
# We initialize this first because FAISS needs the "model logic" to read the saved files
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, task_type="retrieval_document")

# 2. LOCAL STORAGE LOGIC
# This block is the most important part for your 1000/1000 limit problem.
if os.path.exists(INDEX_PATH):
    print(f" SUCCESS: Local index found at '{INDEX_PATH}'.")
    print(" Loading existing vectors from disk... (API Quota saved!)")
    # allow_dangerous_deserialization is required to load FAISS files locally
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("NOTICE: No local index found. Initializing first-time embedding...")
    print("This will use your Gemini API quota once to create the 'brain'.")
    
    # Load and Split the PDF
    loader = PyPDFLoader(FILE_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks from {FILE_PATH}.")

    # Create the Index (Sending all chunks in one go to stay under 100 RPM)
    try:
        db = FAISS.from_documents(texts, embeddings)
        
        # SAVE THE INDEX LOCALLY
        # This creates the 'faiss_index_store' folder on your computer
        db.save_local(INDEX_PATH)
        print(f"SUCCESS: Index saved locally to '{INDEX_PATH}'.")
        print("You can now restart this script without using embedding quota.")
        
    except Exception as e:
        print(f"CRITICAL ERROR during embedding: {e}")
        print("If you see 'RESOURCE_EXHAUSTED', wait until 12:30 PM IST today for the reset.")
        exit()

# 3. Setup LLM & Chain (Using Gemini 3 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-3-flash", temperature=0.1)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=db.as_retriever(search_kwargs={"k": 3})
)

# 4. Chat Loop
print("\n--- RAG SYSTEM ONLINE ---")
print("(Type 'exit' to quit)\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit", "bye"]: break
    if not user_input: continue

    try:
        print("--- Thinking ---")
        response = rag_chain.invoke(user_input)
        print(f"\nAI: {response['result']}\n")
    except Exception as e:
        if "429" in str(e):
            print("Rate limit reached for the Chat model. Please wait a moment.")
        else:
            print(f"Error: {e}")