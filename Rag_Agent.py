from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from typing import TypedDict, List, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Fixed: was "Chrom"
from langchain_core.tools import Tool

load_dotenv()

# Fixed: Use correct Gemini model name
llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-pro',  # Changed from 'gemini-2.5-pro'
    temperature=0.0,
)

# Fixed: Use correct embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"  # Changed from "gemini-embedding-exp-03-07"
)

pdf_path = "C:\\Users\\anirb\\Downloads\\Crypto_Analysis.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

try:
    # Fixed: Use PyPDFLoader properly
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from the PDF.")
except Exception as e:
    raise RuntimeError(f"Failed to load PDF: {e}")

# Add text splitter and vector store setup
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split documents
splits = text_splitter.split_documents(pages)
persist_directory= "C:\\Users\\anirb\\Downloads\\vector_store"
# Create vector store
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="crypto_analysis" , # Fixed: Added missing comma
    persist_directory=persist_directory
)
collection_name = "crypto_analysis"

print(f"Created vector store with {len(splits)} document chunks.")

retriever=vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}  # Adjust k as needed
)
@Tool
def retriver_tool(query:str)  -> str:
    docs=retriever.invoke(query)
    return docs
