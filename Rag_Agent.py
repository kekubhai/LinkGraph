from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from typing import TypedDict, List, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

load_dotenv()

# Fixed: Use correct Gemini model name
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    temperature=0.0,
)

# Fixed: Use correct embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

pdf_path = "C:\\Users\\anirb\\Downloads\\Crypto_Analysis.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

try:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from the PDF.")
except Exception as e:
    raise RuntimeError(f"Failed to load PDF: {e}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(pages)
persist_directory = "C:\\Users\\anirb\\Downloads\\vector_store"

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="crypto_analysis",
    persist_directory=persist_directory
)

print(f"Created vector store with {len(splits)} document chunks.")

retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}
)

@tool
def retriever_tool(query: str) -> str:
    """Retrieve relevant information from the crypto analysis document."""
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the document. Please try asking something else."
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}\n")
    return "\n\n".join(results)

tools = [retriever_tool]
llm_with_tools = llm.bind_tools(tools)  # Fixed: Corrected binding

class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages] 

def should_continue(state: AgentState) -> str:
    """Determine if the agent should continue or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"

def agent_node(state: AgentState) -> AgentState:
    """The main agent node that processes messages."""
    system_prompt = SystemMessage(content="""You are a knowledgeable and reliable assistant specialized in cryptocurrency trends from June 2024 to June 2025. You use both your foundational knowledge and the retrieved documents to provide accurate, concise, and up-to-date responses.

Your purpose is to help users understand key developments in the crypto world, including:

- Global crypto market trends and investor behavior
- Price movements and technology updates of major cryptocurrencies (Bitcoin, Ethereum, etc.)
- Developments in DeFi, NFTs, altcoins, and stablecoins
- Regulatory changes across different countries
- Significant security incidents and ecosystem risks
- Layer-2 advancements and cross-chain protocols

Instructions:
- Use the retrieved information from the document to answer with relevant facts and data
- Keep responses fact-based and neutral
- Always cite dates and context for any statistics or events
- If the question is too vague, ask for clarification
- Do not fabricate information
- Summarize retrieved excerpts clearly in your own words

Tone: Professional, clear, and data-driven.""")
    
    messages = [system_prompt] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Create the graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")

app = graph.compile()

def run_rag_agent():
    """Run the RAG agent with user interaction."""
    print("RAG Agent started. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nAsk a question about crypto trends: ")
        if user_input.lower() == 'exit':
            break
            
        try:
            state = {"messages": [HumanMessage(content=user_input)]}
            for step in app.stream(state):
                if "agent" in step:
                    response = step["agent"]["messages"][-1]
                    if hasattr(response, 'content'):
                        print(f"\nAssistant: {response.content}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_rag_agent()



