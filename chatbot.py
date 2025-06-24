from typing import TypedDict,List
from langchain_core.messages  import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    
    messages: List[HumanMessage, AIMessage]
    
    
    
    
    
    
    
        
    