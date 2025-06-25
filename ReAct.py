from typing import Annoted ,Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage,SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()
# Annoted - provides 
class AgentState(TypedDict):
    """State of the agent."""
    messages: Sequence[Sequence[BaseMessage,add_messages]]
def add(a:int, b:int):
    return a+b
tools=[add]
model=ChatGoogleGenerativeAI(
    model_name="gemini-1.5-flash",
    temperature=0.2,
).bind_tools(tools)
def model_call(state:AgentState)->AgentState:
    system_prompt=SystemMessage(content="You are a helpful assistant.")
    
    response=model.invoke({system_prompt})
    return {"messages": [response]}
def should_continue(state:AgentState):
    messages=state["messages"]
    if not messages[-1].tool_call:
        return "end"
    return "continue"

# Create the graph
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools", ToolNode(tools))
graph.add_edge("tools","our_agent")
