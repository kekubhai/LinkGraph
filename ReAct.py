from typing import Sequence, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """Add two numbers."""
    return a + b

@tool
def sub(a: int, b: int):
    """Subtract two numbers."""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiply two numbers."""
    return a * b

tools = [add, sub, multiply]
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are a helpful assistant.")
    messages = [system_prompt] + list(state["messages"])
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    return "end"

# Create the graph
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools", ToolNode(tools))

# Add edges
graph.add_edge(START, "our_agent")
graph.add_conditional_edges("our_agent", should_continue, {
    "continue": "tools",
    "end": END
})
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# Fixed input format
inputs = {"messages": [HumanMessage(content="Add 2 + 3 and then multiply the result with 2")]} 
print_stream(app.stream(inputs, stream_mode="values"))
