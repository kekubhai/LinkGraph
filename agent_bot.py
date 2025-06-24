from typing import TypedDict,List
from langchain_core.messages  import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    """State of the agent."""
    messages: List[HumanMessage]

# Initialize the LLM with Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

def process(state: AgentState) -> AgentState:
    """Process the agent state and generate a response."""
    response = llm.invoke(state["messages"])
    print(f"\n{response.content}")
    return state
 

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent=graph.compile()

user_input = input("Enter your message:   ")
while user_input!="exit":
      agent.invoke({"messages": [HumanMessage(content=user_input)]})
      user_input = input("Enter your message:   ")
