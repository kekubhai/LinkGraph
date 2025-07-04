from typing import TypedDict, List, Union
from langchain_core.messages  import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    
    messages: List[Union[HumanMessage, AIMessage]]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
def process(state:AgentState)->AgentState:
    """This node will process the agent state and generate a response."""
    response=llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\n This is the response {response.content}")
    print("This is the Current State", state["messages"])
    return state

graph=StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent=graph.compile()

conversation_history=[]
user_input=input("Enter your message:   ")
while user_input!="exit":
    conversation_history.append(HumanMessage(content=user_input))
    result=agent.invoke({"messages": conversation_history})
    
    conversation_history=result["messages"]
    user_input=input("Enter your message:   ")



with open("logging.txt", "w") as file:
    file.write("Your messages are logges :\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
        file.write("\n End of conversation.\n")
            






