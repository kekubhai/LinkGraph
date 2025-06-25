from typing import Sequence, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
load_dotenv()

document_content=""
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    
    
@tool
def update(content:str)->str:
    """Update the document content."""
    global document_content
    document_content = content
    return f"Document updated with: {content}"
@tool 
def save(filename:str) ->str:
    """Save the document content to a file."""
    global document_content
    if not filename.endswith(".txt"):
        filename += ".txt"
    try :
        with open(filename, "w") as file:
            file.write(document_content)
            print(f"Document saved to {filename}")
            return f"document saved to {filename}"
    except Exception as e:
        print(f"Error saving document: {e}")
        return f"Error saving document: {e}"
tools=[update,save]
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2).bind_tools(tools)

def my_agent(state:AgentState)->AgentState:
    system_prompt=SystemMessage(content=f"""Your are a Drafter , a helpful writing assistant . You are going to help the user to update and modify documents.
                                " - If the user wants to update the document, they can use the update tool ."
                                " - If the user wants to save and finish the document, they can use the save tool ."
                                "Make sure to always show the current document state after modifications"
                                " - The current document content is :{document_content}" """)
    if not state['messages']:
        user_input="I am ready to help you update a document . What would you like to create?"
        user_message=HumanMessage(content=user_input)
        
    else :
        user_input=input("What would you like to do with the document? ") 
        print(f"\nUser input: {user_input}")
        user_message=HumanMessage(content=user_input)
           
           
    all_messages = [system_prompt] + list(state["messages"]) + [user_message] 
    response=model.invoke(all_messages)   
    
    print(f"\nAI response: {response.content}")   
    return {"messages": [response]}

def should_continue(state:AgentState)->str:
    
    messages=state["messages"]
    if not messages:
        return "continue"
    for message in reversed(messages):
        
        if isinstance(message,ToolMessage) and \
              "saved" in message.content.lower() and "document" in message.content.lower():
              return  "end"
          
        return "continue"
    
    
def print_messages(messages):
    if not messages:
        return 
    for message in messages[:-3]:
        if isinstance(message.ToolMessage):
            print(f"Tool call: {message.content}")
            
graph = StateGraph(AgentState)    
graph.add_node("my_agent", my_agent)
graph.add_node("tools", ToolNode(tools)) 

graph.set_entry_point("my_agent")
graph.add_edge("my_agent", "tools")  # Fixed: was "agent"
graph.add_conditional_edges("my_agent", should_continue, {
    "continue": "tools",
    "end": END
}) 
app = graph.compile()
              
    
def run_drafter():
    print("\n ================== Drafter Agent =================")
    state={"messages":[]}
    for step in app.stream(state,stream_mode="values"):
        if "messages"in step:
            print_messages(step["messages"])
    print("\n ================== End of Drafter Agent =================")            
    
if __name__ == "__main__":
    run_drafter()