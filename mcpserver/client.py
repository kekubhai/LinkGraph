from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # <--- import this if using ChatGroq
from langchain_core.messages import HumanMessage

import asyncio
import os

load_dotenv()

async def main():
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["mathserver.py"],
            "transport": "stdio",
        },
        "weather": {
            "url": "http://localhost:8000/mcp/weather",
            "transport": "streamable_http",
        }
    })

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    tools = await client.get_tools()
    model = ChatGroq(model="qwen-qwq-32b")
    agent = create_react_agent(model, tools)

    math_response = await agent.ainvoke({"messages":[{"role": "user", "content": "What is 2 + 2?"}]})
    print("Math response:", math_response['messages'][-1].content)
    
    

# run outside
asyncio.run(main())
