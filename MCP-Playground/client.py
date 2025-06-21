from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()
import logging
logging.basicConfig(level=logging.DEBUG)

async def main():
    print("Starting MCP client...")

    client = MultiServerMCPClient(
        {
            
            "math": {
                "command": os.sys.executable,
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    tools = await client.get_tools()
    print(f"Loaded tools: {tools}")

    model = ChatGroq(model="qwen/qwen3-32b")
    
    agent = create_react_agent(model, tools)
    
    math_response = await agent.ainvoke({
    "messages": [
        {"role": "user", "content": "What's (3 + 5) * 12?"}
    ]
        
    })
    

    weather_response = await agent.ainvoke({
    "messages": [
        {"role": "user", "content": "How is the weather in NJ?"}
    ]
    })

    print("Weather response:", weather_response["messages"][-1].content)
    print("math_response:", math_response["messages"][-1].content)

asyncio.run(main())
