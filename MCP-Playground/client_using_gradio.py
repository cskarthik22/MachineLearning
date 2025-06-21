import gradio as gr
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Async helper
async def setup_agent():
    client = MultiServerMCPClient({
        '''
        "math": {
            "command": os.sys.executable,
            "args": ["mathserver.py"],
            "transport": "stdio",
        },
        '''
        "math": {
            "url": "http://localhost:8081/mcp",
            "transport": "streamable_http",
        },
        
        "weather": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()
    llm = ChatGroq(model="qwen/qwen3-32b")
    agent = create_react_agent(llm, tools)
    return agent

agent = asyncio.run(setup_agent())

# Gradio wrapper
def ask_agent(question):
    response = asyncio.run(agent.ainvoke({
    "messages": [
        {"role": "user", "content": question}
    ]
    }))
    return response["messages"][-1].content

gr.Interface(fn=ask_agent, inputs="text", outputs="text", title="LangGraph + Groq Agent").launch()
