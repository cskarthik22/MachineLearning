#from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP
import os

mcp = FastMCP("Math")
os.environ["FASTAPI_PORT"] = "8081"

@mcp.tool()
def add(a:int,b:int) -> int:
    """Summary_
    Add two numbers
    """
    print(f"🧮 add() called with a={a}, b={b}")
    return a+b

@mcp.tool()
def multiply(a:int,b:int) -> int:
    """Summary_
    Multiply two numbers
    """
    print(f"🧮 mul() called with a={a}, b={b}")
    return a*b

if __name__ == "__main__":
    
    print("👂 Listening for stdio requests...")
    import sys
    sys.stdout.flush()
    ## "stdio" is used to test locally
    #mcp.run(transport="stdio") 
    #mcp.run(transport="streamable-http")
    mcp.run(
    transport="streamable-http",
    host="0.0.0.0",
    port=8081,
    path="/mcp",
    log_level="debug"
)

