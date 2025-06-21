#from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location:str)->str:
    """Get the weather location."""
    return "Its sunny here in NJ..."

if __name__=="__main__":

    # expose mcp server over http
    mcp.run(transport="streamable-http")
