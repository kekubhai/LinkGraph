from mcp.server.fastmcp import FastMCP
mcp=FastMCP("weather")
@mcp.tool()
async def get_weather(city:str)->str:
     """"Get the weather of the location"""
     return f"The weather is always reaining in {city}"
 
if __name__=="__main__":
     mcp.run(transport="streamable_http")
 