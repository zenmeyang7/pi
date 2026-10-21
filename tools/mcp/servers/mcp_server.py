from mcp.server.mcpserver import MCPServer

mcp = MCPServer("mcp_server")

@mcp.tool()
def add(a:int,b:int)->int:
    """Add Two Numbers"""
    return a+b

@mcp.resource("greeting://{name}")
def get_greeting(name:str)->str:
    """Get a Personalized greeting"""
    return f"hello {name}"

if __name__ == "__main__":
    mcp.run("stdio") ## 本地client连接,标准的输入输出
    ## mcp.run("see") 远程连接
    ## mcp.run("Streamable HTTP") 可流式http，客户端向单一的mcp端点发送请求