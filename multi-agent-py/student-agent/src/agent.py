import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

app = FastAPI(title="MCP Agent API")
model = ChatOllama(model="qwen2.5:7b", temperature=0) # 换模型改这里

# 记得你的路径要改成你的电脑里的路径
SERVER_PARAMS = StdioServerParameters(
    command=r"",
    args=[r""],
)

class Request(BaseModel):
    # 这个请求数据只是暂时的，要改成下面task.json的形式
    query: str = Field(..., description="用户的问题内容")
    user_id: str = Field(default="guest", description="用户唯一标识")

@app.post("/student/chat")
async def chat_endpoint(request_data: Request):
    query = request_data.query

    async def event_generator():
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await load_mcp_tools(session)
                agent = create_react_agent(model, tools)

                inputs = {"messages": [("user", query)]}

                async for chunk, metadata in agent.astream(inputs, stream_mode="messages"):
                    node = metadata.get("langgraph_node")

                    # 返回值的json是下面的message.json之后来改
                    if node == "agent" and chunk.content:
                        yield f"data: {json.dumps({'type': 'content', 'data': chunk.content}, ensure_ascii=False)}\n\n"

                    elif node == "tools" and chunk.content:
                        tool_name = getattr(chunk, 'name', '未知工具')
                        status_msg = f"正在调用工具：[{tool_name}] 检索知识库..."

                        yield f"data: {json.dumps({'type': 'status', 'data': status_msg}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)