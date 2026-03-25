import json
from typing import Optional, List, Literal

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

# ==========================================
# 1. 基础组件 (Base Components)
# ==========================================

class FileData(BaseModel):
    name: str = ""
    mimeType: str = "image/png"
    bytes: str = ""  # Base64 字符串

class Part(BaseModel):
    kind: Literal["text", "file"]
    text: Optional[str] = None
    file: Optional[FileData] = None

# ==========================================
# 2. 消息与历史实体 (Message & History)
# ==========================================

class MessageEntity(BaseModel):
    contextId: str = ""
    kind: str = "message"
    messageId: str = ""
    parts: List[Part] = Field(default_factory=list)
    role: Literal["user", "assistant", "system"]
    taskId: Optional[str] = ""

# ==========================================
# 3. 任务结果组件 (Task Result Components)
# ==========================================

class Artifact(BaseModel):
    artifactsId: str = ""
    description: str = ""
    name: str = ""
    parts: List[Part] = Field(default_factory=list)

class TaskStatus(BaseModel):
    state: Literal["pending", "processing", "completed", "failed"] = "completed"

# ==========================================
# 4. 完整的请求模型 (Request Model: message/send)
# ==========================================

class Configuration(BaseModel):
    acceptedOutputModes: List[str] = ["text", "text/plain", "image/png"]

class MessageSendParams(BaseModel):
    configuration: Configuration = Field(default_factory=Configuration)
    message: MessageEntity
    history: List[MessageEntity] = Field(default_factory=list)

class MessageSendRequest(BaseModel):
    id: str = ""
    jsonrpc: str = "2.0"
    method: str = "message/send"
    params: MessageSendParams

# ==========================================
# 5. 完整的响应模型 (Response Model: Task Result)
# ==========================================

class TaskResultData(BaseModel):
    id: str = ""
    kind: str = "task"
    status: TaskStatus = Field(default_factory=TaskStatus)
    contextId: str = ""
    artifacts: List[Artifact] = Field(default_factory=list)
    history: List[MessageEntity] = Field(default_factory=list)

class MessageSendResponse(BaseModel):
    id: str = ""
    jsonrpc: str = "2.0"
    result: TaskResultData

SYSTEM_PROMPT = ""

@app.post("/student/chat")
async def chat_endpoint(message_send_request:MessageSendRequest):

    query = ""

    async def event_generator():
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:

                await session.initialize()

                tools = await load_mcp_tools(session)
                agent = create_react_agent(model, tools, state_modifier=SYSTEM_PROMPT)

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