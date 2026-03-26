import uuid

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from fastapi import FastAPI
import task
import agent_card
import message

MAIN_TITLE = "倚天屠龙记助手" # 帮我定个title
app = FastAPI(title=MAIN_TITLE)
model = ChatOllama(model="qwen2.5:7b", temperature=0) # 换模型改这里

# 记得你的路径要改成你的电脑里的路径
SERVER_PARAMS = StdioServerParameters(
    command=r"",
    args=[r""],
)

@app.post("/student/chat")
async def chat_endpoint(request: message.MessageSendRequest):
    # 1. 提取基础信息
    message_query = message.get_all_text_parts(request)
    message_history = message.get_history_text(request)
    message_configuration = message.get_config_as_json_str(request)

    # 2. 构造系统提示词
    message_system_prompt = (
        f"你是一个名为 {MAIN_TITLE} 的智能助手。 "
        f"当前任务环境配置：输出模式支持 {message_configuration}。 "
        f"对话历史：{message_history}。 "
        "请结合历史上下文回答用户。如果需要使用工具，请直接调用。"
    )

    # 3. 直接在接口函数内运行异步逻辑 (不需要 event_generator)
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 加载工具并创建 Agent
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools, prompt=message_system_prompt)

            # 调用 Agent (注意输入格式)
            result = await agent.ainvoke({"messages": [("user", message_query)]})

            # 获取 AI 最后的文本回复
            ai_content = result["messages"][-1].content

            # 4. 使用你的 task.py 函数构造响应
            # 先创建带内容的响应对象
            message_send_response = task.create_response_with_artifact(str(ai_content))

            # 再追加请求到历史中（建议 taskId 动态化，这里先用你的 "task_01"）
            final_response = task.append_message_to_task_history(
                message_send_response,
                request,
                f"task_{uuid.uuid4().hex[:8]}"
            )

            # 直接返回 Pydantic 对象，FastAPI 会自动帮你转成 JSON
            return final_response


@app.get("/student/manifest", response_model=agent_card.AgentManifest)
async def get_agent_manifest():
    return {
        "capabilities": {
            "streaming": False
        },
        "defaultInputModes": [
            "text"
        ],
        "defaultOutputModes": [
            "text"
        ],
        "description": "",
        "name": "Agent",
        "skills": [
            {
                "description": "",
                "example": [
                    ""
                ],
                "id": "",
                "name": "",
                "tags": [
                    ""
                ]
            },
            {
                "description": "",
                "example": [
                    ""
                ],
                "id": "",
                "name": "",
                "tags": [
                    ""
                ]
            }
        ],
        "url": "",
        "version": ""
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)