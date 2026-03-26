from typing import Optional, List, Literal
from pydantic import BaseModel, Field
import message

"""
{
    "id": "",
    "jsonrpc": "2.0",
    "result": {
        "artifacts": [
            {
                "artifactsId": "",
                "description": "",
                "name": "",
                "parts": [
                    {
                        "kind": "text",
                        "text": ""
                    }
                ]
            }
        ],
        "contextId": "",
        "history": [
            {
                "contextId": "",
                "kind": "message",
                "messageId": "",
                "parts": [
                    {
                        "kind": "text",
                        "text": ""
                    },
                    {
                        "kind": "file",
                        "file": {
                            "name": "",
                            "mimeType": "image/png",
                            "bytes": ""
                        }
                    }
                ],
                "role": "user",
                "taskId": ""
            }
        ],
        "id":"",
        "kind": "task",
        "status": {
            "state": "completed"
        }
    }
}
"""

# class FileData(BaseModel):
#     name: str = ""
#     mimeType: str = "image/png"
#     bytes: str = ""  # Base64 字符串

class Part(BaseModel):
    kind: Literal["text", "file"]
    text: Optional[str] = None
    # file: Optional[FileData] = None

class Artifact(BaseModel):
    artifactsId: str = ""
    description: str = ""
    name: str = ""
    parts: List[Part] = Field(default_factory=list)

class TaskStatus(BaseModel):
    state: Literal["pending", "processing", "completed", "failed"] = "completed"

class HistoryEntity(BaseModel):
    contextId: str = ""
    kind: str = "message"
    messageId: str = ""
    parts: List[Part] = Field(default_factory=list)
    role: Literal["user", "assistant", "system"]
    taskId: Optional[str] = ""

class TaskResultData(BaseModel):
    id: str = ""
    kind: str = "task"
    status: TaskStatus = Field(default_factory=TaskStatus)
    contextId: str = ""
    artifacts: List[Artifact] = Field(default_factory=list)
    history: List[HistoryEntity] = Field(default_factory=list)

class MessageSendResponse(BaseModel):
    id: str = ""
    jsonrpc: str = "2.0"
    result: TaskResultData


def append_message_to_task_history(
        task_response: MessageSendResponse,
        message_request: message.MessageSendRequest,
        task_id: str
):
    """
    将 message 提取出来，添加 taskId，并追加到 task 响应的 history 后面
    """
    source_message = message_request.params.message
    # 1. 将 MessageEntity 转换为字典
    msg_dict = source_message.model_dump()

    # 2. 补充 taskId 字段
    msg_dict["taskId"] = task_id

    # 3. 将字典转换为 task.py 中的 HistoryEntity 类型
    new_history_entry = HistoryEntity(**msg_dict)

    # 4. 追加到 task_response 的 history 列表中
    task_response.result.history.append(new_history_entry)

    return task_response


def create_response_with_artifact(content: str) -> MessageSendResponse:
    """
    输入一个字符串，自动创建一个完整的 MessageSendResponse，
    并将该字符串封装进 artifacts 的 parts 列表中。
    """
    # 1. 创建一个 Part 对象
    new_part = Part(kind="text", text=content)

    # 2. 创建一个 Artifact 对象，并把 part 放进去
    new_artifact = Artifact(
        artifactsId="art_001",
        name="Generated Content",
        description="AI generated artifact",
        parts=[new_part]
    )

    # 3. 组装成完整的 MessageSendResponse
    response = MessageSendResponse(
        id="1",
        jsonrpc="2.0",
        result=TaskResultData(
            id="task_001",
            kind="task",
            status=TaskStatus(state="completed"),
            contextId="ctx_001",
            artifacts=[new_artifact],
            history=[]  # 初始历史为空
        )
    )

    return response

