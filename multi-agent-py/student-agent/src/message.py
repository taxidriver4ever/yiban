from typing import Optional, List, Literal
from pydantic import BaseModel, Field

"""
{
    "id":"",
    "jsonrpc":"2.0",
    "method":"message/send",
    "params": {
        "configuration": {
            "acceptedOutputModes": [
                "text",
                "text/plain",
                "image/png"
            ]
        },
        "message": {
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
            "role": "user"
        },
        "history": [
            {
                "contextId": "",
                "kind": "message",
                "messageId": "",
                "parts": [
                    {
                        "kind": "text",
                        "text": ""
                    }
                ],
                "role": "user",
                "taskId": ""
            }
        ]
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



class HistoryEntity(BaseModel):
    contextId: str = ""
    kind: str = "message"
    messageId: str = ""
    parts: List[Part] = Field(default_factory=list)
    role: Literal["user", "assistant", "system"]
    taskId: Optional[str] = ""

class MessageEntity(BaseModel):
    contextId: str = ""
    kind: str = "message"
    messageId: str = ""
    parts: List[Part] = Field(default_factory=list)
    role: Literal["user", "assistant", "system"]

class Configuration(BaseModel):
    acceptedOutputModes: List[str] = ["text", "text/plain", "image/png"]



class MessageSendParams(BaseModel):
    configuration: Configuration = Field(default_factory=Configuration)
    message: MessageEntity
    history: List[HistoryEntity] = Field(default_factory=list)



class MessageSendRequest(BaseModel):
    id: str = ""
    jsonrpc: str = "2.0"
    method: str = "message/send"
    params: MessageSendParams


def get_all_text_parts(req: MessageSendRequest) -> str:
    # 1. 访问 params -> message -> parts
    # 2. 遍历 parts，如果 part.text 有值就取出来
    # 3. 用换行符或其他分隔符拼接起来
    texts = [part.text for part in req.params.message.parts if part.text is not None]

    return "\n".join(texts)


def get_history_text(req: MessageSendRequest) -> str:
    all_history_texts = []

    # 1. 遍历 history 列表（每一项是一个 HistoryEntity）
    for entry in req.params.history:
        # 2. 遍历该条消息下的 parts
        parts_text = [p.text for p in entry.parts if p.kind == "text" and p.text]

        # 3. 将这一轮的角色和内容组合（可选，方便阅读）
        if parts_text:
            content = " ".join(parts_text)
            all_history_texts.append(f"{entry.role}: {content}")

    return "\n".join(all_history_texts)

def get_config_as_json_str(req: MessageSendRequest) -> str:
    """将 configuration 对象完整转换为 JSON 字符串"""
    # indent=2 可以让输出带有缩进，方便阅读
    return req.params.configuration.model_dump_json(indent=2)