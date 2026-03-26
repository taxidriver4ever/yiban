from typing import List
from pydantic import BaseModel, Field
"""
{
    "capabilities":{
        "streaming":false
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
    "url":"",
    "version":""
}
"""

class Skill(BaseModel):
    id: str = ""
    name: str = ""
    description: str = ""
    # 示例用法，通常是字符串列表
    example: List[str] = Field(default_factory=list)
    # 标签，用于分类或检索
    tags: List[str] = Field(default_factory=list)

class Capabilities(BaseModel):
    # 是否支持流式传输
    streaming: bool = False

class AgentManifest(BaseModel):
    name: str = "Agent"
    description: str = ""
    version: str = ""
    url: str = ""

    # 能力配置
    capabilities: Capabilities = Field(default_factory=Capabilities)

    # 默认输入/输出模式，例如 ["text", "image/png"]
    defaultInputModes: List[str] = Field(default_factory=lambda: ["text"])
    defaultOutputModes: List[str] = Field(default_factory=lambda: ["text"])

    # 技能清单
    skills: List[Skill] = Field(default_factory=list)