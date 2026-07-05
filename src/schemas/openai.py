from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

class ChatCompletionSystemMessage(BaseModel):
    role: str = "system"
    content: str

class ImageUrl(BaseModel):
    url: str # This will be the data:image/...;base64,... string

class MessageContentPartImage(BaseModel):
    type: str = "image_url"
    image_url: ImageUrl

class MessageContentPartText(BaseModel):
    type: str = "text"
    text: str

class ChatCompletionUserMessage(BaseModel):
    role: str = "user"
    content: Union[str, List[Union[MessageContentPartText, MessageContentPartImage]]]

class ChatCompletionRequest(BaseModel):
    model: str = "ndlocr-lite"
    messages: List[Union[ChatCompletionSystemMessage, ChatCompletionUserMessage]]
    stream: Optional[bool] = False

class ChatCompletionChoiceMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    model_config = {
        "extra": "ignore"
    }

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: "chatcmpl-" + "default")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: 0)
    model: str = "ndlocr-lite"
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
