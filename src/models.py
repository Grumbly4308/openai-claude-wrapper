from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .json_mode import responses_text_format


# ---------- Chat completion request ----------


class TextContent(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="allow")
    url: str
    detail: Optional[str] = None


class ImageContent(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["image_url"]
    image_url: ImageURL


class InputAudioPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    data: str
    format: Optional[str] = None


class InputAudioContent(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["input_audio"]
    input_audio: InputAudioPayload


class FilePayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    file_id: Optional[str] = None
    file_data: Optional[str] = None
    filename: Optional[str] = None
    mime_type: Optional[str] = None


class FileContent(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["file"]
    file: FilePayload


ContentPart = Union[TextContent, ImageContent, InputAudioContent, FileContent]


# ---------- OpenAI function calling (tools) ----------


class ToolFunctionDef(BaseModel):
    """The ``function`` member of a ``tools[]`` entry: a caller-defined tool."""

    model_config = ConfigDict(extra="allow")
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None  # JSON Schema, passed verbatim


class ToolDef(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "function"
    function: ToolFunctionDef


class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: Optional[str] = None
    arguments: Optional[str] = None  # serialized JSON string, per OpenAI spec


class ToolCall(BaseModel):
    """An assistant-emitted tool call, echoed back by clients in history."""

    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Union[str, list[ContentPart], None] = None
    name: Optional[str] = None
    # Function-calling history fields: an assistant turn that requested tool
    # calls, and the tool turn carrying a result (keyed by tool_call_id).
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ResponseFormat(BaseModel):
    """OpenAI structured-output request (the ``response_format`` parameter).

    ``json_schema`` holds the OpenAI envelope ({"name", "schema", "strict"})
    or, from lenient clients, the bare JSON Schema itself.
    """

    model_config = ConfigDict(extra="allow")
    type: str = "text"  # "text" | "json_object" | "json_schema"
    json_schema: Optional[dict[str, Any]] = None


class ResponseFormat(BaseModel):
    """OpenAI structured-output request (the ``response_format`` parameter).

    ``json_schema`` holds the OpenAI envelope ({"name", "schema", "strict"})
    or, from lenient clients, the bare JSON Schema itself.
    """

    model_config = ConfigDict(extra="allow")
    type: str = "text"  # "text" | "json_object" | "json_schema"
    json_schema: Optional[dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
    # Non-standard helper field: pin a session id explicitly so the
    # server keeps the same Claude Code session across turns.
    session_id: Optional[str] = Field(default=None, alias="session_id")
    # Whether to return generated files inline as base64 in the
    # final assistant message (true) or as file_id references (false).
    inline_generated_files: bool = False
    # Per-request override for the interactive clarification protocol. None
    # (default) follows the server setting; false opts this request out (e.g. a
    # non-interactive/automated caller that must never pause to ask questions).
    clarify: Optional[bool] = None
    # OpenAI structured output: when type is json_object/json_schema the
    # response content is guaranteed to be a raw JSON value — no markdown
    # fences, no prose, no file-reference trailer. Sent by e.g. the Vercel
    # AI SDK's generateObject (Vane, and other OpenAI-compatible clients).
    response_format: Optional[ResponseFormat] = None
    # OpenAI function calling. When `tools` is a non-empty list the CLIENT owns
    # the agent loop: the request bypasses the Claude Code CLI entirely (no
    # built-in tools, no internal loop) and is served by the tool bridge, which
    # calls the Anthropic Messages API directly and returns tool_calls for the
    # client to execute.
    tools: Optional[list[ToolDef]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    # {"include_usage": true} => streaming responses append a usage chunk
    # before [DONE] (Vercel AI SDK sends this).
    stream_options: Optional[dict[str, Any]] = None


# ---------- Chat completion response ----------


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    attachments: Optional[list[dict[str, Any]]] = None
    # Function calls requested by the model (tool-bridge path only). Each entry
    # is {"id", "type": "function", "function": {"name", "arguments": <str>}}.
    tool_calls: Optional[list[dict[str, Any]]] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)
    session_id: Optional[str] = None
    # Non-standard: the reasoning effort actually applied to this run and where
    # it came from, e.g. {"applied": "medium", "source": "server-default",
    # "requested": null}. Lets clients confirm effort selection took effect.
    effort: Optional[dict[str, Any]] = None


# ---------- Streaming chunks ----------


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    # Reasoning is streamed on its own channel so it renders as a collapsible
    # "thinking" block (OpenWebUI et al.) instead of being concatenated into the
    # answer. Clients that don't understand the field simply ignore it.
    reasoning_content: Optional[str] = None
    # Incremental tool-call frames (tool-bridge path only). First frame per call
    # carries {"index", "id", "type", "function": {"name", "arguments": ""}};
    # later frames carry only {"index", "function": {"arguments": <fragment>}}.
    tool_calls: Optional[list[dict[str, Any]]] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    session_id: Optional[str] = None
    # Non-standard: resolved reasoning effort (emitted on the first chunk only),
    # same shape as ChatCompletionResponse.effort.
    effort: Optional[dict[str, Any]] = None
    # Populated only on the trailing usage chunk when the client asked for it
    # via stream_options.include_usage (that chunk carries empty choices).
    usage: Optional[Usage] = None


# ---------- Models list ----------


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "anthropic"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ---------- Legacy completions ----------


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    prompt: Union[str, list[str]]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, list[str]]] = None
    user: Optional[str] = None
    n: int = 1
    echo: bool = False
    suffix: Optional[str] = None
    logprobs: Optional[int] = None


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: Optional[Any] = None
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# ---------- Responses API ----------


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    input: Union[str, list[Any]]
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    user: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    # Structured output, Responses-API spelling: {"format": {"type":
    # "json_schema", "name": …, "schema": {…}}}. Normalized below into the same
    # `response_format` the chat path uses, so one JSON-mode implementation
    # covers both endpoints.
    text: Optional[dict[str, Any]] = None
    response_format: Optional[ResponseFormat] = None

    @model_validator(mode="after")
    def _adopt_text_format(self) -> "ResponsesRequest":
        if self.response_format is None:
            kwargs = responses_text_format(self.text)
            if kwargs is not None:
                self.response_format = ResponseFormat(**kwargs)
        return self


# ---------- Embeddings ----------


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    input: Union[str, list[str], list[int], list[list[int]]]
    model: str = "text-embedding-3-small"
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: Union[list[float], str]
    index: int


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    usage: Usage = Field(default_factory=Usage)


# ---------- Moderations ----------


class ModerationsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    input: Union[str, list[str]]
    model: str = "omni-moderation-latest"


# ---------- Images ----------


class ImageGenRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt: str
    model: str = "claude-svg"
    n: int = 1
    size: Optional[str] = "1024x1024"
    response_format: Literal["url", "b64_json"] = "b64_json"
    quality: Optional[str] = None
    style: Optional[str] = None
    user: Optional[str] = None
    background: Optional[str] = None


# ---------- TTS ----------


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "claude-tts"
    input: str
    voice: str = "en"
    response_format: Literal["mp3", "wav", "ogg", "flac", "opus", "aac", "pcm"] = "mp3"
    speed: float = 1.0


# ---------- Batches ----------


class BatchCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_file_id: str
    endpoint: str
    completion_window: str = "24h"
    metadata: Optional[dict[str, Any]] = None


# ---------- Assistants / Threads ----------


class AssistantCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[list[dict[str, Any]]] = None
    metadata: Optional[dict[str, Any]] = None


class ThreadCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    messages: Optional[list[dict[str, Any]]] = None
    metadata: Optional[dict[str, Any]] = None


class ThreadMessageCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["user", "assistant"] = "user"
    content: Union[str, list[Any]]
    attachments: Optional[list[dict[str, Any]]] = None
    metadata: Optional[dict[str, Any]] = None


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    assistant_id: str
    model: Optional[str] = None
    instructions: Optional[str] = None
    additional_instructions: Optional[str] = None
    stream: bool = False
    metadata: Optional[dict[str, Any]] = None


# ---------- Vector stores ----------


class VectorStoreCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: Optional[str] = None
    file_ids: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    expires_after: Optional[dict[str, Any]] = None


class VectorStoreFileAdd(BaseModel):
    model_config = ConfigDict(extra="allow")
    file_id: str


class VectorStoreSearchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    query: str
    max_num_results: int = 5
    filters: Optional[dict[str, Any]] = None
