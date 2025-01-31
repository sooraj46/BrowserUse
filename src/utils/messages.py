from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Literal


class BaseMessage(ABC):
    """Abstract base class for chat messages."""

    def __init__(
        self, 
        content: Union[str, List[Union[str, Dict]]], 
        additional_kwargs: Optional[dict] = None
    ):
        """
        Initialize a BaseMessage.

        Args:
            content: The content of the message. Can be a string or a list
                     (which in turn can contain strings or dicts).
            additional_kwargs: Optional additional keyword arguments.
        """
        self.content = content
        self.additional_kwargs = additional_kwargs or {}

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the message, used for serialization."""

    def __repr__(self) -> str:
        """Return a string representation of the message."""
        return f"{self.__class__.__name__}(content={self.content!r})"

    def to_json_dict(self) -> Dict:
        """
        Serialize to a dictionary. 
        The 'type' and 'content' fields are always included;
        additional kwargs are also included if present.
        """
        return {
            "type": self.type,
            "content": self.content,
            **self.additional_kwargs,
        }

    @classmethod
    def from_json_dict(cls, data: Dict) -> "BaseMessage":
        """Deserialize from a dictionary."""
        _type = data.pop("type")
        content = data.pop("content")

        if _type == "human":
            return HumanMessage(content=content, additional_kwargs=data)
        elif _type == "ai":
            return AIMessage(content=content, additional_kwargs=data)
        elif _type == "system":
            return SystemMessage(content=content, additional_kwargs=data)
        elif _type == "SystemMessageChunk":
            return SystemMessageChunk(content=content, additional_kwargs=data)
        elif _type == "tool":
            return ToolMessage(
                content=content,
                tool_call_id=data.pop("tool_call_id"),
                additional_kwargs=data,
            )
        else:
            raise ValueError(f"Invalid message type: {_type}")


class HumanMessage(BaseMessage):
    """A message from a human."""

    type: Literal["human"] = "human"


class AIMessage(BaseMessage):
    """A message from an AI."""

    type: Literal["ai"] = "ai"


class SystemMessage(BaseMessage):
    """
    Message for priming AI behavior.

    Example:
        messages = [
            SystemMessage(content="You are a helpful assistant! Your name is Bob."),
            HumanMessage(content="What is your name?")
        ]
    """

    type: Literal["system"] = "system"

    def __init__(
        self, 
        content: Union[str, List[Union[str, Dict]]], 
        additional_kwargs: Optional[dict] = None
    ):
        """
        Pass in content as positional arg.

        Args:
            content: The string (or list) contents of the system message.
            additional_kwargs: Additional fields to pass to the message.
        """
        super().__init__(content=content, additional_kwargs=additional_kwargs)


class ToolMessage(BaseMessage):
    """A message from a tool."""

    type: Literal["tool"] = "tool"

    def __init__(
        self,
        content: Union[str, List[Union[str, Dict]]],
        tool_call_id: str,
        additional_kwargs: Optional[dict] = None,
    ):
        """
        Initialize a ToolMessage.

        Args:
            content: The content of the tool message.
            tool_call_id: The unique ID of the tool call.
            additional_kwargs: Additional fields for the message.
        """
        super().__init__(content=content, additional_kwargs=additional_kwargs)
        self.tool_call_id = tool_call_id

    def to_json_dict(self) -> Dict:
        """
        Serialize to a dictionary. 
        Includes 'tool_call_id' in addition to the base fields.
        """
        return {
            "type": self.type,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            **self.additional_kwargs,
        }


#
# Below is an example chunk class mirroring the style of SystemMessageChunk
#

class BaseMessageChunk(ABC):
    """
    Stub class to show how you'd structure a 'chunk' variant 
    if you need partial or streaming message content.
    """
    pass


class SystemMessageChunk(SystemMessage, BaseMessageChunk):
    """System Message chunk variant for streaming or partial content."""

    type: Literal["SystemMessageChunk"] = "SystemMessageChunk"

    def __init__(
        self, 
        content: Union[str, List[Union[str, Dict]]],
        additional_kwargs: Optional[dict] = None
    ):
        """
        Initialize a SystemMessageChunk.

        Args:
            content: The string (or list) contents of the system message chunk.
            additional_kwargs: Additional fields to pass to the message.
        """
        super().__init__(content=content, additional_kwargs=additional_kwargs)
