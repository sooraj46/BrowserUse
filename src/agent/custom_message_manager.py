from __future__ import annotations

import logging
from typing import List, Optional, Type

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.message_manager.views import MessageHistory
from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.views import ActionResult, AgentStepInfo
from browser_use.browser.views import BrowserState

#from ..utils.messages import BaseMessage, HumanMessage
from langchain_core.messages import HumanMessage, BaseMessage
from .custom_prompts import CustomAgentMessagePrompt

logger = logging.getLogger(__name__)

class CustomMessageManager(MessageManager):
    def __init__(
        self,
        llm=None,  # Make LLM optional
        task: str = "",
        action_descriptions: str = "",
        system_prompt_class: Type[SystemPrompt] = SystemPrompt,
        max_input_tokens: int = 128000,
        estimated_characters_per_token: int = 3,
        image_tokens: int = 800,
        include_attributes: list[str] = None,
        max_error_length: int = 400,
        max_actions_per_step: int = 10,
        message_context: Optional[str] = None
    ):
        if include_attributes is None:
            include_attributes = []

        # Pass llm (which may be None) to the parent constructor
        super().__init__(
            llm=llm,
            task=task,
            action_descriptions=action_descriptions,
            system_prompt_class=system_prompt_class,
            max_input_tokens=max_input_tokens,
            estimated_characters_per_token=estimated_characters_per_token,
            image_tokens=image_tokens,
            include_attributes=include_attributes,
            max_error_length=max_error_length,
            max_actions_per_step=max_actions_per_step,
            message_context=message_context
        )

        self.tool_id = 1

        # Keep a separate local history if needed
        self.history = MessageHistory()

        # Add the system prompt as the first message
        self._add_message_with_tokens(self.system_prompt)

        # If there's extra context, add it as a HumanMessage
        if self.message_context:
            context_message = HumanMessage(content=self.message_context)
            self._add_message_with_tokens(context_message)

        task_message = HumanMessage(
                        content=[
                                {
                                    "type": "text",
                                    "text": f"Your ultimate task is: {self.task}"
                                }
                                ]
                                    )
        self._add_message_with_tokens(task_message)

    def build_message_list(self) -> list[BaseMessage]:
        """
        Builds the list of messages for the LLM, including the system prompt and the conversation history.
        """
        self.cut_messages()
        return self.history.messages

    def cut_messages(self):
        """Trim the current message list to respect max_input_tokens."""
        diff = self.history.total_tokens - self.max_input_tokens
        min_message_len = 2 if self.message_context is not None else 1

        while diff > 0 and len(self.history.messages) > min_message_len:
            self.history.remove_message(min_message_len)
            diff = self.history.total_tokens - self.max_input_tokens

    def add_state_message(
        self,
        state: BrowserState,
        result: Optional[List[ActionResult]] = None,
        step_info: Optional[AgentStepInfo] = None,
    ) -> None:
        """
        Add browser state as human message so that LLM sees the current environment and results.
        """
        state_message = CustomAgentMessagePrompt(
            state,
            result,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            step_info=step_info,
        ).get_user_message()
        self._add_message_with_tokens(state_message)

    def _count_text_tokens(self, text: str) -> int:
        # Using character-based estimation.
        return len(text) // self.estimated_characters_per_token
    
    def add_screenshot_to_latest_message(
        self, messages: list[BaseMessage], screenshot_base64: str
    ) -> list[BaseMessage]:
        if not messages:
            messages.append(
                HumanMessage(content=[
                    {"type": "text", "text": ""},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                ])
            )
            return messages

        latest_managed_message = messages[-1]
        if not isinstance(latest_managed_message, BaseMessage):
            latest_message = latest_managed_message.message
        else:
            latest_message = latest_managed_message

        if isinstance(latest_message, HumanMessage):
            # **Check for str content, convert to list if needed**
            if isinstance(latest_message.content, str):
                latest_message.content = [{"type": "text", "text": latest_message.content}]

            # Now safely append
            latest_message.content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}
                }
            )
        else:
            messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": ""},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}
                        }
                    ]
                )
            )

        return messages
