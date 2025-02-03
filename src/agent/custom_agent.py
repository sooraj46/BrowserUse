# src/agent/custom_agent.py

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Type

import google.genai as genai
from google.genai import types

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList, AgentOutput
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller
from browser_use.utils import time_execution_async

from src.utils.agent_state import AgentState
from src.agent.custom_views import (
    CustomActionResult,
    CustomAgentOutput,
    CustomAgentStepInfo,
    ActionModel,
)
from src.agent.custom_message_manager import CustomMessageManager

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class CustomAgent(Agent):
    def __init__(
        self,
        task: str,
        llm=None,  # Must match Agent.__init__(..., llm: BaseChatModel)
        add_infos: str = "",
        browser: Browser | None = None,
        browser_context: BrowserContext | None = None,
        controller: Controller = Controller(),
        use_vision: bool = True,
        save_conversation_path: Optional[str] = None,
        save_conversation_path_encoding: Optional[str] = 'utf-8',
        max_failures: int = 3,
        retry_delay: int = 10,
        system_prompt_class=None,
        max_input_tokens: int = 128000,
        validate_output: bool = False,
        message_context: Optional[str] = None,
        generate_gif: bool | str = True,
        include_attributes: list[str] = [
            "title",
            "type",
            "name",
            "role",
            "tabindex",
            "aria-label",
            "placeholder",
            "value",
            "alt",
            "aria-expanded",
        ],
        max_error_length: int = 400,
        max_actions_per_step: int = 10,
        tool_call_in_content: bool = True,
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        register_new_step_callback: Callable[[BrowserContext, AgentOutput, int], None] | None = None,
        register_done_callback: Callable[[AgentHistoryList], None] | None = None,
        tool_calling_method: Optional[str] = 'auto',
        user_profile: Optional[str] = '',
        # Gemini parameters
        gemini_api_key: Optional[str] = None,
        gemini_model_name: str = "gemini-2.0-flash-exp",
        gemini_temperature: float = 1.0,
        agent_state: AgentState = None,
    ):
        """
        Initializes the CustomAgent with Gemini-specific configurations.
        """
        super().__init__(
            task=task,
            llm=llm,  # Even if None, pass it to match base class
            browser=browser,
            browser_context=browser_context,
            controller=controller,
            use_vision=use_vision,
            save_conversation_path=save_conversation_path,
            save_conversation_path_encoding=save_conversation_path_encoding,
            max_failures=max_failures,
            retry_delay=retry_delay,
            system_prompt_class=system_prompt_class,
            max_input_tokens=max_input_tokens,
            validate_output=validate_output,
            message_context=message_context,
            generate_gif=generate_gif,
            include_attributes=include_attributes,
            max_error_length=max_error_length,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content,
            initial_actions=initial_actions,
            register_new_step_callback=register_new_step_callback,
            register_done_callback=register_done_callback,
            tool_calling_method=tool_calling_method,
        )

        # Additional Gemini-specific configurations
        self.add_infos = add_infos
        self.agent_state = agent_state or AgentState()

        # Set up Google GenAI client
        api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("No Google API key found. Provide gemini_api_key or set GOOGLE_API_KEY env.")
        self.gemini_client = genai.Client(api_key=api_key)

        # Gemini model configuration
        self.gemini_model_name = gemini_model_name
        self.gemini_temperature = gemini_temperature

        # Initialize CustomMessageManager with llm=None since we're using genai directly
        self.message_manager = CustomMessageManager(
            llm=None,
            task=self.task,
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=system_prompt_class,
            max_input_tokens=self.max_input_tokens,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            max_actions_per_step=self.max_actions_per_step,
            message_context=message_context,
            user_profile=user_profile,
        )

    def _setup_action_models(self) -> None:
        """
        Sets up dynamic action models from the controller's registry and wraps them in CustomAgentOutput.
        """
        self.ActionModel = self.controller.registry.create_action_model()
        self.AgentOutput = CustomAgentOutput.type_with_custom_actions(self.ActionModel)

    def _log_response(self, response: CustomAgentOutput) -> None:
        """
        Logs the agent's response in a structured format.
        """
        if "Success" in response.current_state.prev_action_evaluation:
            emoji = "✅"
        elif "Failed" in response.current_state.prev_action_evaluation:
            emoji = "❌"
        else:
            emoji = "🤷"

        logger.info(f"{emoji} Eval: {response.current_state.prev_action_evaluation}")
        logger.info(f"🧠 New Memory: {response.current_state.important_contents}")
        logger.info(f"⏳ Task Progress: \n{response.current_state.task_progress}")
        logger.info(f"📋 Future Plans: \n{response.current_state.future_plans}")
        logger.info(f"🤔 Thought: {response.current_state.thought}")
        logger.info(f"🎯 Summary: {response.current_state.summary}")
        for i, action in enumerate(response.action):
            logger.info(
                f"🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}"
            )

    def update_step_info(
        self,
        model_output: CustomAgentOutput,
        step_info: CustomAgentStepInfo = None
    ):
        """
        Updates the step information based on the agent's output.
        """
        if step_info is None:
            return

        step_info.step_number += 1

    def _unwrap_output(self, data):
        """
        Recursively unwrap dictionaries of the form {"type": "string", "value": <str>}
        into a plain string.
        """
        if isinstance(data, dict):
            if set(data.keys()) == {"type", "value"} and data.get("type") == "string":
                return data["value"]
            return {k: self._unwrap_output(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._unwrap_output(item) for item in data]
        else:
            return data 

    def _data_url_to_part(self, data_url: str) -> types.Part:
        """
        Converts a data URL to a GenAI Part object for image handling.
        """
        if not data_url.startswith("data:"):
            # Fallback if the data URL is unexpected
            return types.Part(content=data_url)

        header, b64_data = data_url.split(",", 1)
        mime_type = header.replace("data:", "").replace(";base64", "")  # e.g., 'image/png'
        raw_bytes = base64.b64decode(b64_data)
        return types.Part.from_bytes(data=raw_bytes, mime_type=mime_type)

    def _build_contents_array(self, prompt_messages: List[BaseMessage]) -> List[str | types.Part]:
        """
        Builds the contents array from prompt messages for the GenAI client.
        """

        if len(prompt_messages) > 12:
            prompt_messages = prompt_messages[:3] + prompt_messages[-9:]

        contents_list = []
        for msg in prompt_messages:
            # Access the .message attribute for ManagedMessage
            if isinstance(msg, BaseMessage):
                actual_msg = msg
            else:
                actual_msg = msg.message  # Access the BaseMessage from ManagedMessage

            if isinstance(actual_msg.content, list):
                for item in actual_msg.content:
                    if item.get("type") == "text":
                        contents_list.append(item["text"])
                    elif item.get("type") == "image_url":
                        image_part = self._data_url_to_part(item["image_url"]["url"])
                        contents_list.append(image_part)
                    else:
                        contents_list.append(str(item))
            else:
                # Fallback for raw text
                contents_list.append(actual_msg.content)

        # Remove empty or None items
        contents_list = list(filter(lambda item: item != '' and item is not None, contents_list))
        return contents_list

    def _gemini_generate_content(self, prompt_messages: List[BaseMessage]) -> str:
        """
        Generates content using the Gemini model based on the prompt messages.
        """
        contents_list = self._build_contents_array(prompt_messages)

        config = types.GenerateContentConfig(
            temperature=self.gemini_temperature,
            top_p=0.95,
            top_k=40,
            candidate_count=1,
            max_output_tokens=8192,
            stop_sequences=None,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model_name,
            contents=contents_list,
            config=config
        )
        return response.text

    @time_execution_async("--get_next_action")
    async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
        """
        Overrides the base Agent's get_next_action to utilize Gemini for generating content.
        Parses the response and returns a structured AgentOutput.
        """
        # Merge consecutive user messages if needed
        merged_input_messages = input_messages

        # If vision is enabled, capture and add screenshot
        if self.browser_context and self.use_vision:
            screenshot_base64 = await self.browser_context.get_screenshot_base64()
            merged_input_messages = self.message_manager.add_screenshot_to_latest_message(
                merged_input_messages, screenshot_base64
            )

        # Generate content using Gemini
        raw_response_str = self._gemini_generate_content(merged_input_messages)

        # Keep track of raw LLM text
        self.message_manager._add_message_with_tokens(BaseMessage(content=raw_response_str, type="gemini"))

        # Clean up possible code fences
        sanitized = raw_response_str.replace("```json", "").replace("```", "")
        try:
            parsed_dict = json.loads(sanitized)
            # Unwrap any nested string wrappers if present
            parsed_dict = self._unwrap_output(parsed_dict)
        except json.JSONDecodeError:
            logger.error(f"LLM response is not valid JSON: {raw_response_str}")
            raise ValueError("LLM response is not valid JSON.")

        # Insert missing fields
        parsed_dict["type"] = "CustomAgentOutput"
        parsed_dict.setdefault("current_state", {})
        parsed_dict["current_state"].setdefault("important_contents", "")
        parsed_dict["current_state"].setdefault("future_plans", "")
        parsed_dict["current_state"].setdefault("summary", "")
        parsed_dict["current_state"].setdefault("prev_action_evaluation", "")
        parsed_dict["current_state"].setdefault("task_progress", "")
        parsed_dict["current_state"].setdefault("thought", "")
        parsed_dict["current_state"].setdefault("next_goal", "")

        # Convert to typed AgentOutput
        try:
            parsed_output: AgentOutput = self.AgentOutput(**parsed_dict)
        except Exception as e:
            logger.error(f"Failed to parse CustomAgentOutput: {e}")
            raise ValueError("Failed to parse CustomAgentOutput.")

        # Limit actions if needed
        parsed_output.action = parsed_output.action[: self.max_actions_per_step]
        self._log_response(parsed_output)
        self.n_steps += 1

        return parsed_output

    async def run_stream(self, max_steps: int = 10):
        """
        Executes the agent's tasks step-by-step, yielding partial results for streaming.
        """
        logger.info(f"Starting streaming run for up to {max_steps} steps.")
        self.n_steps = 0
        self._setup_action_models()
        step_info = CustomAgentStepInfo(
            step_number=0,
            max_steps=max_steps,
            task=self.task,
            add_infos=self.add_infos
        )

        for step_i in range(max_steps):
            if self.agent_state.is_stop_requested():
                logger.info("User requested stop.")
                yield {
                    "step": step_i,
                    "thoughts": "Stopped by user request.",
                    "done": True,
                    "final_result": "",
                    "actions": [],  # No actions when stopped
                }
                break
                # Inject any mid–task chat messages submitted by the user.
            pending_chats = self.agent_state.get_pending_chat_messages()
            for chat in pending_chats:
                from langchain_core.messages import HumanMessage
                self.message_manager._add_message_with_tokens(HumanMessage(content=chat))

            input_messages = self.message_manager.build_message_list()

            # If vision is enabled, capture and add screenshot
            if self.browser_context and self.use_vision:
                screenshot_base64 = await self.browser_context.get_screenshot_base64()
                input_messages = self.message_manager.add_screenshot_to_latest_message(
                    input_messages, screenshot_base64
                )

            try:
                # Get the next action from the LLM
                model_output: CustomAgentOutput = await self.get_next_action(input_messages)
            except ValueError as e:
                logger.error(f"Error during get_next_action: {e}")
                yield {
                    "step": step_i + 1,
                    "thoughts": "An error occurred while processing your request.",
                    "done": True,
                    "final_result": "",
                    "actions": [],
                }
                break

            # Update step information
            self.update_step_info(model_output, step_info)

            try:
                # Execute the extracted actions
                action_results, is_done = await self.controller.execute_actions(
                    model_output.action,
                    browser_context=self.browser_context,
                    step_info=step_info,
                )
            except Exception as e:
                logger.error(f"Error executing actions: {e}")
                yield {
                    "step": step_i + 1,
                    "thoughts": "An error occurred while executing actions.",
                    "done": True,
                    "final_result": "",
                    "actions": [],
                }
                break

            # Add the new browser state to the conversation:
            browser_state = await self.browser_context.get_state()
            self.message_manager.add_state_message(browser_state, action_results, step_info)


            # Prepare partial information to yield for streaming
            partial_info = {
                "step": step_i + 1,
                "thoughts": model_output.current_state.thought,
                "done": is_done,
                "final_result": model_output.action[-1].done.text if is_done else "",
                "actions": model_output.action
            }
            yield partial_info

            # Check if the agent has completed its task
            if is_done:
                logger.info(f"Task completed after {step_i + 1} steps.")
                break

            self.n_steps += 1

        # After completing all steps or stopping, yield the final state if not already done
        if not partial_info.get("done", False):
            yield {
                "step": self.n_steps,
                "thoughts": "Reached maximum steps. Task incomplete.",
                "done": True,
                "final_result": "Task incomplete after maximum steps.",
                "actions": [],
            }
