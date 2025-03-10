from dataclasses import dataclass
from typing import Any, Optional, Type

from browser_use.agent.views import AgentOutput
from browser_use.controller.registry.views import ActionModel
from pydantic import BaseModel, ConfigDict, Field, create_model
from browser_use.agent.views import ActionResult


@dataclass
class CustomAgentStepInfo:
    step_number: int
    max_steps: int
    task: str
    add_infos: str

class ActionModel(BaseModel):
    action_name: str
    params: dict
class CustomAgentBrain(BaseModel):
    prev_action_evaluation: str
    important_contents: str
    task_progress: str
    future_plans: str
    thought: str
    summary: str
    next_goal: str


class DownloadFileAction(BaseModel):
    url: str


class UploadFileAction(BaseModel):
    selector: str
    file_path: str

class WriteFileAction(BaseModel):
    file_path: str
    content: str

class AppendFileAction(BaseModel):
    file_path: str
    content: str

class ReadFileAction(BaseModel):
    file_path: str

class ExecuteJavascriptAction(BaseModel):
    script: str


class TakeScreenshotAction(BaseModel):
    path: str


class WaitAction(BaseModel):
    seconds: int


class MouseHoverAction(BaseModel):
    index: int
    xpath: Optional[str] = None


class ExtractTableDataAction(BaseModel):
    selector: str


class ExtractListItemsAction(BaseModel):
    selector: str


class GetElementTextAction(BaseModel):
    selector: str


class SetCheckboxStateAction(BaseModel):
    selector: str
    checked: bool


class SelectRadioButtonAction(BaseModel):
    selector: str


class SetDateInputAction(BaseModel):
    selector: str
    date: str


class GoForwardAction(BaseModel):
    pass  # No parameters required


class ReloadPageAction(BaseModel):
    pass  # No parameters required


class CloseTabAction(BaseModel):
    pass  # No parameters required


class GetElementAttributeAction(BaseModel):
    selector: str
    attribute: str


class GetElementBoundingBoxAction(BaseModel):
    selector: str


class IsElementVisibleAction(BaseModel):
    selector: str


class GetCookiesAction(BaseModel):
    pass  # No parameters required


class SwitchToFrameAction(BaseModel):
    selector: str


class HandleAlertAction(BaseModel):
    action: str  



class PauseForHumanInputAction(BaseModel):
    message: str


class CustomAgentOutput(AgentOutput):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_state: CustomAgentBrain
    action: list[ActionModel]

    @staticmethod
    def type_with_custom_actions(
        custom_actions: Type[ActionModel],
    ) -> Type["CustomAgentOutput"]:
        """Extend actions with custom actions"""
        return create_model(
            "CustomAgentOutput",
            __base__=CustomAgentOutput,
            action=(
                list[custom_actions],
                Field(...),
            ),  # Properly annotated field with no default
            __module__=CustomAgentOutput.__module__,
        )
    
class CustomActionResult(ActionResult):
    """
    Represents the result of an action execution, extending ActionResult.
    """

    success: Optional[bool] = None
    """Indicates if the action was successful."""

    error: Optional[str] = None
    """Error message if the action failed."""

    extracted_content: Optional[str] = None
    """Extracted content from the action."""

    include_in_memory: bool = True
    """Whether to include the action in memory."""

    done: bool = False
    """Indicates if the task is considered done after this action."""

    stop_all: bool = False
    """Indicates if all further actions should be stopped."""