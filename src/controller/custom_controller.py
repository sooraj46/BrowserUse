import asyncio
from asyncio.log import logger
import json
from pydantic import BaseModel
import pyperclip
import pyautogui
import platform
from typing import List, Optional, Type
from browser_use.agent.views import ActionResult, AgentOutput
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller
from browser_use.controller.registry.service import ActionRegistry

from src.utils.agent_state import AgentState

from ..agent.custom_views import (
    CustomActionResult,
    CustomAgentStepInfo,
    DownloadFileAction,
    UploadFileAction,
    ExecuteJavascriptAction,
    TakeScreenshotAction,
    WaitAction,
    MouseHoverAction,
    ExtractTableDataAction,
    ExtractListItemsAction,
    GetElementTextAction,
    SetCheckboxStateAction,
    SelectRadioButtonAction,
    SetDateInputAction,
    GoForwardAction,
    ReloadPageAction,
    CloseTabAction,
    GetElementAttributeAction,
    GetElementBoundingBoxAction,
    IsElementVisibleAction,
    GetCookiesAction,
    SwitchToFrameAction,
    HandleAlertAction,
    PauseForHumanInputAction, 
    WriteFileAction,
    AppendFileAction,
    ReadFileAction
)

class CustomController(Controller):
    def __init__(self, exclude_actions: List[str] = [],
                output_model: Optional[Type[BaseModel]] = None,
                agent_state: Optional[AgentState] = None):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()
        self.agent_state = agent_state or AgentState()
        

    async def execute_actions(
        self,
        actions: List[BaseModel],
        browser_context: BrowserContext,
        step_info: Optional[CustomAgentStepInfo] = None,
        ) -> tuple[List[CustomActionResult], bool]:
        action_results: List[CustomActionResult] = []
        is_done = False

        for action in actions:
            if self.agent_state.is_stop_requested():
                logger.info("Stopping further actions execution as requested by user.")
                break

            if self.agent_state.is_pause_requested():
                logger.info("Pausing further actions execution as requested by user.")
                action_results.append(CustomActionResult(
                    success=False,
                    error="Agent execution paused. Waiting for user input or resume command."
                ))
                break

            try:
                # Instead of action.action_name, detect which field is set
                action_name = None
                params = None
                for field_name, field_value in action.__dict__.items():
                    if field_value is not None:
                        action_name = field_name  # e.g. "done"
                        params = field_value      # e.g. DoneAction(text="Hi")
                        break

                if not action_name:
                    raise ValueError(f"No recognized action in {action}")

                # Call the action registry with that name and param
                action_result = await self.registry.execute_action(
                    action_name,
                    params.dict() if hasattr(params, "dict") else params,
                    browser_context
                )

                # Convert ActionResult to CustomActionResult
                custom_action_result = CustomActionResult(
                    success=None,
                    error=action_result.error,
                    extracted_content=action_result.extracted_content,
                    include_in_memory=action_result.include_in_memory,
                    done=action_result.is_done,   # default
                    stop_all=False
                )

            except Exception as e:
                logger.exception(f"Error executing {action=}: {e}")
                custom_action_result = CustomActionResult(success=False, error=str(e))

            action_results.append(custom_action_result)

            if custom_action_result.stop_all:
                logger.info("Stopping all actions as requested by action result.")
                break
            if custom_action_result.done:
                is_done = True
                break

        # Update the agent_state with the last valid state
        self.agent_state.set_last_valid_state(browser_context)
        return action_results, is_done

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action("Copy text to clipboard")
        def copy_to_clipboard(text: str):
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)

        @self.registry.action("""Paste text from clipboard.
                              - Before pasting target input field should be on focus""", requires_browser=True)
        async def paste_from_clipboard(browser: BrowserContext):
            text = pyperclip.paste()
            page = await browser.get_current_page()
            system = platform.system()
            if system == "Windows":
                pyautogui.hotkey("ctrl", "v")
            elif system == "Darwin":  # macOS
                pyautogui.hotkey("command", "v")
            elif system == "Linux":
                pyautogui.hotkey("ctrl", "v")
            else:
                raise RuntimeError("Unsupported platform")

        @self.registry.action("Download a file from a given URL", param_model=DownloadFileAction, requires_browser=True)
        async def download_file(params: DownloadFileAction, browser: BrowserContext):
            page = await browser.get_current_page()
            await page.goto(params.url)
            return ActionResult(extracted_content=f"Downloading file from {params.url}")

        @self.registry.action("Upload a file to an input field", param_model=UploadFileAction, requires_browser=True)
        async def upload_file(params: UploadFileAction, browser: BrowserContext):
            page = await browser.get_current_page()
            input_element = await page.query_selector(params.selector)
            await input_element.set_input_files(params.file_path)
            return ActionResult(success=True)

        @self.registry.action("Get current page URL", requires_browser=True)
        async def get_current_url( browser: BrowserContext):
            page = await browser.get_current_page()
            return ActionResult(extracted_content=page.url)

        @self.registry.action("Get current page title",  requires_browser=True)
        async def get_page_title( browser: BrowserContext):
            page = await browser.get_current_page()
            return ActionResult(extracted_content=await page.title())

        @self.registry.action("Execute JavaScript code", param_model=ExecuteJavascriptAction, requires_browser=True)
        async def execute_javascript(params: ExecuteJavascriptAction, browser: BrowserContext):
            page = await browser.get_current_page()
            result = await page.evaluate(params.script)
            return ActionResult(extracted_content=str(result))

        @self.registry.action("Take a screenshot", param_model=TakeScreenshotAction, requires_browser=True)
        async def take_screenshot(params: TakeScreenshotAction, browser: BrowserContext):
            page = await browser.get_current_page()
            path = params.path or "screenshot.png"
            await page.screenshot(path=path)
            msg = f"Screenshot saved at {path}"
            return ActionResult(extracted_content=msg)

        @self.registry.action("Wait for a specific time", param_model=WaitAction)
        async def wait_action(params: WaitAction):
            await asyncio.sleep(params.seconds)
            return ActionResult(success=True)

        @self.registry.action("Mouse hover over an element", param_model=MouseHoverAction, requires_browser=True)
        async def mouse_hover(params: MouseHoverAction, browser: BrowserContext):
            session = await browser.get_session()
            state = session.cached_state

            if params.index not in state.selector_map:
                raise Exception(f"Element with index {params.index} does not exist - retry or use alternative actions")

            element_node = state.selector_map[params.index]

            try:
                page = await browser.get_current_page()
                element_handle = await page.query_selector(element_node.xpath)

                if not element_handle:
                    return ActionResult(error=f"Element with index {params.index} not found on the page.")

                await element_handle.hover()
                msg = f"🖱️ Hovered over element with index {params.index}: {element_node.get_all_text_till_next_clickable_element(max_depth=2)}"
                logger.info(msg)
                logger.debug(f"Element xpath: {element_node.xpath}")
                return ActionResult(success=True, extracted_content=msg)

            except Exception as e:
                logger.warning(f"Element not hoverable with index {params.index} - most likely the page changed")
                return ActionResult(error=str(e))

        @self.registry.action("Extract table data", param_model=ExtractTableDataAction, requires_browser=True)
        async def extract_table_data(params: ExtractTableDataAction, browser: BrowserContext):
            page = await browser.get_current_page()
            rows = await page.eval_on_selector_all(params.selector, "elements => elements.map(e => e.innerText)")
            return ActionResult(extracted_content=str(rows))

        @self.registry.action("Extract list items", param_model=ExtractListItemsAction, requires_browser=True)
        async def extract_list_items(params: ExtractListItemsAction, browser: BrowserContext):
            page = await browser.get_current_page()
            items = await page.eval_on_selector_all(params.selector, "elements => elements.map(e => e.innerText)")
            return ActionResult(extracted_content=str(items))

        @self.registry.action("Get element text", param_model=GetElementTextAction, requires_browser=True)
        async def get_element_text(params: GetElementTextAction, browser: BrowserContext):
            page = await browser.get_current_page()
            element = await page.query_selector(params.selector)
            if not element:
                return ActionResult(error=f"Selector not found: {params.selector}")
            text = await element.inner_text()
            return ActionResult(extracted_content=text)

        @self.registry.action("Set checkbox state", param_model=SetCheckboxStateAction, requires_browser=True)
        async def set_checkbox_state(params: SetCheckboxStateAction, browser: BrowserContext):
            page = await browser.get_current_page()
            checkbox = await page.query_selector(params.selector)
            if not checkbox:
                return ActionResult(error=f"Checkbox not found: {params.selector}")
            await checkbox.set_checked(params.checked)
            return ActionResult(success=True)

        @self.registry.action("Select radio button", param_model=SelectRadioButtonAction, requires_browser=True)
        async def select_radio_button(params: SelectRadioButtonAction, browser: BrowserContext):
            page = await browser.get_current_page()
            radio_button = await page.query_selector(params.selector)
            if not radio_button:
                return ActionResult(error=f"Radio button not found: {params.selector}")
            await radio_button.click()
            return ActionResult(success=True)

        @self.registry.action("Set date input field", param_model=SetDateInputAction, requires_browser=True)
        async def set_date_input(params: SetDateInputAction, browser: BrowserContext):
            page = await browser.get_current_page()
            input_element = await page.query_selector(params.selector)
            if not input_element:
                return ActionResult(error=f"Date input not found: {params.selector}")
            await input_element.fill(params.date)
            return ActionResult(success=True)

        @self.registry.action("Go forward in browser history", param_model=GoForwardAction, requires_browser=True)
        async def go_forward(_, browser: BrowserContext):
            page = await browser.get_current_page()
            await page.go_forward()
            return ActionResult(success=True)

        @self.registry.action("Reload the page", param_model=ReloadPageAction, requires_browser=True)
        async def reload_page(_, browser: BrowserContext):
            page = await browser.get_current_page()
            await page.reload()
            return ActionResult(success=True)

        @self.registry.action("Close current tab", param_model=CloseTabAction, requires_browser=True)
        async def close_tab(_, browser: BrowserContext):
            page = await browser.get_current_page()
            await page.close()
            return ActionResult(success=True)

        @self.registry.action("Get element attribute", param_model=GetElementAttributeAction, requires_browser=True)
        async def get_element_attribute(params: GetElementAttributeAction, browser: BrowserContext):
            page = await browser.get_current_page()
            element = await page.query_selector(params.selector)
            if not element:
                return ActionResult(error=f"Selector not found: {params.selector}")
            value = await element.get_attribute(params.attribute)
            return ActionResult(extracted_content=str(value))

        @self.registry.action("Get element bounding box", param_model=GetElementBoundingBoxAction, requires_browser=True)
        async def get_element_bounding_box(params: GetElementBoundingBoxAction, browser: BrowserContext):
            page = await browser.get_current_page()
            element = await page.query_selector(params.selector)
            if not element:
                return ActionResult(error=f"Selector not found: {params.selector}")
            bbox = await element.bounding_box()
            return ActionResult(extracted_content=json.dumps(bbox))

        @self.registry.action("Check if an element is visible", param_model=IsElementVisibleAction, requires_browser=True)
        async def is_element_visible(params: IsElementVisibleAction, browser: BrowserContext):
            page = await browser.get_current_page()
            element = await page.query_selector(params.selector)
            if not element:
                return ActionResult(error=f"Selector not found: {params.selector}")
            visible = await element.is_visible()
            return ActionResult(extracted_content=str(visible))

        @self.registry.action("Get cookies", param_model=GetCookiesAction, requires_browser=True)
        async def get_cookies(_, browser: BrowserContext):
            context = browser.context
            cookies = await context.cookies()
            return ActionResult(extracted_content=json.dumps(cookies))

        @self.registry.action("Switch to frame", param_model=SwitchToFrameAction, requires_browser=True)
        async def switch_to_frame(params: SwitchToFrameAction, browser: BrowserContext):
            page = await browser.get_current_page()
            frame = page.frame_locator(params.selector)
            if not frame:
                return ActionResult(error=f"Frame not found by selector: {params.selector}")
            return ActionResult(success=True)

        @self.registry.action("Handle browser alert", param_model=HandleAlertAction, requires_browser=True)
        async def handle_alert(params: HandleAlertAction, browser: BrowserContext):
            page = await browser.get_current_page()
            dialog_event = None
            try:
                dialog_event = await page.wait_for_event("dialog", timeout=3000)
            except:
                return ActionResult(error="No dialog appeared.")

            if params.action.lower() == "accept":
                await dialog_event.dialog.accept()
            else:
                await dialog_event.dialog.dismiss()
            return ActionResult(success=True)

        @self.registry.action("Pause for user input", param_model=PauseForHumanInputAction)
        def pause_for_user_input(params: PauseForHumanInputAction):
            
            return CustomActionResult(
                success=True,
                extracted_content=f"Pausing for user input. Prompt: {params.message}"
            )

        @self.registry.action("Write to a text file", param_model=WriteFileAction)
        def write_text_file(params: WriteFileAction):
            with open(params.file_path, "w", encoding="utf-8") as f:
                f.write(params.content)
            return ActionResult(success=True, extracted_content=f"Wrote to file {params.file_path}")

        @self.registry.action("Append text to a file", param_model=AppendFileAction)
        def append_text_file(params: AppendFileAction):
            with open(params.file_path, "a", encoding="utf-8") as f:
                f.write(params.content)
            return ActionResult(success=True, extracted_content=f"Appended to file {params.file_path}")

        @self.registry.action("Read text file", param_model=ReadFileAction)
        def read_text_file(params: ReadFileAction):
            try:
                with open(params.file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return ActionResult(success=True, extracted_content=content)
            except Exception as e:
                return ActionResult(success=False, error=str(e))