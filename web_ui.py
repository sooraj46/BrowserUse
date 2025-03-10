import pdb
import logging
import os
import glob
import asyncio
import argparse

import gradio as gr
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from src.utils.agent_state import AgentState
from src.agent.custom_views import CustomAgentStepInfo
from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt
from src.browser.custom_context import CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.default_config_settings import (
    default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config
)
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot

load_dotenv()

logger = logging.getLogger(__name__)

_global_browser = None
_global_browser_context = None
_global_agent_state = AgentState()
_conversation_browser_initialized = False
_global_agent = None


async def stop_agent():
    """
    Allows user to manually request the agent to stop.
    """
    global _global_agent_state, _global_browser_context, _global_browser
    try:
        _global_agent_state.request_stop()
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        return (
            message,                                        # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),                      # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            error_msg,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method
):
    """
    Example for an 'org' agent. (No longer actively used, but code kept for reference.)
    """

    global _global_browser, _global_browser_context, _global_agent_state
    _global_agent_state.clear_stop()

    # Basic browser setup
    if use_own_browser:
        chrome_path = os.getenv("CHROME_PATH", None)
        if chrome_path == "":
            chrome_path = None
    else:
        chrome_path = None

    if _global_browser is None:
        _global_browser = Browser(
            config=BrowserConfig(
                headless=headless,
                disable_security=disable_security,
                chrome_instance_path=chrome_path,
                extra_chromium_args=[f"--window-size={window_w},{window_h}"],
            )
        )

    if _global_browser_context is None:
        _global_browser_context = await _global_browser.new_context(
            config=BrowserContextConfig(
                trace_path=save_trace_path if save_trace_path else None,
                save_recording_path=save_recording_path if save_recording_path else None,
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
            )
        )

    agent = Agent(
        task=task,
        llm=llm,
        use_vision=use_vision,
        browser=_global_browser,
        browser_context=_global_browser_context,
        max_actions_per_step=max_actions_per_step,
        tool_calling_method=tool_calling_method
    )

    # Run the agent
    history = await agent.run(max_steps=max_steps)

    # Save agent history
    import os
    history_file = os.path.join(save_agent_history_path, f"{agent.agent_id}.json")
    agent.save_history(history_file)

    # Collect results
    final_result = history.final_result()
    errors = history.errors()
    model_actions = history.model_actions()
    model_thoughts = history.model_thoughts()

    # Attempt to find trace file if any
    trace_file = get_latest_files(save_trace_path)

    # Cleanup if we aren't keeping the browser open
    if not keep_browser_open:
        if _global_browser_context:
            await _global_browser_context.close()
        if _global_browser:
            await _global_browser.close()
        _global_browser_context = None
        _global_browser = None

    # Return results
    return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file

async def run_custom_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        user_profile
):
    
    global _global_browser, _global_browser_context, _global_agent_state
    _global_agent_state.clear_stop()

    if use_own_browser:
        chrome_path = os.getenv("CHROME_PATH", None)
        if chrome_path == "":
            chrome_path = None
    else:
        chrome_path = None

    controller = CustomController()

    if _global_browser is None:
        _global_browser = CustomBrowser(
            config=BrowserConfig(
                headless=headless,
                disable_security=disable_security,
                chrome_instance_path=chrome_path,
                extra_chromium_args=[f"--window-size={window_w},{window_h}"],
            )
        )

    if _global_browser_context is None:
        _global_browser_context = await _global_browser.new_context(
            config=BrowserContextConfig(
                trace_path=save_trace_path if save_trace_path else None,
                save_recording_path=save_recording_path if save_recording_path else None,
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
            )
        )

    agent = CustomAgent(
        task=task,
        add_infos=add_infos,
        use_vision=use_vision,
        browser=_global_browser,
        browser_context=_global_browser_context,
        controller=controller,
        system_prompt_class=CustomSystemPrompt,
        max_actions_per_step=max_actions_per_step,
        agent_state=_global_agent_state,
        tool_calling_method=tool_calling_method,
        user_profile=user_profile
    )

    global _global_agent
    _global_agent = agent
    
    history = await agent.run(max_steps=max_steps)
    
    history_file = os.path.join(save_agent_history_path, f"{agent.agent_id}.json")
    agent.save_history(history_file)

    final_result = history.final_result()
    errors = history.errors()
    model_actions = history.model_actions()
    model_thoughts = history.model_thoughts()

    trace_file = get_latest_files(save_trace_path)

    if not keep_browser_open:
        if _global_browser_context:
            await _global_browser_context.close()
        if _global_browser:
            await _global_browser.close()
        _global_browser_context = None
        _global_browser = None

    _global_agent = None
    return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file

async def run_browser_agent(
    agent_type,
    llm_model_name,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method,
    profile_info
):

    from src.utils import utils

    if not enable_recording:
        save_recording_path = None
    if save_recording_path:
        os.makedirs(save_recording_path, exist_ok=True)

    existing_videos = set()
    if save_recording_path:
        existing_videos = set(
            glob.glob(os.path.join(save_recording_path, "*.mp4"))
            + glob.glob(os.path.join(save_recording_path, "*.webm"))
        )

    # Build the LLM
    llm = utils.get_llm_model(
        provider="gemini",  # Hardcoded to gemini in this example
        model_name=llm_model_name,
        temperature=llm_temperature,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )

    if agent_type == "org":
        final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
            llm=llm,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            task=task,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method
        )
    else:
        # default = 'custom'
        final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
            llm=llm,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            user_profile= profile_info
        )

    # Find newly created video
    latest_video = None
    if save_recording_path:
        new_videos = set(
            glob.glob(os.path.join(save_recording_path, "*.mp4"))
            + glob.glob(os.path.join(save_recording_path, "*.webm"))
        )
        diff = new_videos - existing_videos
        if diff:
            latest_video = list(diff)[0]

    return (
        final_result,
        errors,
        model_actions,
        model_thoughts,
        latest_video,
        trace_file,
        history_file,
        gr.update(value="Stop", interactive=True),
        gr.update(interactive=True),
    )

async def run_with_stream(
    agent_type,
    llm_model_name,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method,
    profile_info
):

    global _global_agent_state
    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)

    _global_agent_state.clear_stop()
    agent_task = asyncio.create_task(
        run_browser_agent(
            agent_type,
            llm_model_name,
            llm_temperature,
            llm_base_url,
            llm_api_key,
            use_own_browser,
            keep_browser_open,
            headless,
            disable_security,
            window_w,
            window_h,
            save_recording_path,
            save_agent_history_path,
            save_trace_path,
            enable_recording,
            task,
            add_infos,
            max_steps,
            use_vision,
            max_actions_per_step,
            tool_calling_method,
            profile_info=profile_info
        )
    )

    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
    final_result = errors = model_actions = model_thoughts = ""
    latest_videos = trace = history_file = None

    while not agent_task.done():
        try:
            encoded_screenshot = await capture_screenshot(_global_agent_state.last_valid_state or _global_browser_context)
            if encoded_screenshot is not None:
                html_content = (
                    f'<img src="data:image/jpeg;base64,{encoded_screenshot}" '
                    f'style="width:{stream_vw}vw; height:{stream_vh}vh; border:1px solid #ccc;">'
                )
            else:
                html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
        except Exception:
            html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"

        if _global_agent_state.is_stop_requested():
            yield [
                html_content,
                final_result,
                errors + "\nUser requested STOP.",
                model_actions,
                model_thoughts,
                latest_videos,
                trace,
                history_file,
                gr.update(value="Stopping...", interactive=False),
                gr.update(interactive=False),
            ]
            break

        yield [
            html_content,
            final_result,
            errors,
            model_actions,
            model_thoughts,
            latest_videos,
            trace,
            history_file,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True),
        ]
        await asyncio.sleep(0.2)

    # Once done, gather final
    try:
        result = await agent_task
        (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            latest_videos,
            trace,
            history_file,
            stop_button,
            run_button
        ) = result
    except Exception as e:
        errors = f"Agent error: {str(e)}"

    # If needed, check if the agent is waiting for user input
    if "human_input_needed" in errors or "pause_reason" in errors:
        errors += "\n**Agent is waiting for your input.**"

    yield [
        html_content,
        final_result,
        errors,
        model_actions,
        model_thoughts,
        latest_videos,
        trace,
        history_file,
        gr.update(value="Stop", interactive=True),
        gr.update(interactive=True),
    ]



async def agent_respond_stream(
    conv_state,
    user_input,
    agent_type,
    llm_model_name,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method
):
    """
    This function responds to a chat message from the user.
    If an agent is already running, the new user input is added to its state,
    so that the message is injected mid–task. Otherwise, a new agent is started.
    Once a task has finished (final result output), the global agent is cleared
    so that a new task can be started on subsequent input.
    """
    user_message = user_input.strip()
    if not user_message:
        yield conv_state, gr.update(value="")
        return

    global _global_agent
    # If an agent is already running, add the new chat message to its state.
    if _global_agent is not None:
        # (If the agent has finished its run, _global_agent should have been cleared.
        # Otherwise, this message is treated as mid–task input.)
        _global_agent.agent_state.add_chat_message(user_message)
        conv_state.append({"role": "user", "content": user_message})
        yield conv_state, gr.update(value="")
        return

    # Otherwise, start a new conversation.
    conv_state.append({"role": "user", "content": user_message})
    yield conv_state, gr.update(value="")

    # Initialize global browser and context if not already initialized.
    global _global_browser, _global_browser_context, _conversation_browser_initialized, _global_agent_state
    if not _conversation_browser_initialized:
        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
        else:
            chrome_path = None

        if _global_browser is None:
            from src.browser.custom_browser import CustomBrowser
            from browser_use.browser.browser import BrowserConfig
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=[f"--window-size={window_w},{window_h}"],
                )
            )

        if _global_browser_context is None:
            # Import BrowserContextWindowSize from the correct package.
            from src.browser.custom_context import BrowserContextConfig
            from browser_use.browser.context import BrowserContextWindowSize
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path,
                    save_recording_path=save_recording_path,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
                )
            )
        _conversation_browser_initialized = True  # Mark that initialization is done.

    # Create a controller and new agent for this conversation.
    controller = CustomController(agent_state=_global_agent_state)
    agent = CustomAgent(
        task=user_message,  # Use the user message as the initial task.
        browser=_global_browser,
        browser_context=_global_browser_context,
        controller=controller,
        system_prompt_class=CustomSystemPrompt,
        max_actions_per_step=max_actions_per_step,
        agent_state=_global_agent_state,
        tool_calling_method=tool_calling_method,
    )
    _global_agent = agent  # Save the running agent globally.

    agent_response_text = ""  # Accumulate the agent's responses.
    try:
        async for partial_info in agent.run_stream(max_steps=max_steps):
            agent_thoughts = partial_info.get("thoughts", "")
            done_flag = partial_info.get("done", False)
            step_output = ""
            if agent_thoughts:
                step_output = f"**Thought**: {agent_thoughts}\n\n"
            if step_output:
                agent_response_text = step_output
            conv_state.append({"role": "assistant", "content": agent_response_text})
            yield conv_state, gr.update(value="")
            if done_flag:
                final_result = partial_info.get("final_result", "")
                if final_result:
                    agent_response_text = f"\n**Final Result**: {final_result}"
                    conv_state.append({"role": "assistant", "content": agent_response_text})
                break
    except Exception as e:
        if "Timeout" in str(e):
            error_message = "Error: Screenshot timed out. Please try again later."
        else:
            error_message = f"An unexpected error occurred: {str(e)}"
        conv_state.append({"role": "assistant", "content": error_message})
        yield conv_state, gr.update(value="")

    # Clear the global agent variable after completion so that new input will start a new task.
    _global_agent = None
    yield conv_state, gr.update(value="")


def close_global_browser():
    """
    We can close the global browser if the user toggles certain checkboxes.
    """
    global _global_browser, _global_browser_context ,_conversation_browser_initialized
    try:
        if _global_browser_context:
            asyncio.run(_global_browser_context.close())
            _global_browser_context = None
        if _global_browser:
            asyncio.run(_global_browser.close())
            _global_browser = None
    except Exception as e:
        _conversation_browser_initialized = False

def create_ui(config, theme_name="Ocean"):
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        padding-top: 20px !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 30px;
    }
    """

    # You can pick a theme from [Default, Soft, Monochrome, Glass, Origin, Citrus, Ocean, Base]
    chosen_theme = {
        "Default": Default(),
        "Soft": Soft(),
        "Monochrome": Monochrome(),
        "Glass": Glass(),
        "Origin": Origin(),
        "Citrus": Citrus(),
        "Ocean": Ocean(),
        "Base": Base()
    }.get(theme_name, Ocean())

    with gr.Blocks(title="Browser Use WebUI", theme=chosen_theme, css=css) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🌐 Browser Use WebUI
                ### Control your browser with AI assistance
                """,
                elem_classes=["header-text"],
            )

        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Agent Settings"):
                with gr.Group():
                    agent_type = gr.Radio(
                        ["custom", "org"],
                        label="Agent Type",
                        value=config['agent_type'],
                        info="Select the type of agent to use",
                    )
                    max_steps = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=config['max_steps'],
                        step=1,
                        label="Max Run Steps",
                        info="Maximum number of steps the agent will take",
                    )
                    max_actions_per_step = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=config['max_actions_per_step'],
                        step=1,
                        label="Max Actions per Step",
                        info="Maximum number of actions the agent will take per step",
                    )
                    use_vision = gr.Checkbox(
                        label="Use Vision",
                        value=config['use_vision'],
                        info="Enable visual processing capabilities",
                    )
                    tool_calling_method = gr.Dropdown(
                        label="Tool Calling Method",
                        value=config['tool_calling_method'],
                        interactive=True,
                        allow_custom_value=True,
                        choices=["auto", "json_schema", "function_calling"],
                        info="Tool Calls Funtion Name",
                        visible=False
                    )

            with gr.TabItem("🔧 LLM Configuration"):
                llm_provider = gr.Dropdown(
                    choices=["gemini"],
                    label="LLM Provider",
                    value="gemini",
                    interactive=False,
                    info="Select your preferred language model provider"
                )
                llm_model_name = gr.Dropdown(
                    label="Model Name",
                    choices=utils.model_names['gemini'],
                    value=config['llm_model_name'],
                    interactive=True,
                    allow_custom_value=True,
                    info="Select a model from the dropdown or type a custom model name"
                )
                llm_temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=config['llm_temperature'],
                    step=0.1,
                    label="Temperature",
                    info="Controls randomness in model outputs"
                )
                llm_base_url = gr.Textbox(
                    label="Base URL",
                    value=config['llm_base_url'],
                    info="API endpoint URL (if required)",
                    visible=False
                )
                llm_api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    value=config['llm_api_key'],
                    info="Your API key (leave blank to use .env)"
                )

            with gr.TabItem("🌐 Browser Settings"):
                with gr.Group():
                    use_own_browser = gr.Checkbox(
                        label="Use Own Browser",
                        value=config['use_own_browser'],
                        info="Use your existing browser instance",
                    )
                    keep_browser_open = gr.Checkbox(
                        label="Keep Browser Open",
                        value=config['keep_browser_open'],
                        info="Keep Browser Open between Tasks",
                    )
                    headless = gr.Checkbox(
                        label="Headless Mode",
                        value=config['headless'],
                        info="Run browser without GUI",
                    )
                    disable_security = gr.Checkbox(
                        label="Disable Security",
                        value=config['disable_security'],
                        info="Disable browser security features",
                    )
                    enable_recording = gr.Checkbox(
                        label="Enable Recording",
                        value=config['enable_recording'],
                        info="Enable saving browser recordings",
                    )

                    window_w = gr.Number(
                        label="Window Width",
                        value=config['window_w'],
                        info="Browser window width",
                    )
                    window_h = gr.Number(
                        label="Window Height",
                        value=config['window_h'],
                        info="Browser window height",
                    )

                    save_recording_path = gr.Textbox(
                        label="Recording Path",
                        placeholder="e.g. ./tmp/record_videos",
                        value=config['save_recording_path'],
                        info="Path to save browser recordings",
                        interactive=True,
                    )

                    save_trace_path = gr.Textbox(
                        label="Trace Path",
                        placeholder="e.g. ./tmp/traces",
                        value=config['save_trace_path'],
                        info="Path to save Agent traces",
                        interactive=True,
                    )

                    save_agent_history_path = gr.Textbox(
                        label="Agent History Save Path",
                        placeholder="e.g., ./tmp/agent_history",
                        value=config['save_agent_history_path'],
                        info="Directory to store agent history JSON files",
                        interactive=True,
                    )

            with gr.TabItem("🤖 Run Agent"):
                task = gr.Textbox(
                    label="Task Description",
                    lines=4,
                    placeholder="Enter your task here...",
                    value=config['task'],
                    info="Describe what you want the agent to do",
                )
                add_infos = gr.Textbox(
                    label="Additional Information",
                    lines=3,
                    placeholder="Add any helpful context or instructions...",
                    info="Optional hints to help the LLM complete the task",
                )

                with gr.Row():
                    run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2)
                    stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)

                with gr.Row():
                    browser_view = gr.HTML(
                        value="<h1 style='width:80vw; height:50vh'>Waiting for browser session...</h1>",
                        label="Live Browser View",
                    )

            with gr.TabItem("💬 Conversational Agent"):
                chatbox = gr.Chatbot(label="Chat with the Agent", type="messages")
                user_input = gr.Textbox(label="Your Message", lines=2) 

                with gr.Row():
                    #send_button = gr.Button("Send", variant="primary") # No send button anymore
                    pause_button = gr.Button("Pause", variant="secondary")
                    resume_button = gr.Button("Resume", variant="secondary")

                conversation_state = gr.State([])  # list of (user_str, agent_str)

                # User input textbox now triggers the response on Enter press
                user_input.submit(
                    fn=agent_respond_stream,
                    inputs=[
                        conversation_state, user_input,
                        agent_type, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security,
                        window_w, window_h, save_recording_path, save_agent_history_path,
                        save_trace_path, enable_recording, max_steps, use_vision,
                        max_actions_per_step, tool_calling_method
                    ],
                    outputs=[conversation_state, user_input],
                    queue=True  # Must be True for streaming
                )

                # When conversation_state changes, update the chatbox
                conversation_state.change(lambda x: x, conversation_state, chatbox)

                # Pause/Resume
                pause_button.click(fn=lambda: _global_agent_state.request_stop(), outputs=[])
                resume_button.click(fn=lambda: _global_agent_state.clear_stop(), outputs=[])

            with gr.TabItem("👤 User Profile"):
                with gr.Group():
                    profile_name = gr.Textbox(label="Name", placeholder="Your name", lines=1)
                    profile_email = gr.Textbox(label="Email", placeholder="Your email", lines=1)
                    profile_interests = gr.Textbox(label="Interests", placeholder="Your interests", lines=2)
                    # You could add more fields as needed.
                    # Combine the fields into one profile string when saving
                    profile_info = gr.Textbox(label="Combined Profile", visible=False)

                    def combine_profile(name, email, interests):
                        # You can format this string as needed.
                        return f"Name: {name}\nEmail: {email}\nInterests: {interests}"
                    combine_button = gr.Button("Save Profile")
                    combine_button.click(fn=combine_profile,
                                         inputs=[profile_name, profile_email, profile_interests],
                                         outputs=profile_info)


            with gr.TabItem("📁 Configuration"):
                with gr.Group():
                    # Possibly hidden by default if you want:
                    config_file_input = gr.File(
                        label="Load Config File",
                        file_types=[".pkl"],
                        interactive=True
                    )
                    load_config_button = gr.Button("Load Existing Config From File", variant="primary")
                    save_config_button = gr.Button("Save Current Config", variant="primary")

                    config_status = gr.Textbox(
                        label="Status",
                        lines=2,
                        interactive=False
                    )

                load_config_button.click(
                    fn=update_ui_from_config,
                    inputs=[config_file_input],
                    outputs=[
                        agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                        llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security,
                        enable_recording, window_w, window_h, save_recording_path,
                        save_trace_path, save_agent_history_path, task, config_status
                    ]
                )

                save_config_button.click(
                    fn=save_current_config,
                    inputs=[
                        agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                        llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security,
                        enable_recording, window_w, window_h, save_recording_path, save_trace_path,
                        save_agent_history_path, task,
                    ],
                    outputs=[config_status]
                )

            with gr.TabItem("📊 Results"):
                with gr.Group():
                    recording_display = gr.Video(label="Latest Recording")
                    gr.Markdown("### Results")
                    with gr.Row():
                        with gr.Column():
                            final_result_output = gr.Textbox(
                                label="Final Result", lines=3, show_label=True
                            )
                        with gr.Column():
                            errors_output = gr.Textbox(
                                label="Errors", lines=3, show_label=True
                            )

                    with gr.Row():
                        with gr.Column():
                            model_actions_output = gr.Textbox(
                                label="Model Actions", lines=3, show_label=True
                            )
                        with gr.Column():
                            model_thoughts_output = gr.Textbox(
                                label="Model Thoughts", lines=3, show_label=True
                            )

                    trace_file = gr.File(label="Trace File")
                    agent_history_file = gr.File(label="Agent History")

                stop_button.click(
                    fn=stop_agent,
                    inputs=[],
                    outputs=[errors_output, stop_button, run_button],
                )

                run_button.click(
                    fn=run_with_stream,
                    inputs=[
                        agent_type, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                        save_recording_path, save_agent_history_path, save_trace_path,
                        enable_recording, task, add_infos, max_steps, use_vision, max_actions_per_step,
                        tool_calling_method, profile_info
                    ],
                    outputs=[
                        browser_view,           # Browser view
                        final_result_output,    # Final result
                        errors_output,          # Errors
                        model_actions_output,   # Model actions
                        model_thoughts_output,  # Model thoughts
                        recording_display,      # Latest recording
                        trace_file,             # Trace file
                        agent_history_file,     # Agent history file
                        stop_button,            # Stop button
                        run_button              # Run button
                    ],
                    queue=True
                )

            with gr.TabItem("🎥 Recordings"):
                def list_recordings(save_recording_path):
                    if not os.path.exists(save_recording_path):
                        os.makedirs(save_recording_path, exist_ok=True)
                        return []
                    recordings = glob.glob(os.path.join(save_recording_path, "*.mp4")) \
                                 + glob.glob(os.path.join(save_recording_path, "*.webm"))
                    recordings.sort(key=os.path.getctime)
                    numbered_recordings = []
                    for idx, recording in enumerate(recordings, start=1):
                        filename = os.path.basename(recording)
                        numbered_recordings.append((recording, f"{idx}. {filename}"))
                    return numbered_recordings

                recordings_gallery = gr.Gallery(
                    label="Recordings",
                    value=list_recordings(config['save_recording_path']),
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )

                refresh_button = gr.Button("🔄 Refresh Recordings", variant="secondary")
                refresh_button.click(
                    fn=list_recordings,
                    inputs=save_recording_path,
                    outputs=recordings_gallery
                )

        enable_recording.change(
            lambda enabled: gr.update(interactive=enabled),
            inputs=enable_recording,
            outputs=save_recording_path
        )

        use_own_browser.change(fn=close_global_browser)
        keep_browser_open.change(fn=close_global_browser)

    return demo

def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument(
        "--theme", type=str, default="Monochrome",
        choices=["Default", "Soft", "Monochrome", "Glass", "Origin", "Citrus", "Ocean", "Base"],
        help="Theme to use for the UI"
    )
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")

    args = parser.parse_args()
    config_dict = default_config()

    demo = create_ui(config_dict, theme_name=args.theme)
    demo.launch(server_name=args.ip, server_port=args.port, allowed_paths=["./tmp/"])

if __name__ == "__main__":
    main()