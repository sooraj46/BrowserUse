
# Browser Use WebUI - Custom Agent Example

This repository provides a custom implementation of a browser automation agent, extending the capabilities of the `browser_use` library. It leverages Google Gemini models for intelligent browser interaction, offering a web user interface for task execution and configuration. This customization is specifically designed to work with Google Gemini in AI Studio's free tier, without the use of Langchain, making it accessible and efficient for users experimenting with browser automation and Gemini.

**This project is a customization of the excellent [browser-use](https://github.com/browser-use) library. We extend our sincere gratitude to the browser-use developers for creating such a powerful and versatile foundation. This project would not be possible without their work.**

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
   ```

3. **Install the required Python packages:**

   ```bash
   pip install -r requirements.txt
   playwright install 
   ```

4. **Set up environment variables:**

   - You need a Google API key to use the Gemini models.
   - Set the `GOOGLE_API_KEY` environment variable with your API key. You can do this by creating a `.env` file in the root directory with the following content:

     ```
     GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
     ```
     Replace `YOUR_GEMINI_API_KEY` with your actual Gemini API key.

   - (Optional) If you want to use your own Chrome browser instance, you may need to set `CHROME_PATH` and `CHROME_USER_DATA` environment variables. Refer to the `browser_use` library documentation for details.

## Usage

### Running the Web UI

1.  Navigate to the repository directory in your terminal.
2.  Run the `web_ui.py` script:

    ```bash
    python web_ui.py
    ```

3.  Open your web browser and go to the address displayed in the terminal (usually `http://127.0.0.1:7788`).
4.  In the Web UI, you can:
    - Configure agent settings, LLM parameters, and browser options under the respective tabs.
    - Enter a task description and optional hints in the "Run Agent" tab.
    - Click "▶️ Run Agent" to start the browser automation process.
    - Monitor the live browser view, results, errors, and model outputs in the "Results" tab.
    - Stop the agent using the "⏹️ Stop" button.
    - View recordings and traces in the "Recordings" and "Results" tabs after the agent run.

### Running Tests

To run the tests, navigate to the repository root directory and use `pytest`:

```bash
pytest tests/
```

## Directory Structure

```
└── ./
    ├── src
    │   ├── agent
    │   │   ├── __init__.py
    │   │   ├── custom_agent.py         # Custom agent implementation using Gemini API directly.
    │   │   ├── custom_message_manager.py # Custom message management for Gemini interaction.
    │   │   ├── custom_prompts.py       # Custom system and agent prompts.
    │   │   └── custom_views.py         # Custom data models for agent output and step information.
    │   ├── browser
    │   │   ├── __init__.py
    │   │   ├── custom_browser.py       # Custom browser class extending base Browser functionalities.
    │   │   └── custom_context.py       # Custom browser context class.
    │   ├── controller
    │   │   ├── __init__.py
    │   │   └── custom_controller.py    # Custom controller with extended actions like clipboard interaction.
    │   ├── utils
    │   │   ├── __init__.py
    │   │   ├── agent_state.py        # Manages the state of the agent, including stop requests.
    │   │   ├── default_config_settings.py # Default configurations and settings management.
    │   │   ├── llm.py                # Utility for loading LLM models (currently Gemini).
    │   │   ├── messages.py           # Custom message classes.
    │   │   └── utils.py              # General utility functions.
    │   └── __init__.py
    ├── tests
    │   ├── test_browser_use.py       # Integration tests for browser automation.
    │   ├── test_gemini_chat.py       # Tests Gemini API interaction.
    │   └── test_playwright.py        # Tests Playwright browser functionalities.
    └── web_ui.py                     # Gradio Web UI for interacting with the agent.
```

- **`src/`**: Contains the core source code of the custom browser agent.
    - **`agent/`**:  Houses the custom agent implementation, message management, prompts, and data views tailored for Gemini.
        - `custom_agent.py`:  The main custom agent class (`CustomAgent`) that overrides the base agent to use Google Gemini directly for action generation. It handles communication with the Gemini API, parses responses, and manages agent steps.
        - `custom_message_manager.py`:  A custom message manager (`CustomMessageManager`) responsible for constructing and managing the conversation history with the LLM, optimized for Gemini.
        - `custom_prompts.py`: Defines custom system and agent prompts (`CustomSystemPrompt`, `CustomAgentMessagePrompt`) to guide the Gemini model's behavior and format.
        - `custom_views.py`:  Defines custom data models (`CustomAgentOutput`, `CustomAgentStepInfo`, `CustomAgentBrain`) for structuring agent outputs, step information, and agent's internal state.
    - **`browser/`**: Contains custom browser and browser context classes.
        - `custom_browser.py`:  Extends the base `Browser` class with custom functionalities, such as managing persistent Chrome instances (`CustomBrowser`).
        - `custom_context.py`:  Provides a custom browser context class (`CustomBrowserContext`) for managing browser sessions.
    - **`controller/`**: Contains the custom controller with extended actions.
        - `custom_controller.py`:  Defines a custom controller (`CustomController`) that registers additional actions like clipboard interaction (`copy_to_clipboard`, `paste_from_clipboard`).
    - **`utils/`**:  Includes utility modules for agent state management, configurations, LLM loading, and general utilities.
        - `agent_state.py`:  Manages the global agent state (`AgentState`), allowing for stopping and resetting the agent.
        - `default_config_settings.py`:  Provides default configuration settings, loading, and saving functionalities for the Web UI.
        - `llm.py`:  Utility functions for loading and managing Large Language Models (`get_llm_model`), currently supporting Gemini.
        - `messages.py`:  Defines base message classes (`BaseMessage`, `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`) used for communication with the LLM.
        - `utils.py`:  General utility functions such as encoding images, retrieving latest files, and capturing screenshots, as well as helper functions for LLM model selection in the UI.
    - `__init__.py`: Initializes the `src` directory as a Python package.
- **`tests/`**: Contains test files for verifying different functionalities.
    - `test_browser_use.py`: Integration tests to verify the browser automation capabilities of the custom agent.
    - `test_gemini_chat.py`: Tests specifically the interaction with the Gemini API using the `google.genai` library.
    - `test_playwright.py`: Tests core Playwright browser functionalities and connection.
- **`web_ui.py`**:  A Gradio-based web user interface (`Browser Use WebUI`) to interact with the custom browser agent. It allows users to set tasks, configure agent parameters, run the agent, and view results, recordings, and traces.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these guidelines:

1.  **Reporting Issues:** If you encounter a bug or issue, please open a new issue on the https://github.com/sooraj46/BrowserUse/issues. Provide a clear and descriptive title, steps to reproduce the issue, and your environment details.

2.  **Suggesting Enhancements:** For feature requests or suggestions, please open a new issue on the https://github.com/sooraj46/BrowserUse/issues with a detailed description of the proposed enhancement.

3.  **Contributing Code:**
    - Fork the repository.
    - Create a new branch for your feature or bug fix.
    - Make your changes and ensure they are well-tested.
    - Submit a pull request to the `main` branch with a clear description of your changes.

    Please adhere to the existing code style and conventions. Ensure your contributions are aligned with the project's goals and roadmap.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

```
MIT License

Copyright (c) [Year] [Your Name or Organization Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
