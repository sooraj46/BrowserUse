# Browser Use WebUI

This repository provides a **custom implementation** of a browser automation agent, extending the capabilities of the [`browser_use`](https://github.com/browser-use) library. It integrates **Google Gemini models** for intelligent browser interaction and features a **web-based UI** for task execution and configuration.

Key features include:

- Customization to work seamlessly with **Google Gemini AI Studio's free tier** (no Langchain required).
- **Web-based UI** for easy interaction and control.
- Enhanced **browser automation** using Playwright.

> **Acknowledgment:** This project is built upon the excellent [`browser-use`](https://github.com/browser-use) library. Huge thanks to its developers for their contributions!

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Linux/macOS
source venv/bin/activate  
# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
playwright install
```

### 4. Set Up Environment Variables

#### Required:

- **Google API Key** to use Gemini models. Add it to a `.env` file:
  ```ini
  GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
  ```
  Replace `YOUR_GEMINI_API_KEY` with your actual key.

#### Optional:

- To use a **custom Chrome browser instance**, set:
  ```ini
  CHROME_PATH=/path/to/chrome
  CHROME_USER_DATA=/path/to/user/data
  ```
  Refer to the [`browser_use`](https://github.com/browser-use)[ documentation](https://github.com/browser-use) for more details.

---

## 📌 Usage

### Running the Web UI

1. Navigate to the repository directory:
   ```bash
   python web_ui.py
   ```
2. Open a browser and go to the displayed address (e.g., `http://127.0.0.1:7788`).
3. Use the **Web UI** to:
   - Configure **agent settings**, **LLM parameters**, and **browser options**.
   - Enter a task description and optional hints.
   - Click **▶️ Run Agent** to execute automation.
   - Monitor execution logs and browser actions.
   - View **recordings** and **traces** after execution.
   - Stop the agent anytime with **⏹️ Stop**.

### Running Tests

To ensure everything works correctly, run:

```bash
pytest tests/
```

---

## 📂 Directory Structure

```plaintext
.
├── src/
│   ├── agent/             # Custom agent logic
│   │   ├── custom_agent.py          # Direct integration with Gemini API
│   │   ├── custom_message_manager.py # Handles message communication
│   │   ├── custom_prompts.py        # Defines custom agent prompts
│   │   └── custom_views.py          # Data models for output/steps
│   ├── browser/          # Custom browser automation classes
│   │   ├── custom_browser.py       # Extends base Browser functionality
│   │   └── custom_context.py       # Manages browser sessions
│   ├── controller/       # Custom actions for browser control
│   │   └── custom_controller.py    # Clipboard interaction & more
│   ├── utils/            # Utility functions
│   │   ├── agent_state.py         # Manages agent state & stop requests
│   │   ├── default_config_settings.py # Default settings management
│   │   ├── llm.py                 # Loads LLM models (Gemini)
│   │   ├── messages.py            # Custom message classes
│   │   └── utils.py               # General utilities
│   └── __init__.py
├── tests/               # Test cases for automation & AI interactions
│   ├── test_browser_use.py   # Browser automation tests
│   ├── test_gemini_chat.py   # Gemini API interaction tests
│   └── test_playwright.py    # Playwright functionality tests
└── web_ui.py            # Gradio-based Web UI
```

---

## 🤝 Contributing

We welcome contributions! Follow these steps:

1. **Report Issues**: Open an issue on [GitHub Issues](https://github.com/sooraj46/BrowserUse/issues) with details.
2. **Feature Requests**: Suggest enhancements via an issue.
3. **Code Contributions**:
   - Fork the repository.
   - Create a new branch (`feature/your-feature` or `fix/your-bug`).
   - Make changes, add tests, and ensure everything works.
   - Submit a **pull request (PR)** to the `main` branch.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2025 Sooraj J Sundar

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
```

