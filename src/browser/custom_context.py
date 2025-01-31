import base64
import json
import logging
import os

from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import BrowserContext as PlaywrightBrowserContext

logger = logging.getLogger(__name__)


class CustomBrowserContext(BrowserContext):
    def __init__(
        self,
        browser: "Browser",
        config: BrowserContextConfig = BrowserContextConfig()
    ):
        super(CustomBrowserContext, self).__init__(browser=browser, config=config)

    async def get_screenshot_base64(self) -> str:
        """
        Captures a screenshot of the current page and returns it as a base64-encoded string.
        """
        page = await self.get_current_page()
        screenshot = await page.screenshot(type="png")  # Capture as PNG
        encoded = base64.b64encode(screenshot).decode("utf-8")
        return encoded