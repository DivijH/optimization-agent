import asyncio
from datetime import datetime
import os
import json
import base64
import shutil
import sys
import subprocess
import signal
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict, is_dataclass
import click
import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from urllib.parse import urljoin, quote

from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.controller.service import Controller
from browser_use.controller.views import (
    GoToUrlAction,
    InputTextAction,
    SendKeysAction,
    ScrollAction,
    OpenTabAction,
    CloseTabAction,
    SwitchTabAction,
)
from browser_use.browser.views import BrowserStateSummary
from memory import MemoryModule, ProductMemory

from prompts import (
    PRODUCT_ANALYSIS_PROMPT,
    FINAL_DECISION_PROMPT,
)

# Pricing per million tokens for OpenAI models
MODEL_PRICING = {
    # model_name: {"input": cost_per_million, "output": cost_per_million}
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "openai/o4-mini": {"input": 1.10, "output": 4.40},
}

# Constants for token usage breakdown
IMAGE_TOKEN_PERCENTAGE = 0.7806  # 78.06% of analysis tokens are from images
TEXT_TOKEN_PERCENTAGE = 0.2194   # 21.94% of analysis tokens are from text

# Default task and persona for the agent (Persona 21)
DEFAULT_TASK = """
indoor frisbee
""".strip()
DEFAULT_PERSONA = """
Persona: Samantha

Background:
Samantha is a successful entrepreneur who founded a thriving tech company in Silicon Valley. With a keen eye for innovation and a talent for identifying market opportunities, she has built her business from the ground up and is now reaping the rewards of her hard work.

Demographics:
Age: 35
Gender: Female
Education: Bachelor's degree in Computer Science
Profession: Founder and CEO of a tech startup
Income: $250,000

Financial Situation:
Samantha has a comfortable financial situation, with a high income from her successful tech company. She is financially savvy and invests her wealth in a diverse portfolio, aiming to grow her assets and secure her financial future.

Shopping Habits:
Samantha has a discerning eye for quality and design, and she enjoys browsing high-end stores and boutiques for unique and stylish items. She is not afraid to splurge on luxury goods that she believes are worth the investment. However, she also values efficiency and often shops online for convenience.

Professional Life:
As the founder and CEO of her tech startup, Samantha's professional life is fast-paced and demanding. She juggles a variety of responsibilities, from overseeing product development to managing a team of employees. Despite the challenges, she thrives on the excitement and sense of accomplishment that comes with building a successful business.

Personal Style:
Samantha has a chic, modern style that reflects her professional success. She favors well-tailored, sophisticated outfits that exude confidence and elegance. She enjoys accessorizing with stylish jewelry and high-quality handbags, and she typically wears size medium clothing.

Samantha is a frequent traveler and loves to fly with Emirates Airlines. She typically wakes up at 7 am each day to start her busy schedule.

Samantha lives in San Francisco.
""".strip()

os.environ["OPENAI_API_KEY"] = open("keys/litellm.key").read().strip()
os.environ["OPENAI_API_BASE"] = "https://litellm.litellm.kn.ml-platform.etsy-mlinfra-dev.etsycloud.com"

@dataclass
class EtsyShoppingAgent:
    """
    An agent that shops on Etsy based on a given task and persona.
    """
    task: str = DEFAULT_TASK
    persona: str = DEFAULT_PERSONA
    manual: bool = False
    headless: bool = False
    max_steps: Optional[int] = None
    debug_path: Optional[str] = "debug_run"
    viewport_width: int = 1920
    viewport_height: int = 1080
    user_data_dir: Optional[str] = None
    logger: Optional[logging.Logger] = None
    non_interactive: bool = False

    # LLM configuration
    model_name: str = "gpt-4o-mini"
    final_decision_model_name: Optional[str] = None
    temperature: float = 0.7

    # Recording configuration
    record_video: bool = False

    # Internal state, not initialized by the user
    history: List[str] = field(init=False, default_factory=list)
    browser_session: Optional[BrowserSession] = field(init=False, default=None)
    controller: Controller = field(init=False, default_factory=Controller)
    llm: BaseChatModel = field(init=False)
    final_decision_llm: BaseChatModel = field(init=False)
    memory: MemoryModule = field(init=False, default_factory=MemoryModule)
    current_product_name: Optional[str] = field(init=False, default=None)  # Store the current product being analyzed
    visited_listing_ids: set = field(init=False, default_factory=set)  # Track visited listing IDs
    token_usage: Dict[str, Dict[str, Any]] = field(init=False, default_factory=dict)

    # Internal attribute: ffmpeg process
    _record_proc: Optional[subprocess.Popen] = field(init=False, default=None, repr=False)

    async def _save_token_usage(self):
        """Calculates total cost and saves token usage to a JSON file in real-time."""
        if not self.debug_path:
            return

        # Calculate total session cost by summing all model costs
        total_session_cost = 0.0
        for model_usage in self.token_usage.values():
            # Add analysis costs
            if "analysis" in model_usage:
                total_session_cost += model_usage["analysis"]["total_cost"]
            # Add final decision costs
            if "final_decision" in model_usage:
                total_session_cost += model_usage["final_decision"]["total_cost"]

        token_usage_path = os.path.join(self.debug_path, "_token_usage.json")
        token_usage_data = {
            "models": self.token_usage,
            "total_session_cost": total_session_cost,
        }
        
        def _write_file_sync():
            # This function contains the blocking file I/O
            os.makedirs(self.debug_path, exist_ok=True)
            with open(token_usage_path, "w") as f:
                json.dump(token_usage_data, f, indent=2)

        try:
            # Run the synchronous file write operation in a separate thread
            await asyncio.to_thread(_write_file_sync)
        except Exception as e:
            self._log(f"   - Failed to save token usage in real-time: {e}", level="error")

    def _log(self, message: str, level: str = "info"):
        """Logs a message using the provided logger or prints to stdout."""
        if self.logger:
            if level == "error":
                self.logger.error(message)
            else:
                self.logger.info(message)
        else:
            if level == "error":
                print(message, file=sys.stderr)
            else:
                print(message)

    async def _update_token_usage(self, model_name: str, usage_metadata: Optional[Dict[str, Any]], usage_type: str = "analysis"):
        """Updates the token usage count for a given model.
        
        Args:
            model_name: The name of the model used
            usage_metadata: The usage metadata from the model response
            usage_type: Type of usage - either 'analysis' or 'final_decision'
        """
        if not usage_metadata:
            return

        input_tokens = usage_metadata.get("input_tokens", 0)
        output_tokens = usage_metadata.get("output_tokens", 0)
        total_tokens = usage_metadata.get("total_tokens", 0)

        if not total_tokens:
            return

        # Initialize model usage stats if not exists
        if model_name not in self.token_usage:
            self.token_usage[model_name] = {
                "analysis": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "image_tokens": 0,
                    "text_tokens": 0,
                    "input_text_cost": 0.0,
                    "input_image_cost": 0.0,
                    "input_total_cost": 0.0,
                    "output_cost": 0.0,
                    "total_cost": 0.0
                },
                "final_decision": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "input_cost": 0.0,
                    "output_cost": 0.0,
                    "total_cost": 0.0
                }
            }

        # Update the appropriate usage type
        usage_data = self.token_usage[model_name][usage_type]
        usage_data["input_tokens"] += input_tokens
        usage_data["output_tokens"] += output_tokens
        usage_data["total_tokens"] += total_tokens

        # Calculate costs
        if model_name in MODEL_PRICING:
            price_per_million = MODEL_PRICING[model_name]
            
            if usage_type == "analysis":
                # Calculate image and text tokens
                image_tokens = int(input_tokens * IMAGE_TOKEN_PERCENTAGE)
                text_tokens = int(input_tokens * TEXT_TOKEN_PERCENTAGE)
                usage_data["image_tokens"] = image_tokens
                usage_data["text_tokens"] = text_tokens
                
                # Calculate input costs (text and image)
                input_text_cost = (text_tokens / 1_000_000) * price_per_million["input"]
                input_image_cost = (image_tokens / 1_000_000) * price_per_million["input"]
                output_cost = (output_tokens / 1_000_000) * price_per_million["output"]
                
                usage_data["input_text_cost"] += input_text_cost
                usage_data["input_image_cost"] += input_image_cost
                usage_data["input_total_cost"] = usage_data["input_text_cost"] + usage_data["input_image_cost"]
                usage_data["output_cost"] += output_cost
                usage_data["total_cost"] = usage_data["input_total_cost"] + usage_data["output_cost"]
            else:  # final_decision
                input_cost = (input_tokens / 1_000_000) * price_per_million["input"]
                output_cost = (output_tokens / 1_000_000) * price_per_million["output"]
                
                usage_data["input_cost"] += input_cost
                usage_data["output_cost"] += output_cost
                usage_data["total_cost"] = usage_data["input_cost"] + usage_data["output_cost"]

        # Log the usage details
        if usage_type == "analysis":
            self._log(
                f"   - Token usage for {model_name} (analysis): "
                f"Input={input_tokens} (Image={usage_data['image_tokens']}, Text={usage_data['text_tokens']}), "
                f"Output={output_tokens}, Total={total_tokens}"
            )
            self._log(
                f"   - Cost breakdown for {model_name} (analysis):"
                f"\n     Input Text: ${usage_data['input_text_cost']:.6f}"
                f"\n     Input Image: ${usage_data['input_image_cost']:.6f}"
                f"\n     Input Total: ${usage_data['input_total_cost']:.6f}"
                f"\n     Output: ${usage_data['output_cost']:.6f}"
                f"\n     Total: ${usage_data['total_cost']:.6f}"
            )
        else:  # final_decision
            self._log(
                f"   - Token usage for {model_name} (final_decision): "
                f"Input={input_tokens}, Output={output_tokens}, Total={total_tokens}"
            )
            self._log(
                f"   - Cost breakdown for {model_name} (final_decision):"
                f"\n     Input: ${usage_data['input_cost']:.6f}"
                f"\n     Output: ${usage_data['output_cost']:.6f}"
                f"\n     Total: ${usage_data['total_cost']:.6f}"
            )

        # Save token usage to file in real-time
        await self._save_token_usage()

    def __post_init__(self):
        """Initializes the agent, handling debug path setup."""
        if self.model_name not in MODEL_PRICING:
            self._log(f"Model '{self.model_name}' not found in MODEL_PRICING. Aborting.", level="error")
            sys.exit(1)
        if self.final_decision_model_name and self.final_decision_model_name not in MODEL_PRICING:
            self._log(
                f"Final decision model '{self.final_decision_model_name}' not found in MODEL_PRICING. Aborting.",
                level="error",
            )
            sys.exit(1)

        if self.debug_path:
            if os.path.isdir(self.debug_path):
                if self.non_interactive or click.confirm(
                    f"Debug path '{self.debug_path}' already exists. Do you want to remove it and all its contents?",
                    default=False
                ):
                    try:
                        shutil.rmtree(self.debug_path)
                        self._log(f"Removed existing debug path: {self.debug_path}")
                    except Exception as e:
                        self._log(f"Error removing debug path: {e}", level="error")
                        sys.exit(1)
                else:
                    self._log("Aborting. Please choose a different debug path or remove the existing one manually.")
                    sys.exit(0)
            
            # Create the debug directory to ensure it's available for all subsequent operations.
            try:
                os.makedirs(self.debug_path)
                self._log(f"Created debug directory: {self.debug_path}")
            except OSError as e:
                self._log(f"Could not create debug directory: {e}", level="error")
                sys.exit(1)

        # Initialize the language model with user-specified (or default) parameters
        self.llm = ChatOpenAI(temperature=self.temperature, model=self.model_name)
        
        # Initialize the final decision LLM
        final_decision_model = self.final_decision_model_name or self.model_name
        self.final_decision_llm = ChatOpenAI(temperature=self.temperature, model=final_decision_model)
        
        # The user wants to use the task as the search query.
        self._log(f"Using task as the only search query: {self.task}")

    def _find_search_bar(self, state: BrowserStateSummary) -> Optional[int]:
        """Finds the search bar element on the page."""
        for index, element in state.selector_map.items():
            if (element.attributes.get("name") == "search_query" or
                element.attributes.get("id") == "global-enhancements-search-query" or
                element.attributes.get("aria-label", "").lower() == "search"):
                return index
        return None

    def _extract_product_name_from_url(self, href: str) -> Optional[str]:
        """Extract product name from Etsy listing URL.
        
        URL pattern: /listing/<listing_id>/<product-name-with-dashes>...
        Example: /listing/719316991/giant-halloween-spider-photo-props
        """
        try:
            import re
            # Pattern to match /listing/<number>/<product-name-part>
            match = re.search(r'/listing/\d+/([^/?]+)', href)
            if match:
                product_slug = match.group(1)
                # Convert dashes to spaces and title case
                product_name = product_slug.replace('-', ' ').title()
                return product_name
            return None
        except Exception:
            return None

    async def _save_debug_screenshots(self, state: BrowserStateSummary, step: int) -> tuple[Optional[str], Optional[str]]:
        """
        Saves debug screenshots for the current step.
        
        Returns:
            tuple: (highlighted_image_path, plain_image_path) - both can be None if saving failed
        """
        image_path = None
        plain_image_path = None
        
        if self.debug_path and state.screenshot:
            os.makedirs(self.debug_path, exist_ok=True)
            # Save the highlighted screenshot (with bounding boxes) that comes from get_state_summary
            image_path = os.path.join(self.debug_path, f"screenshot_step_{step}_with_boxes.png")
            try:
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(state.screenshot))
                self._log(f"   - Saved highlighted debug screenshot to {image_path}")
            except Exception as e:
                self._log(f"   - Failed to save highlighted debug screenshot: {e}")
                image_path = None

        # Capture a plain screenshot without highlight overlays
        plain_screenshot_b64 = None
        try:
            # Remove existing highlights (if any) then take a fresh screenshot
            await self.browser_session.remove_highlights()
            plain_screenshot_b64 = await self.browser_session.take_screenshot()

            if self.debug_path and plain_screenshot_b64:
                plain_image_path = os.path.join(self.debug_path, f"screenshot_step_{step}_plain.png")
                try:
                    with open(plain_image_path, "wb") as f:
                        f.write(base64.b64decode(plain_screenshot_b64))
                    self._log(f"   - Saved plain debug screenshot to {plain_image_path}")
                except Exception as e:
                    self._log(f"   - Failed to save plain debug screenshot: {e}")
        except Exception as e:
            self._log(f"   - Failed to capture plain screenshot: {e}")

        return image_path, plain_image_path

    async def _scroll_and_collect(self, max_scrolls: int = 10, stop_at_bottom: bool = True, delay: float = 1.0) -> tuple[str, List[str]]:
        """Scrolls the current page multiple times, concatenating *all* text and viewport screenshots.
        Only scrolls the main webpage, not embedded scrollable elements like product galleries.

        Args:
            max_scrolls (int): Maximum number of scroll actions to perform. Defaults to 10.
            stop_at_bottom (bool): If True, stop early once the end of the page is reached. Defaults to True.
            delay (float): Seconds to wait after each scroll so that lazy-loaded content can appear. Defaults to 1.0.

        Returns:
            tuple[str, List[str]]: (full_page_text, list_of_viewport_screenshot_base64)
        """
        if not self.browser_session:
            raise RuntimeError("Browser session not initialised - call run() first.")

        screenshots: List[str] = []
        text_chunks: List[str] = []

        # Ensure we start with a clean viewport (no highlight overlays)
        try:
            await self.browser_session.remove_highlights()
        except Exception:
            pass

        page = await self.browser_session.get_current_page()

        # JavaScript snippet for scrolling ONLY the main page (document body/element)
        # This prevents scrolling through embedded elements like product image galleries
        MAIN_PAGE_SCROLL_JS = """(dy) => {
            // Always scroll the main page, never embedded scrollable elements
            const scrollOptions = { top: dy, behavior: 'smooth' };
            
            // Try document.scrollingElement first (modern browsers)
            if (document.scrollingElement) {
                document.scrollingElement.scrollBy(scrollOptions);
            } else {
                // Fallback to document.documentElement or document.body
                const mainElement = document.documentElement || document.body;
                mainElement.scrollBy(scrollOptions);
            }
        }"""

        # Use a named loop counter so we can tell if we've already tried scrolling once. This prevents an early
        # exit on the very first iteration when some pages momentarily report `scrollHeight === innerHeight` while
        # still loading.
        for i in range(max_scrolls):
            # Capture plain screenshot of the current viewport
            try:
                screenshot_b64 = await self.browser_session.take_screenshot(full_page=False)
                screenshots.append(screenshot_b64)
            except Exception as e:
                self._log(f"   - Failed to take screenshot during scroll capture: {e}")

            # Capture all visible text on the page at this scroll position
            try:
                inner_text: str = await self.browser_session.execute_javascript("() => document.body.innerText")
                if inner_text:
                    text_chunks.append(inner_text)
            except Exception as e:
                self._log(f"   - Failed to capture text during scroll capture: {e}")

            # Check how much more we can scroll on the main page
            try:
                _, pixels_below = await self.browser_session.get_scroll_info(page)
            except Exception:
                pixels_below = 0

            # Only stop if we *already* attempted at least one scroll.  This guarantees that the routine performs
            # a real scroll action even when `pixels_below` is incorrectly reported as zero before any scrolling
            # has occurred (a behaviour observed on some dynamic product pages).
            if stop_at_bottom and pixels_below <= 0 and i > 0:
                break

            # Perform smooth scroll roughly one viewport down (90 % of height to get some overlap)
            try:
                dy = int(self.viewport_height * 0.9)
                await page.evaluate(MAIN_PAGE_SCROLL_JS, dy)
            except Exception as e:
                self._log(f"   - Failed to perform smooth scroll: {e}")
                # Fallback to immediate scroll if smooth scroll fails
                try:
                    await self.browser_session._scroll_container(dy)
                except Exception:
                    pass

            # Give the page time to animate & lazy-load new content
            await asyncio.sleep(delay)

        # Concatenate text chunks – later chunks can contain repeated text, keep them but trim huge duplicates
        full_text = "\n".join(text_chunks)
        return full_text, screenshots

    async def _analyze_product_page(self, state: BrowserStateSummary, step: int, product_name: Optional[str] = None) -> Optional[ProductMemory]:
        """Analyzes a product page and adds the product to memory."""
        self._log("   - On a product page. Analyzing product details.")

        # Check if we have already analyzed this page
        if self.memory.get_product_by_url(state.url):
            self._log("   - Already analyzed this product.")
            return None

        _, screenshots_b64 = await self._scroll_and_collect(max_scrolls=15, stop_at_bottom=True, delay=0.3)
        screenshots_b64 = screenshots_b64[:3] # cap at first 3 images
        parser = JsonOutputParser()

        system_prompt = PRODUCT_ANALYSIS_PROMPT
        user_prompt_text = f"{self.persona}\nSearched Query: {self.task}\nCurrent Date: {datetime.now().strftime('%B %d')}"
        image_payloads = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            }
            for b64 in screenshots_b64
        ]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[{"type": "text", "text": user_prompt_text}, *image_payloads])
        ]

        try:
            ai_message = await self.llm.ainvoke(messages)
            if hasattr(ai_message, 'usage_metadata') and ai_message.usage_metadata:
                await self._update_token_usage(self.llm.model_name, ai_message.usage_metadata, usage_type="analysis")
            analysis_response = parser.parse(ai_message.content)
            
            product_memory = ProductMemory(
                product_name=product_name,
                url=state.url,
                price=analysis_response.get("price"),
                pros=analysis_response.get("pros", []),
                cons=analysis_response.get("cons", []),
                summary=analysis_response.get("summary", ""),
                semantic_score=analysis_response.get("semantic_score", "")
            )
            
            self.memory.add_product(product_memory)
            self._log(f"   - Added '{product_memory.product_name}' to memory.")
            
            # Save scroll screenshots for debugging
            if self.debug_path:
                os.makedirs(self.debug_path, exist_ok=True)
                scroll_image_paths = []
                for idx, b64 in enumerate(screenshots_b64):
                    img_path = os.path.join(self.debug_path, f"screenshot_step_{step}_scroll_{idx}.png")
                    try:
                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(b64))
                        scroll_image_paths.append(img_path)
                    except Exception as e:
                        self._log(f"   - Failed to save scroll screenshot {idx}: {e}")

                debug_info = {
                    "type": "product_analysis",
                    "input_url": state.url,
                    "product_name": product_name,
                    "prompt": {
                        "system": system_prompt,
                        "user_text": user_prompt_text
                    },
                    "scroll_screenshots": scroll_image_paths,
                    "output": analysis_response,
                }

                file_path = os.path.join(self.debug_path, f"debug_step_{step}.json")
                try:
                    with open(file_path, "w") as f:
                        json.dump(debug_info, f, indent=2)
                    self._log(f"   - Saved product analysis debug info to {file_path}")
                except Exception as e:
                    self._log(f"   - Failed to save product analysis debug info: {e}")

            return product_memory

        except Exception as e:
            self._log(f"   - An error occurred during product analysis: {e}")
            return None

    async def _choose_product_from_search(self, state: BrowserStateSummary, step: int) -> Optional[Dict[str, Any]]:
        """
        Finds the next unvisited product on a search results page and returns an action to click it.
        If no unvisited products are in the current viewport, it scrolls down.
        """
        # Extract all potential product listings from the current view
        product_listings = []
        for index, element in state.selector_map.items():
            attributes = element.attributes or {}
            href = attributes.get("href", "")
            
            # Heuristic to identify a product link:
            # - It must have a 'data-listing-id' attribute.
            # - The href should point to a listing page.
            # - It must not be an ad (checking text is a fallback).
            is_product_listing = 'data-listing-id' in attributes and '/listing/' in href

            if is_product_listing:
                # Extract listing ID from the data-listing-id attribute
                listing_id = attributes.get("data-listing-id")
                if not listing_id:
                    continue  # Skip if no listing ID found
                
                # Extract product name from the URL
                product_name = self._extract_product_name_from_url(href)
                if not product_name:
                    # Fallback to extracting from element text if URL extraction fails
                    element_text = element.get_all_text_till_next_clickable_element()
                    title_candidates = [line.strip() for line in element_text.split('\n') if len(line.strip()) > 10]
                    product_name = title_candidates[0] if title_candidates else element_text[:100]
                
                # Check if we've already visited this listing ID
                if listing_id in self.visited_listing_ids:
                    continue  # Skip already visited items
                
                # Get element text for ad filtering
                element_text = element.get_all_text_till_next_clickable_element()
                
                # Filter out ads and already-visited items by URL
                full_url = urljoin("https://www.etsy.com", href) if href.startswith("/") else href
                if (
                    'ad from etsy seller' not in element_text.lower() and
                    'rtisement' not in element_text.lower() and
                    not self.memory.get_product_by_url(full_url)
                ):
                    product_listings.append({
                        "index": index,
                        "text": element_text,
                        "product_name": product_name,
                        "href": href,
                        "listing_id": listing_id
                    })

        # If we found at least one new product, act on the first one
        if product_listings:
            chosen_listing = product_listings[0]
            chosen_index = chosen_listing["index"]
            chosen_product_name = chosen_listing["product_name"]
            chosen_listing_id = chosen_listing["listing_id"]

            self._log(f"   - Found unvisited product: '{chosen_product_name}' (ID: {chosen_listing_id}) at index {chosen_index}. Opening it.")
            href = chosen_listing["href"]
            
            # Mark this listing as visited
            self.visited_listing_ids.add(chosen_listing_id)
            
            # Ensure the URL is absolute for navigation
            if href.startswith("/"):
                href = urljoin("https://www.etsy.com", href)

            # Prepare an action to open the product page in a new tab
            return {
                "open_tab": OpenTabAction(url=href),
                "switch_tab": SwitchTabAction(page_id=-1), # -1 switches to the newly opened tab
                "product_name": chosen_product_name,
                "listing_id": chosen_listing_id
            }
        
        # If no new products are visible, check if we can scroll further down
        if state.pixels_below and state.pixels_below > 0:
            self._log("   - No new products visible. Scrolling down to find more.")
            # Use a smaller scroll amount for search pages to avoid skipping product rows
            scroll_amount = int(self.viewport_height * 0.3)  # 30% of viewport height
            return {"scroll_down": ScrollAction(amount=scroll_amount)}

        # If we can't scroll further, we've reached the end of the results
        self._log("   - Reached end of search results page.")
        return None

    async def _think(self, state: BrowserStateSummary, step: int) -> Optional[Dict[str, Any]]:
        """
        The agent's 'brain'. Decides the next action based on the current page state.
        """
        self._log("🤔 Thinking...")
        self._log(f"   - Current URL: {state.url}")
        self._log(f"   - Current Task: {self.task}")

        if "etsy.com" not in state.url:
            search_query_encoded = quote(self.task)
            search_url = f"https://www.etsy.com/search?q={search_query_encoded}&application_behavior=default"
            self._log(f"   - Initial navigation. Going to search page for '{self.task}'.")
            return {
                "go_to_url": GoToUrlAction(url=search_url),
                "search_query": self.task
            }

        if "etsy.com/search" in state.url:
            action = await self._choose_product_from_search(state, step)
            return action

        elif "etsy.com/listing" in state.url:
            # 1️⃣ Analyse the product first
            if self.current_product_name:
                self._log(f"   - Analyzing product: {self.current_product_name}")
            await self._analyze_product_page(state, step, self.current_product_name)

            # 2️⃣ Work out which tab we're on so we can close it later
            try:
                current_tab_id = next((tab.page_id for tab in state.tabs if tab.url == state.url), None)
            except Exception:
                current_tab_id = None

            # 3️⃣ Finished analyzing this product; close the product tab and return to results.
            if current_tab_id is not None and current_tab_id != 0:
                return {"close_tab": CloseTabAction(page_id=current_tab_id)}

        search_bar = self._find_search_bar(state)
        if search_bar and f"searched_{self.task}" not in self.history:
            self._log(f"   - Found search bar. Searching for '{self.task}'.")
            return {
                "input_text": InputTextAction(index=search_bar, text=self.task),
                "send_keys": SendKeysAction(keys="Enter"),
                "search_query": self.task
            }

        self._log("   - No specific action decided.")
        return None

    async def _make_final_purchase_decision(self):
        """Use the LLM to decide which products to finally purchase based on memory."""
        if not self.memory.products:
            self._log("🤷 No products were analyzed, skipping final purchase decision.")
            return

        # Filter products to include only those that are "HIGHLY RELEVANT" or "SOMEWHAT RELEVANT"
        relevant_products = [
            p for p in self.memory.products
            if p.semantic_score.upper() in ["HIGHLY RELEVANT", "SOMEWHAT RELEVANT"]
        ]

        if not relevant_products:
            self._log("🤷 No relevant products found after filtering, skipping final purchase decision.")
            return

        # Build a detailed list of products the agent has analyzed
        product_descriptions = ''
        for idx, product in enumerate(relevant_products, start=1):
            # pros = " ".join(product.pros) if product.pros else "-"
            # cons = " ".join(product.cons) if product.cons else "-"
            summary = product.summary if product.summary else "-"
            price = f"Price: ${product.price:.2f}" if product.price is not None else "Price: Not Available"
            desc = (
                f"{idx}. {product.product_name.title()}\n"
                # f"Pros: {pros}\n"
                # f"Cons: {cons}\n"
                f"Summary: {summary}\n"
                f"{price}"
            )
            product_descriptions += desc + "\n\n"

        user_prompt_text = f"""
Persona: {self.persona}

Searched Query: {self.task}

Here are the products that have been analyzed:
{product_descriptions.strip()}
""".strip()

        messages = [
            SystemMessage(content=FINAL_DECISION_PROMPT),
            HumanMessage(content=user_prompt_text),
        ]

        self._log("🔮 Asking LLM for final purchase recommendations...")
        try:
            ai_message = await self.final_decision_llm.ainvoke(messages)
            if hasattr(ai_message, 'usage_metadata') and ai_message.usage_metadata:
                await self._update_token_usage(self.final_decision_llm.model_name, ai_message.usage_metadata, usage_type="final_decision")
            response_content = ai_message.content.strip()
            try:
                decision_json = json.loads(response_content)
                # Ensure total_cost is calculated and present
                if "total_cost" not in decision_json or not isinstance(decision_json.get("total_cost"), (float, int)):
                    total_cost = 0.0
                    product_price_map = {p.product_name.lower(): p.price for p in relevant_products if p.price is not None}
                    
                    recommendations = decision_json.get("recommendations", [])
                    for rec in recommendations:
                        product_name = rec.get("product_name", "").lower()
                        if product_name in product_price_map:
                            total_cost += product_price_map[product_name]
                    
                    decision_json["total_cost"] = round(total_cost, 2)

            except json.JSONDecodeError:
                # If the model returned something that isn't valid JSON, fall back to raw string
                decision_json = {"raw_response": response_content}

            self._log("💡 Final purchase decision:")
            self._log(json.dumps(decision_json, indent=2))

            if self.debug_path:
                os.makedirs(self.debug_path, exist_ok=True)

                debug_info = {
                    "type": "final_purchase_decision",
                    "prompt": {
                        "system": FINAL_DECISION_PROMPT,
                        "user_text": user_prompt_text,
                    },
                    "output": decision_json,
                }

                decision_path = os.path.join(self.debug_path, "_final_purchase_decision.json")
                with open(decision_path, "w") as f:
                    json.dump(debug_info, f, indent=2)
                self._log(f"   - Final decision saved to {decision_path}")

        except Exception as e:
            self._log(f"   - An error occurred during final purchase decision making: {e}")

    async def run(self):
        """
        Main agent workflow for online shopping on Etsy.
        """
        self._log("🚀 Starting Etsy shopping agent...")
        # Monkey-patch BrowserSession._scroll_container once so that all future
        # ScrollAction invocations use smooth scrolling instead of the default
        # instant jump.  This ensures visual continuity for the end user when
        # the agent scrolls search result pages.
        if not getattr(BrowserSession, "_smooth_scroll_patched", False):
            async def _smooth_scroll_container(self_bs, pixels: int) -> None:
                """Replacement for BrowserSession._scroll_container with smooth behaviour."""
                SMART_SCROLL_JS_SMOOTH = """(dy) => {
                    const bigEnough = el => el.clientHeight >= window.innerHeight * 0.5;
                    const canScroll = el =>
                        el &&
                        /(auto|scroll|overlay)/.test(getComputedStyle(el).overflowY) &&
                        el.scrollHeight > el.clientHeight &&
                        bigEnough(el);

                    let el = document.activeElement;
                    while (el && !canScroll(el) && el !== document.body) el = el.parentElement;

                    el = canScroll(el)
                            ? el
                            : [...document.querySelectorAll('*')].find(canScroll)
                            || document.scrollingElement
                            || document.documentElement;

                    const opts = { top: dy, behavior: 'smooth' };
                    if (el === document.scrollingElement ||
                        el === document.documentElement ||
                        el === document.body) {
                        window.scrollBy(opts);
                    } else {
                        el.scrollBy(opts);
                    }
                }"""
                page = await self_bs.get_current_page()
                await page.evaluate(SMART_SCROLL_JS_SMOOTH, pixels)

            # Apply the patch and mark as done to avoid double-patching
            BrowserSession._scroll_container = _smooth_scroll_container  # type: ignore
            BrowserSession._smooth_scroll_patched = True

        self._log(f"   - Task: {self.task}")
        self._log(f"   - Persona: {self.persona}")

        # Begin screen recording before the browser launches so we capture the whole session
        if self.record_video:
            self._start_screen_recording()

        browser_profile = BrowserProfile(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            window_size={"width": self.viewport_width, "height": self.viewport_height},
            user_data_dir=self.user_data_dir,
        )

        self.browser_session = BrowserSession(
            keep_alive=True, 
            headless=self.headless,
            browser_profile=browser_profile
        )
        await self.browser_session.start()
        self._log("✅ Browser session started.")

        Action = self.controller.registry.create_action_model()

        step = 0
        while True:
            step += 1
            if self.max_steps and step > self.max_steps:
                self._log(f"   - Reached max steps ({self.max_steps}).")
                break

            step_info = f"{step}/{self.max_steps}" if self.max_steps else f"{step}"
            self._log(f"\n--- Step {step_info} ---")

            self._log("👀 Observing page state...")
            state = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=True)

            # Save screenshots when on search pages to see all products
            if "etsy.com/search" in state.url:
                await self._save_debug_screenshots(state, step)

            action_plan = await self._think(state, step)

            # Always write a generic debug file for every step so that debugging information is available even
            # when the agent decides to take no action (i.e. `action_plan` is None).
            if self.debug_path:
                debug_file_path = os.path.join(self.debug_path, f"debug_step_{step}.json")

                # Do not overwrite if some earlier logic (e.g. product analysis or choice) already produced a
                # debug file for this step.
                if not os.path.exists(debug_file_path):
                    serializable_action_plan = None

                    if action_plan is not None:
                        serializable_action_plan = {}
                        for k, v in action_plan.items():
                            if is_dataclass(v):
                                serializable_action_plan[k] = asdict(v)
                            else:
                                serializable_action_plan[k] = str(v)

                    debug_info = {
                        "type": "generic_action",
                        "step": step,
                        "url": state.url,
                        "action_plan": serializable_action_plan,
                    }

                    try:
                        with open(debug_file_path, "w") as f:
                            json.dump(debug_info, f, indent=2)
                        self._log(f"   - Saved generic action debug info to {debug_file_path}")
                    except Exception as e:
                        self._log(f"   - Failed to save generic action debug info: {e}")

            if action_plan:
                self._log("🎬 Taking action...")
                actions_to_perform = []
                if "go_to_url" in action_plan:
                    actions_to_perform.append(Action(go_to_url=action_plan["go_to_url"]))
                    self._log(f"   - Navigating to {action_plan['go_to_url'].url}")
                    # If we're going to a search URL directly, record it as a search action
                    if "search_query" in action_plan:
                        self.history.append(f"searched_{action_plan['search_query']}")
                        self.memory.add_search_query(action_plan['search_query'])
                if "input_text" in action_plan:
                    actions_to_perform.append(Action(input_text=action_plan["input_text"]))
                    self.history.append(f"searched_{action_plan['search_query']}")
                if "send_keys" in action_plan:
                    actions_to_perform.append(Action(send_keys=action_plan["send_keys"]))
                    self._log(f"   - Pressing '{action_plan['send_keys'].keys}'")
                if "click_element_by_index" in action_plan:
                    actions_to_perform.append(Action(click_element_by_index=action_plan["click_element_by_index"]))
                    if "product_name" in action_plan:
                        if "listing_id" in action_plan:
                            self.history.append(f"clicked_listing_{action_plan['listing_id']}")
                        else:
                            # Fallback to old method if listing_id not available
                            self.history.append(f"clicked_product_{action_plan['product_name']}")
                        self.current_product_name = action_plan["product_name"]  # Store for analysis
                    self._log(f"   - Clicking element at index {action_plan['click_element_by_index'].index}")
                if "open_tab" in action_plan:
                    actions_to_perform.append(Action(open_tab=action_plan["open_tab"]))
                    if "product_name" in action_plan:
                        if "listing_id" in action_plan:
                            self.history.append(f"opened_listing_{action_plan['listing_id']}")
                        else:
                            # Fallback to old method if listing_id not available
                            self.history.append(f"clicked_product_{action_plan['product_name']}")
                        self.current_product_name = action_plan["product_name"]  # Store for analysis
                    self._log(f"   - Opening new tab with {action_plan['open_tab'].url}")
                if "close_tab" in action_plan:
                    actions_to_perform.append(Action(close_tab=action_plan["close_tab"]))
                    self.current_product_name = None  # Clear after closing tab
                    self._log(f"   - Closing tab {action_plan['close_tab'].page_id}")
                if "switch_tab" in action_plan:
                    actions_to_perform.append(Action(switch_tab=action_plan["switch_tab"]))
                    self._log(f"   - Switching to tab {action_plan['switch_tab'].page_id}")
                if "scroll_down" in action_plan:
                    actions_to_perform.append(Action(scroll_down=action_plan["scroll_down"]))
                    self._log("   - Scrolling down the page")

                for action in actions_to_perform:
                    result = await self.controller.act(action=action, browser_session=self.browser_session)
                    self._log(f"   ✔️ Action result: {result.extracted_content}")
                    await asyncio.sleep(2)
                if self.manual:
                    input("Press Enter to continue...")
            else:
                self._log("   - No action to take. Ending agent run.")
                break

        self._log("\n✅ Shopping agent finished.")

    def _start_screen_recording(self):
        """Spawn an ffmpeg process to capture the entire primary display as a single video."""
        if self._record_proc:
            return  # already recording

        os.makedirs(self.debug_path, exist_ok=True)
        output_path = os.path.join(self.debug_path, "_session.mp4")

        # Build ffmpeg command depending on OS
        if sys.platform == "darwin":
            # macOS – capture primary display (id 1). Requires FFmpeg with avfoundation support.
            input_spec = "1:none"  # video id 1, no audio
            cmd = [
                "ffmpeg",
                "-y",  # overwrite
                "-f",
                "avfoundation",
                "-framerate",
                "30",
                "-i",
                input_spec,
                "-pix_fmt",
                "yuv420p",
                output_path,
            ]
        elif sys.platform.startswith("linux"):
            # Linux – capture :0 X11 display. Requires FFmpeg with x11grab.
            display = os.environ.get("DISPLAY", ":0")
            resolution = f"{self.viewport_width}x{self.viewport_height}"
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "x11grab",
                "-s",
                resolution,
                "-i",
                f"{display}",
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                output_path,
            ]
        elif sys.platform == "win32":
            # Windows – capture desktop using gdigrab.
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "gdigrab",
                "-framerate",
                "30",
                "-i",
                "desktop",
                "-pix_fmt",
                "yuv420p",
                output_path,
            ]
        else:
            self._log("⚠️  Screen recording not supported on this OS. Skipping video capture.")
            return

        try:
            self._record_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._log(f"🎥 Screen recording started → {output_path}")
        except FileNotFoundError:
            self._log("❌ ffmpeg not found. Install ffmpeg to enable screen recording.")
            self._record_proc = None

    def _stop_screen_recording(self):
        """Terminate ffmpeg process gracefully."""
        if not self._record_proc:
            return
        try:
            self._record_proc.terminate()
            self._record_proc.wait(timeout=10)
            self._log("🎬 Screen recording saved.")
        except Exception:
            try:
                self._record_proc.kill()
            except Exception:
                pass
        finally:
            self._record_proc = None

    async def shutdown(self):
        """Saves memory, makes final decision, and cleans up resources."""
        self._log("\n--- Shutting down and saving state ---")

        # Clean up browser and recording
        if self._record_proc:
            self._stop_screen_recording()
        if self.browser_session:
            await self.browser_session.kill()

        # Save memory
        if self.debug_path:
            memory_path = os.path.join(self.debug_path, "_memory.json")
            try:
                self.memory.save_to_json(memory_path)
                self._log(f"🧠 Memory saved to {memory_path}")
            except Exception as e:
                self._log(f"   - Failed to save memory: {e}", level="error")

            # Calculate and save semantic scores
            all_products = self.memory.products
            top_10_products = all_products[:10]

            def calculate_scores(products: List[ProductMemory]) -> Dict[str, int]:
                scores = {
                    "HIGHLY RELEVANT": 0,
                    "SOMEWHAT RELEVANT": 0,
                    "NOT RELEVANT": 0,
                }
                for p in products:
                    score = p.semantic_score.upper()
                    if score in scores:
                        scores[score] += 1
                
                return {
                    "total": len(products),
                    "highly_relevant": scores["HIGHLY RELEVANT"],
                    "somewhat_relevant": scores["SOMEWHAT RELEVANT"],
                    "not_relevant": scores["NOT RELEVANT"],
                }

            semantic_scores_data = {
                "page1_products": calculate_scores(all_products),
                "top_10_products": calculate_scores(top_10_products),
            }

            scores_path = os.path.join(self.debug_path, "_semantic_scores.json")
            try:
                with open(scores_path, "w") as f:
                    json.dump(semantic_scores_data, f, indent=2)
                self._log(f"📊 Semantic scores saved to {scores_path}")
            except Exception as e:
                self._log(f"   - Failed to save semantic scores: {e}", level="error")
        
        # Decide on the final purchase based on the memory collected
        await self._make_final_purchase_decision()

        if self.debug_path:
            self._log("📊 Final token usage:")
            total_session_cost = 0.0
            
            for model, usages in self.token_usage.items():
                model_total_cost = 0.0
                self._log(f"\n   Model: {model}")
                
                # Analysis stats
                analysis = usages["analysis"]
                analysis_total = analysis["total_cost"]
                model_total_cost += analysis_total
                self._log(
                    f"   - Analysis:"
                    f"\n     Tokens:"
                    f"\n       Input: {analysis['input_tokens']} (Image={analysis['image_tokens']}, Text={analysis['text_tokens']})"
                    f"\n       Output: {analysis['output_tokens']}"
                    f"\n       Total: {analysis['total_tokens']}"
                    f"\n     Costs:"
                    f"\n       Input Text: ${analysis['input_text_cost']:.6f}"
                    f"\n       Input Image: ${analysis['input_image_cost']:.6f}"
                    f"\n       Input Total: ${analysis['input_total_cost']:.6f}"
                    f"\n       Output: ${analysis['output_cost']:.6f}"
                    f"\n       Total: ${analysis_total:.6f}"
                )
                
                # Final decision stats
                final = usages["final_decision"]
                final_total = final["total_cost"]
                model_total_cost += final_total
                self._log(
                    f"   - Final Decision:"
                    f"\n     Tokens:"
                    f"\n       Input: {final['input_tokens']}"
                    f"\n       Output: {final['output_tokens']}"
                    f"\n       Total: {final['total_tokens']}"
                    f"\n     Costs:"
                    f"\n       Input: ${final['input_cost']:.6f}"
                    f"\n       Output: ${final['output_cost']:.6f}"
                    f"\n       Total: ${final_total:.6f}"
                )
                
                self._log(f"   - Model Total Cost: ${model_total_cost:.6f}")
                total_session_cost += model_total_cost
            
            self._log(f"\n   💰 Total Session Cost: ${total_session_cost:.6f}")

            # Token usage is saved in real-time. The final version is already on disk.
            self._log(f"   - Token usage saved to {os.path.join(self.debug_path, '_token_usage.json')}")

async def async_main(agent: EtsyShoppingAgent):
    """Runs the agent and handles graceful shutdown within a single event loop."""
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        await agent.run()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Shutting down...")
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        # Prevent further interruptions during shutdown
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        try:
            print("\nShutting down gracefully. Please wait, this may take a moment and cannot be interrupted...")
            await agent.shutdown()
        finally:
            # Restore the original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            print("\nShutdown complete.")

@click.command()
@click.option("--config-file", type=click.Path(exists=True, dir_okay=False, readable=True), default=None, help="Path to a JSON file containing 'intent' and 'persona' keys. Overrides --task and --persona.")
@click.option("--task", default=DEFAULT_TASK, help="The shopping task for the agent.")
@click.option("--persona", default=DEFAULT_PERSONA, help="The persona for the agent.")
@click.option("--manual", is_flag=True, help="Wait for user to press Enter after each agent action.")
@click.option("--headless", is_flag=True, help="Run the browser in headless mode.")
@click.option("--max-steps", default=None, type=int, help="The maximum number of steps the agent will take. If not provided, the agent will continue until no more products are left to analyze.")
@click.option("--debug-path", type=click.Path(), default="debug_run", help="Path to save debug artifacts, such as screenshots.")
@click.option("--width", default=3024, help="The width of the browser viewport.")
@click.option("--height", default=1964, help="The height of the browser viewport.")
@click.option("--model", "model_name", default="openai/o4-mini", help="Model name to use (e.g. gpt-4o, gpt-4o-mini).")
@click.option("--final-decision-model", "final_decision_model_name", default=None, help="Model name for the final decision. Defaults to the main model.")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature for the language model (0-2).")
@click.option("--record-video", is_flag=True, help="Record the agent's browser session and save it to the debug path.")
@click.option("--user-data-dir", type=click.Path(), help="Path to user data directory for the browser.")
def cli(config_file, task, persona, manual, headless, max_steps, debug_path, width, height, model_name, temperature, record_video, user_data_dir, final_decision_model_name):
    """A command-line interface to run the EtsyShoppingAgent."""
    
    if config_file:
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                task = config_data.get('task', task)
                persona = config_data.get('persona', persona)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading config file '{config_file}': {e}", file=sys.stderr)
            sys.exit(1)

    agent = EtsyShoppingAgent(
        task=task,
        persona=persona,
        manual=manual,
        headless=headless,
        max_steps=max_steps,
        debug_path=debug_path,
        viewport_width=width,
        viewport_height=height,
        model_name=model_name,
        temperature=temperature,
        record_video=record_video,
        user_data_dir=user_data_dir,
        final_decision_model_name=final_decision_model_name,
    )
    asyncio.run(async_main(agent))

if __name__ == "__main__":
    cli() 