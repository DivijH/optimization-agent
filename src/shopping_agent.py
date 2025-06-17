import asyncio
from datetime import datetime
import os
import json
import base64
import shutil
import sys
import subprocess
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict, is_dataclass
import click
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from urllib.parse import urljoin

from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.controller.service import Controller
from browser_use.controller.views import (
    ClickElementAction,
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
    SUBTASK_GENERATION_PROMPT,
    CHOICE_SYSTEM_PROMPT,
    PRODUCT_ANALYSIS_PROMPT,
    FINAL_DECISION_PROMPT,
)

# Default task and persona for the agent
DEFAULT_TASK = """
buy a large, inflatable spider decoration for halloween
""".strip()
DEFAULT_PERSONA = """
Persona: Michael

Background:
Michael is a mid-career professional working as a marketing manager at a technology startup in San Francisco. He is passionate about using data-driven strategies to drive growth and innovation for the company.

Demographics:
Age: 42
Gender: Male
Education: Bachelor's degree in Business Administration
Profession: Marketing Manager
Income: $75,000

Financial Situation:
Michael has a comfortable income that allows him to maintain a decent standard of living in the expensive San Francisco Bay Area. He is financially responsible, saving a portion of his earnings for retirement and emergencies, while also enjoying occasional leisure activities and travel.

Shopping Habits:
Michael prefers to shop online for convenience, but he also enjoys the occasional trip to the mall or specialty stores to browse for new products. He tends to research items thoroughly before making a purchase, looking for quality, functionality, and value. Michael values efficiency and is not influenced by trends or impulse buys.

Professional Life:
As a marketing manager, Michael is responsible for developing and implementing marketing strategies to promote the startup's products and services. He collaborates closely with the product, sales, and design teams to ensure a cohesive brand experience. Michael is always looking for ways to optimize marketing campaigns and stay ahead of industry trends.

Personal Style:
Michael has a casual, yet professional style. He often wears button-down shirts, chinos, and leather shoes to the office. On weekends, he enjoys wearing comfortable, sporty attire for outdoor activities like hiking or cycling. Michael tends to gravitate towards neutral colors and classic, versatile pieces that can be mixed and matched.
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
    max_steps: int = 50
    debug_path: Optional[str] = "debug_run"
    viewport_width: int = 1920
    viewport_height: int = 1080
    products_to_check: int = 5  # -1 means all products on page

    # LLM configuration
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7

    # Recording configuration
    record_video: bool = False

    # Internal state, not initialized by the user
    history: List[str] = field(init=False, default_factory=list)
    sub_tasks: List[str] = field(init=False, default_factory=list)
    current_task_index: int = field(init=False, default=0)
    products_checked: int = field(init=False, default=0)
    browser_session: Optional[BrowserSession] = field(init=False, default=None)
    controller: Controller = field(init=False, default_factory=Controller)
    llm: BaseChatModel = field(init=False)
    memory: MemoryModule = field(init=False, default_factory=MemoryModule)
    current_product_name: Optional[str] = field(init=False, default=None)  # Store the current product being analyzed

    # Internal attribute: ffmpeg process
    _record_proc: Optional[subprocess.Popen] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        """Initializes sub-tasks from the main task after the object is created."""
        if self.debug_path and os.path.isdir(self.debug_path):
            if click.confirm(
                f"Debug path '{self.debug_path}' already exists. Do you want to remove it and all its contents?", 
                default=False
            ):
                try:
                    shutil.rmtree(self.debug_path)
                    print(f"Removed existing debug path: {self.debug_path}")
                except Exception as e:
                    print(f"Error removing debug path: {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                print("Aborting. Please choose a different debug path or remove the existing one manually.")
                sys.exit(0)

        # Initialize the language model with user-specified (or default) parameters
        self.llm = ChatOpenAI(temperature=self.temperature, model=self.model_name)
        
        parser = JsonOutputParser()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SUBTASK_GENERATION_PROMPT),
            ("user", "PERSONA: {persona}\nTASK: {task}"),
        ])
        
        chain = prompt | self.llm | parser
        
        try:
            raw_sub_tasks = chain.invoke({"task": self.task, "persona": self.persona})
            print(f"🔍 Raw sub-tasks derived from the task: {raw_sub_tasks}")

            if self.debug_path:
                os.makedirs(self.debug_path, exist_ok=True)
                debug_info = {
                    "type": "subtask_generation",
                    "prompt": [msg.prompt.template for msg in prompt.messages],
                    "input": {"task": self.task},
                    "output": raw_sub_tasks
                }
                file_path = os.path.join(self.debug_path, "debug_step_0.json")
                try:
                    with open(file_path, "w") as f:
                        json.dump(debug_info, f, indent=2)
                    print(f"   - Saved subtask generation debug info to {file_path}")
                except Exception as e:
                    print(f"   - Failed to save subtask generation debug info: {e}")
            
            sub_tasks_list = []
            if isinstance(raw_sub_tasks, list):
                sub_tasks_list = [str(item) for item in raw_sub_tasks]
            elif isinstance(raw_sub_tasks, dict):
                # Try to find a list within the dictionary
                for key, value in raw_sub_tasks.items():
                    if isinstance(value, list):
                        sub_tasks_list = [str(item) for item in value]
                        break
            
            if sub_tasks_list:
                self.sub_tasks = sub_tasks_list
            else:
                raise ValueError(f"LLM returned unexpected format for sub-tasks: {raw_sub_tasks}")

        except Exception as e:
            raise RuntimeError(f"An error occurred during sub-task creation: {e}")


    def _find_search_bar(self, state: BrowserStateSummary) -> Optional[int]:
        """Finds the search bar element on the page."""
        for index, element in state.selector_map.items():
            if (element.attributes.get("name") == "search_query" or
                element.attributes.get("id") == "global-enhancements-search-query" or
                element.attributes.get("aria-label", "").lower() == "search"):
                return index
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
                print(f"   - Saved highlighted debug screenshot to {image_path}")
            except Exception as e:
                print(f"   - Failed to save highlighted debug screenshot: {e}")
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
                    print(f"   - Saved plain debug screenshot to {plain_image_path}")
                except Exception as e:
                    print(f"   - Failed to save plain debug screenshot: {e}")
        except Exception as e:
            print(f"   - Failed to capture plain screenshot: {e}")

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
                print(f"   - Failed to take screenshot during scroll capture: {e}")

            # Capture all visible text on the page at this scroll position
            try:
                inner_text: str = await self.browser_session.execute_javascript("() => document.body.innerText")
                if inner_text:
                    text_chunks.append(inner_text)
            except Exception as e:
                print(f"   - Failed to capture text during scroll capture: {e}")

            # Check how much more we can scroll on the main page
            try:
                _, pixels_below = await self.browser_session.get_scroll_info(page)
            except Exception:
                pixels_below = 0

            # Only stop if we *already* attempted at least one scroll.  This guarantees that the routine performs
            # a real scroll action even when `pixels_below` is incorrectly reported as zero before any scrolling
            # has occurred (a behaviour observed on some dynamic product pages).
            if stop_at_bottom and pixels_below <= 0 and i > 0:
                break  # Reached (or believe to have reached) the bottom of the page after scrolling

            # Perform smooth scroll roughly one viewport down (90 % of height to get some overlap)
            try:
                dy = int(self.viewport_height * 0.9)
                await page.evaluate(MAIN_PAGE_SCROLL_JS, dy)
            except Exception as e:
                print(f"   - Failed to perform smooth scroll: {e}")
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

    async def _analyze_product_page(self, state: BrowserStateSummary, current_goal: str, step: int, product_name: Optional[str] = None) -> Optional[ProductMemory]:
        """Analyzes a product page and adds the product to memory."""
        print("   - On a product page. Analyzing product details.")

        # Check if we have already analyzed this page
        if self.memory.get_product_by_url(state.url):
            print("   - Already analyzed this product.")
            return None

        _, screenshots_b64 = await self._scroll_and_collect(max_scrolls=15, stop_at_bottom=True, delay=0.2)
        parser = JsonOutputParser()

        system_prompt = PRODUCT_ANALYSIS_PROMPT
        user_prompt_text = f"Persona: {self.persona}\nShopping Goal: {self.task}\nCurrent Date: {datetime.now().strftime('%B %d')}"
        image_payloads = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            }
            for b64 in screenshots_b64[:10]  # cap at first 10 images
        ]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[{"type": "text", "text": user_prompt_text}, *image_payloads])
        ]

        try:
            ai_message = await self.llm.ainvoke(messages)
            analysis_response = parser.parse(ai_message.content)
            
            product_memory = ProductMemory(
                product_name=product_name,
                url=state.url,
                pros=analysis_response.get("pros", []),
                cons=analysis_response.get("cons", []),
                summary=analysis_response.get("summary", "")
            )
            
            self.memory.add_product(product_memory)
            print(f"   - Added '{product_memory.product_name}' to memory.")
            
            if self.debug_path:
                os.makedirs(self.debug_path, exist_ok=True)

                # Save scroll screenshots
                scroll_image_paths = []
                for idx, b64 in enumerate(screenshots_b64):
                    img_path = os.path.join(self.debug_path, f"screenshot_step_{step}_scroll_{idx}.png")
                    try:
                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(b64))
                        scroll_image_paths.append(img_path)
                    except Exception as e:
                        print(f"   - Failed to save scroll screenshot {idx}: {e}")

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
                    print(f"   - Saved product analysis debug info to {file_path}")
                except Exception as e:
                    print(f"   - Failed to save product analysis debug info: {e}")

            return product_memory

        except Exception as e:
            print(f"   - An error occurred during product analysis: {e}")
            return None


    async def _choose_product(self, state: BrowserStateSummary, current_goal: str, step: int) -> Optional[Dict[str, Any]]:
        """
        Uses the model to choose a product from a screenshot of the search results page.
        """
        # Save screenshots for analysis if debug path is provided
        image_path, plain_image_path = await self._save_debug_screenshots(state, step)

        # Get plain screenshot for model input
        plain_screenshot_b64 = None
        try:
            # Remove existing highlights (if any) then take a fresh screenshot
            await self.browser_session.remove_highlights()
            plain_screenshot_b64 = await self.browser_session.take_screenshot()
        except Exception as e:
            print(f"   - Failed to capture plain screenshot: {e}")

        # If we couldn't get a plain screenshot, fall back to the highlighted one
        screenshot_for_model = plain_screenshot_b64 or state.screenshot

        product_listings = []
        for index, element in state.selector_map.items():
            element_text = element.get_all_text_till_next_clickable_element()
            product_name = element_text.strip().lower()[:500]  # Take first 500 chars to avoid too long keys
            if (
                len(element_text.strip()) > 20 and
                f"clicked_product_{product_name}" not in self.history and
                'rtisement' not in element_text.lower() and
                'original price' not in product_name and
                'search for' not in product_name and
                'sellers looking to' not in product_name and
                'order soon' not in product_name and
                'recommended categories' not in product_name
            ):
                # print(f"   - Found product: {product_name}")
                if not self.memory.is_product_in_memory(element_text):
                    product_listings.append({
                        "index": index, 
                        "text": element_text,
                        "product_name": product_name
                    })

        if not product_listings:
            print("   - No new products found to choose from.")
            return None

        parser = JsonOutputParser()

        def _parse_product_listings(product_listings: List[Dict[str, Any]]) -> str:
            return "\n".join([f"{product['index']}. {product['text']}" for product in product_listings])
        
        system_prompt = CHOICE_SYSTEM_PROMPT
        user_prompt_text = f"""
Persona: {self.persona}

Shopping Goal: {current_goal}

I have already seen these products:
{self.memory.get_memory_summary_for_prompt()}

Products:
{_parse_product_listings(product_listings)}
""".strip()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_for_model}"},
                    },
                ]
            )
        ]

        try:
            # Invoke the model and parse the JSON response
            ai_message = await self.llm.ainvoke(messages)
            choice_response = parser.parse(ai_message.content)

            if self.debug_path:
                os.makedirs(self.debug_path, exist_ok=True)
                debug_info = {
                    "type": "choice",
                    "prompt": {
                        "system": system_prompt,
                        "user_text": user_prompt_text
                    },
                    "output": choice_response
                }
                if image_path:
                    debug_info["highlighted_image"] = image_path
                if plain_image_path:
                    debug_info["plain_image"] = plain_image_path
                file_path = os.path.join(self.debug_path, f"debug_step_{step}.json")
                try:
                    with open(file_path, "w") as f:
                        json.dump(debug_info, f, indent=2)
                    print(f"   - Saved choice debug info to {file_path}")
                except Exception as e:
                    print(f"   - Failed to save choice debug info: {e}")
            
            chosen_index = choice_response.get("choice")

            if chosen_index is not None and isinstance(chosen_index, int):
                if chosen_index in state.selector_map:
                    print(f"   - LLM chose product with index {chosen_index}.")
                    chosen_element = state.selector_map[chosen_index]
                    href = chosen_element.attributes.get("href") if hasattr(chosen_element, "attributes") else None

                    # Find the corresponding listing details using the original index
                    chosen_listing = next((item for item in product_listings if item["index"] == chosen_index), None)
                    chosen_product_name = chosen_listing["product_name"] if chosen_listing else None

                    if href:
                        # Etsy often uses relative URLs (e.g. "/listing/123...") for product anchors. If the URL is
                        # relative, convert it to an absolute URL so that the browser can navigate correctly.
                        if href.startswith("/"):
                            href = urljoin("https://www.etsy.com", href)

                        # Open the product in a new tab and immediately switch focus to it so we can analyse the product page next.
                        action_dict = {
                            "open_tab": OpenTabAction(url=href),
                            # `-1` means the last tab that was just opened.
                            "switch_tab": SwitchTabAction(page_id=-1),
                        }
                        if chosen_product_name:
                            action_dict["product_name"] = chosen_product_name
                        return action_dict
                    else:
                        action_dict = {
                            "click_element_by_index": ClickElementAction(index=chosen_index, xpath=chosen_element.xpath),
                            # After clicking a product listing that doesn't provide an explicit href, Etsy usually opens
                            # the product in a new tab. Switch focus to the newest tab so subsequent logic sees the
                            # correct URL.
                            "switch_tab": SwitchTabAction(page_id=-1),
                        }
                        if chosen_product_name:
                            action_dict["product_name"] = chosen_product_name
                        return action_dict
                else:
                    print(f"   - LLM chose an invalid index: {chosen_index}.")
            else:
                print("   - LLM decided no product was suitable.")

        except Exception as e:
            print(f"   - An error occurred while asking LLM to choose a product: {e}")
            
        return None

    async def _choose_product_from_search(self, state: BrowserStateSummary, current_goal: str, step: int) -> Optional[Dict[str, Any]]:
        """
        Uses an LLM to choose a product from a search results page, potentially with visual input.
        """
        if self.products_to_check != -1 and self.products_checked >= self.products_to_check:
            print("   - Desired number of products already checked on this page.")
            # Move to next subtask if any remaining
            self.current_task_index += 1
            self.products_checked = 0
            return None

        # Screenshot-based selection is now the only supported option for choosing products
        if state.screenshot:
            print("   - On search results page. Asking LLM to choose a product.")
            action = await self._choose_product(state, current_goal, step)
        else:
            print("   - Screenshot-based selection is required but a screenshot is unavailable; scrolling or moving on.")
            action = None

        if action:
            return action  # Found something worth clicking / opening

        # No suitable products found in current viewport
        if state.pixels_below and state.pixels_below > 0:
            print("   - No suitable products visible. Scrolling down to load more products.")
            return {"scroll_down": ScrollAction(amount=None)}

        # Reached end of page and still nothing useful; move on
        print("   - Reached end of results without finding a suitable product.")
        self.current_task_index += 1
        self.products_checked = 0
        return None

    async def _think(self, state: BrowserStateSummary, step: int) -> Optional[Dict[str, Any]]:
        """
        The agent's 'brain'. Decides the next action based on the current page state.
        """
        print("🤔 Thinking...")
        print(f"   - Current URL: {state.url}")
        print(f"   - Persona: {self.persona}")
        
        current_goal = self.sub_tasks[self.current_task_index]
        print(f"   - Current Subtask: {current_goal}")

        if "etsy.com" not in state.url:
            return {"go_to_url": GoToUrlAction(url="https://www.etsy.com/")}

        if "etsy.com/search" in state.url:
            action = await self._choose_product_from_search(state, current_goal, step)
            # If _choose_product_from_search returns None, it means we should move to the next task
            # Check if we've moved to the next task
            if action is None and self.current_task_index < len(self.sub_tasks):
                # We've completed the current task, check if there are more tasks
                print(f"   - Completed task: {current_goal}")
                print(f"   - Moving to next task: {self.sub_tasks[self.current_task_index]}")
                # Clear search history for the new task to allow searching again
                search_history_to_remove = f"searched_{self.sub_tasks[self.current_task_index]}"
                if search_history_to_remove in self.history:
                    self.history.remove(search_history_to_remove)
                    print(f"   - Cleared search history for: {self.sub_tasks[self.current_task_index]}")
                # Return to Etsy homepage to start the next search
                return {"go_to_url": GoToUrlAction(url="https://www.etsy.com/")}
            return action

        elif "etsy.com/listing" in state.url:
            # 1️⃣ Analyse the product first
            if self.current_product_name:
                print(f"   - Analyzing product: {self.current_product_name}")
            await self._analyze_product_page(state, current_goal, step, self.current_product_name)

            # 2️⃣ Keep track that we've looked at another item
            self.products_checked += 1

            # 3️⃣ Work out which tab we're on so we can close it later
            try:
                current_tab_id = next((tab.page_id for tab in state.tabs if tab.url == state.url), None)
            except Exception:
                current_tab_id = None

            # 4️⃣ Finished analyzing this product; close the product tab and return to results.
            if current_tab_id is not None and current_tab_id != 0:
                return {"close_tab": CloseTabAction(page_id=current_tab_id)}

        search_bar = self._find_search_bar(state)
        if search_bar and f"searched_{current_goal}" not in self.history:
            print(f"   - Found search bar. Searching for '{current_goal}'.")
            return {
                "input_text": InputTextAction(index=search_bar, text=current_goal),
                "send_keys": SendKeysAction(keys="Enter"),
                "search_query": current_goal
            }

        print("   - No specific action decided.")
        return None

    async def _make_final_purchase_decision(self):
        """Use the LLM to decide which products to finally purchase based on memory."""
        if not self.memory.products:
            print("🤷 No products were analyzed, skipping final purchase decision.")
            return

        # Build a detailed list of products the agent has analyzed
        product_descriptions = []
        for idx, product in enumerate(self.memory.products, start=1):
            pros = ", ".join(product.pros) if product.pros else "-"
            cons = ", ".join(product.cons) if product.cons else "-"
            summary = product.summary if product.summary else "-"
            desc = (
                f"{idx}. {product.product_name}\n"
                f"URL: {product.url}\n"
                f"Pros: {pros}\n"
                f"Cons: {cons}\n"
                f"Summary: {summary}"
            )
            product_descriptions.append(desc)

        user_prompt_text = f"""
Persona: {self.persona}

Shopping Goal: {self.task}

Here are the products that have been analyzed:
{product_descriptions}
""".strip()

        messages = [
            SystemMessage(content=FINAL_DECISION_PROMPT),
            HumanMessage(content=user_prompt_text),
        ]

        print("🔮 Asking LLM for final purchase recommendations...")
        try:
            ai_message = await self.llm.ainvoke(messages)
            response_content = ai_message.content.strip()
            try:
                decision_json = json.loads(response_content)
            except json.JSONDecodeError:
                # If the model returned something that isn't valid JSON, fall back to raw string
                decision_json = {"raw_response": response_content}

            print("💡 Final purchase decision:")
            print(json.dumps(decision_json, indent=2))

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
                print(f"   - Final decision saved to {decision_path}")

        except Exception as e:
            print(f"   - An error occurred during final purchase decision making: {e}")

    async def run(self):
        """
        Main agent workflow for online shopping on Etsy.
        """
        print("🚀 Starting Etsy shopping agent...")
        print(f"   - Task: {self.task}")
        print(f"   - Persona: {self.persona}")

        # Begin screen recording before the browser launches so we capture the whole session
        if self.record_video:
            self._start_screen_recording()

        browser_profile = BrowserProfile(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            window_size={"width": self.viewport_width, "height": self.viewport_height},
        )

        self.browser_session = BrowserSession(
            keep_alive=True, 
            headless=self.headless,
            browser_profile=browser_profile
        )
        await self.browser_session.start()
        print("✅ Browser session started.")

        Action = self.controller.registry.create_action_model()

        for i in range(self.max_steps):
            print(f"\n--- Step {i+1}/{self.max_steps} ---")

            if self.current_task_index >= len(self.sub_tasks):
                print("   - All sub-tasks are completed.")
                break
            
            print("👀 Observing page state...")
            state = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=True)

            step = i + 1
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
                        print(f"   - Saved generic action debug info to {debug_file_path}")
                    except Exception as e:
                        print(f"   - Failed to save generic action debug info: {e}")

            if action_plan:
                print("🎬 Taking action...")
                actions_to_perform = []
                if "go_to_url" in action_plan:
                    actions_to_perform.append(Action(go_to_url=action_plan["go_to_url"]))
                    print(f"   - Navigating to {action_plan['go_to_url'].url}")
                if "input_text" in action_plan:
                    actions_to_perform.append(Action(input_text=action_plan["input_text"]))
                    self.history.append(f"searched_{action_plan['search_query']}")
                    # Keep track of search queries in memory
                    self.memory.add_search_query(action_plan["search_query"])
                    print(f"   - Typing '{action_plan['input_text'].text}'")
                if "send_keys" in action_plan:
                    actions_to_perform.append(Action(send_keys=action_plan["send_keys"]))
                    print(f"   - Pressing '{action_plan['send_keys'].keys}'")
                if "click_element_by_index" in action_plan:
                    actions_to_perform.append(Action(click_element_by_index=action_plan["click_element_by_index"]))
                    if "product_name" in action_plan:
                        self.history.append(f"clicked_product_{action_plan['product_name']}")
                        self.current_product_name = action_plan["product_name"]  # Store for analysis
                    print(f"   - Clicking element at index {action_plan['click_element_by_index'].index}")
                if "open_tab" in action_plan:
                    actions_to_perform.append(Action(open_tab=action_plan["open_tab"]))
                    if "product_name" in action_plan:
                        self.history.append(f"clicked_product_{action_plan['product_name']}")
                        self.current_product_name = action_plan["product_name"]  # Store for analysis
                    print(f"   - Opening new tab with {action_plan['open_tab'].url}")
                if "close_tab" in action_plan:
                    actions_to_perform.append(Action(close_tab=action_plan["close_tab"]))
                    self.current_product_name = None  # Clear after closing tab
                    print(f"   - Closing tab {action_plan['close_tab'].page_id}")
                if "switch_tab" in action_plan:
                    actions_to_perform.append(Action(switch_tab=action_plan["switch_tab"]))
                    print(f"   - Switching to tab {action_plan['switch_tab'].page_id}")
                if "scroll_down" in action_plan:
                    actions_to_perform.append(Action(scroll_down=action_plan["scroll_down"]))
                    print("   - Scrolling down the page")

                for action in actions_to_perform:
                    result = await self.controller.act(action=action, browser_session=self.browser_session)
                    print(f"   ✔️ Action result: {result.extracted_content}")
                    await asyncio.sleep(2)
                if self.manual:
                    input("Press Enter to continue...")
            else:
                print("   - No action to take. Continuing to next iteration to check for task transitions.")
                # Don't break here - continue to next iteration to allow task transitions
                # Only break if we've truly exhausted all options
                if self.current_task_index >= len(self.sub_tasks):
                    print("   - All sub-tasks completed. Ending.")
                    break
                await asyncio.sleep(1)  # Small delay before next iteration

        print("\n✅ Shopping agent finished.")

        if self.debug_path:
            memory_path = os.path.join(self.debug_path, "_memory.json")
            try:
                self.memory.save_to_json(memory_path)
                print(f"🧠 Memory saved to {memory_path}")
            except Exception as e:
                print(f"   - Failed to save memory: {e}", file=sys.stderr)

        # Decide on the final purchase based on the memory collected
        await self._make_final_purchase_decision()

        await asyncio.sleep(5)
        if self.browser_session:
            await self.browser_session.stop()
            await self.browser_session.kill()

        # Stop recorder if running
        if self.record_video:
            self._stop_screen_recording()

    def _start_screen_recording(self):
        """Spawn an ffmpeg process to capture the entire primary display as a single video."""
        if self._record_proc:
            return  # already recording

        os.makedirs(self.debug_path, exist_ok=True)
        output_path = os.path.join(self.debug_path, "session.mp4")

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
            print("⚠️  Screen recording not supported on this OS. Skipping video capture.")
            return

        try:
            self._record_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"🎥 Screen recording started → {output_path}")
        except FileNotFoundError:
            print("❌ ffmpeg not found. Install ffmpeg to enable screen recording.")
            self._record_proc = None

    def _stop_screen_recording(self):
        """Terminate ffmpeg process gracefully."""
        if not self._record_proc:
            return
        try:
            self._record_proc.terminate()
            self._record_proc.wait(timeout=10)
            print("🎬 Screen recording saved.")
        except Exception:
            try:
                self._record_proc.kill()
            except Exception:
                pass
        finally:
            self._record_proc = None

@click.command()
@click.option("--task", default=DEFAULT_TASK, help="The shopping task for the agent.")
@click.option("--persona", default=DEFAULT_PERSONA, help="The persona for the agent.")
@click.option("--manual", is_flag=True, help="Wait for user to press Enter after each agent action.")
@click.option("--headless", is_flag=True, help="Run the browser in headless mode.")
@click.option("--max-steps", default=10, help="The maximum number of steps the agent will take.")
@click.option("--debug-path", type=click.Path(), default="debug_run", help="Path to save debug artifacts, such as screenshots.")
@click.option("--width", default=1920, help="The width of the browser viewport.")
@click.option("--height", default=1080, help="The height of the browser viewport.")
@click.option("--n-products", default=3, help="Number of products to analyze from the search results (-1 for all).")
@click.option("--model", "model_name", default="gpt-4o-mini", help="Model name to use (e.g. gpt-4o, gpt-4o-mini).")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature for the language model (0-2).")
@click.option("--record-video", is_flag=True, help="Record the agent's browser session and save it to the debug path.")
def cli(task, persona, manual, headless, max_steps, debug_path, width, height, n_products, model_name, temperature, record_video):
    """Runs the Etsy Shopping Agent."""
    if headless and record_video:
        print("Error: --headless and --record-video options cannot be used together.", file=sys.stderr)
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
        products_to_check=n_products,
        model_name=model_name,
        temperature=temperature,
        record_video=record_video,
    )
    asyncio.run(agent.run())

if __name__ == "__main__":
    cli() 