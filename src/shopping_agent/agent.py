import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import click
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.browser.views import BrowserStateSummary
from browser_use.controller.service import Controller
from browser_use.controller.views import (
    CloseTabAction,
    InputTextAction,
    SendKeysAction,
    GoToUrlAction,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from src.shopping_agent.memory import MemoryModule, ProductMemory

from src.shopping_agent.agent_actions import (
    create_debug_info,
    handle_actions,
    save_and_upload_debug_info,
    save_and_upload_screenshots,
)
from src.shopping_agent.browser_utils import (
    analyze_product_page,
    choose_product_from_search,
    find_search_bar,
    make_final_purchase_decision,
)
from src.shopping_agent.config import DEFAULT_PERSONA, DEFAULT_TASK, MODEL_PRICING
from src.shopping_agent.gcs_utils import GCSManager


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
    model_name: str = "openai/o4-mini"
    final_decision_model_name: Optional[str] = None
    temperature: float = 0.7

    # Recording configuration
    record_video: bool = False

    # Storage configuration
    save_local: bool = True
    save_gcs: bool = True
    gcs_bucket_name: str = "training-dev-search-data-jtzn"
    gcs_prefix: str = "smu-agent-optimizer"

    # Internal state, not initialized by the user
    history: List[str] = field(init=False, default_factory=list)
    browser_session: Optional[BrowserSession] = field(init=False, default=None)
    controller: Controller = field(init=False, default_factory=Controller)
    llm: BaseChatModel = field(init=False)
    final_decision_llm: BaseChatModel = field(init=False)
    memory: MemoryModule = field(init=False, default_factory=MemoryModule)
    current_product_name: Optional[str] = field(init=False, default=None)
    visited_listing_ids: set = field(init=False, default_factory=set)
    token_usage: Dict[str, Dict[str, Any]] = field(init=False, default_factory=dict)
    _record_proc: Optional[subprocess.Popen] = field(
        init=False, default=None, repr=False
    )
    gcs_manager: Optional[GCSManager] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        """Initializes the agent, handling debug path setup."""
        self._setup_llms()
        self._setup_debug_path()
        if self.save_gcs:
            self.gcs_manager = GCSManager(self)
        self._log(f"Using task as the only search query: {self.task}")

    def _setup_llms(self):
        if self.model_name not in MODEL_PRICING:
            self._log(
                f"Model '{self.model_name}' not found in MODEL_PRICING. Aborting.",
                level="error",
            )
            sys.exit(1)
        if (
            self.final_decision_model_name
            and self.final_decision_model_name not in MODEL_PRICING
        ):
            self._log(
                f"Final decision model '{self.final_decision_model_name}' not found in MODEL_PRICING. Aborting.",
                level="error",
            )
            sys.exit(1)
        self.llm = ChatOpenAI(temperature=self.temperature, model=self.model_name)
        final_decision_model = self.final_decision_model_name or self.model_name
        self.final_decision_llm = ChatOpenAI(
            temperature=self.temperature, model=final_decision_model
        )

    def _setup_debug_path(self):
        if self.debug_path:
            if os.path.isdir(self.debug_path):
                # If running non-interactively and the directory already exists,
                # assume it was set up by the calling code (e.g., analyze_query.py)
                # and don't remove it to preserve any existing log files
                if not self.non_interactive and click.confirm(
                    f"Debug path '{self.debug_path}' already exists. Do you want to remove it and all its contents?",
                    default=False,
                ):
                    try:
                        shutil.rmtree(self.debug_path)
                        self._log(f"Removed existing debug path: {self.debug_path}")
                    except Exception as e:
                        self._log(f"Error removing debug path: {e}", level="error")
                        sys.exit(1)
                elif not self.non_interactive:
                    self._log(
                        "Aborting. Please choose a different debug path or remove the existing one manually."
                    )
                    sys.exit(0)
                else:
                    # Directory exists and we're non-interactive, so assume it's properly set up
                    self._log(f"Using existing debug directory: {self.debug_path}")
                    return
            
            try:
                os.makedirs(self.debug_path, exist_ok=True)
                self._log(f"Created debug directory: {self.debug_path}")
            except OSError as e:
                self._log(f"Could not create debug directory: {e}", level="error")
                sys.exit(1)

    def _log(self, message: str, level: str = "info"):
        """Logs a message using the provided logger or prints to stdout."""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            print(message, file=sys.stderr if level == "error" else sys.stdout)

    async def _think(
        self, state: BrowserStateSummary, step: int
    ) -> Optional[Dict[str, Any]]:
        self._log("🤔 Thinking...")
        self._log(f"   - Current URL: {state.url}")
        self._log(f"   - Current Task: {self.task}")

        if "etsy.com" not in state.url:
            return self.navigate_to_etsy_search()

        if "etsy.com/search" in state.url:
            return await choose_product_from_search(self, state)

        elif "etsy.com/listing" in state.url:
            return await self.handle_listing_page(state, step)

        if (
            search_bar := find_search_bar(state)
        ) and f"searched_{self.task}" not in self.history:
            return self.perform_search(search_bar)

        self._log("   - No specific action decided.")
        return None

    def navigate_to_etsy_search(self):
        search_query_encoded = quote(self.task)
        search_url = f"https://www.etsy.com/search?q={search_query_encoded}&application_behavior=default"
        self._log(f"   - Initial navigation. Going to search page for '{self.task}'.")
        return {
            "go_to_url": GoToUrlAction(url=search_url),
            "search_query": self.task,
        }

    async def handle_listing_page(self, state, step):
        if self.current_product_name:
            self._log(f"   - Analyzing product: {self.current_product_name}")
        await analyze_product_page(self, state, step, self.current_product_name)

        current_tab_id = next(
            (tab.page_id for tab in state.tabs if tab.url == state.url), None
        )

        if current_tab_id is not None and current_tab_id != 0:
            return {"close_tab": CloseTabAction(page_id=current_tab_id)}
        return None

    def perform_search(self, search_bar_index: int):
        self._log(f"   - Found search bar. Searching for '{self.task}'.")
        return {
            "input_text": InputTextAction(index=search_bar_index, text=self.task),
            "send_keys": SendKeysAction(keys="Enter"),
            "search_query": self.task,
        }

    async def run(self):
        """Main agent workflow for online shopping on Etsy."""
        self._log("🚀 Starting Etsy shopping agent...")
        self.patch_browser_session_scroll()

        if self.record_video:
            self._start_screen_recording()

        await self.start_browser_session()
        Action = self.controller.registry.create_action_model()
        step = 0
        while not self.max_steps or step < self.max_steps:
            step += 1
            self._log(f"\n--- Step {step}/{self.max_steps or '∞'} ---")

            state = await self.browser_session.get_state_summary(
                cache_clickable_elements_hashes=True
            )
            if "etsy.com/search" in state.url:
                await save_and_upload_screenshots(self, state, step)

            action_plan = await self._think(state, step)

            await self.execute_step(step, state, action_plan, Action)
            if not action_plan:
                break

        self._log("\n✅ Shopping agent finished.")

    def patch_browser_session_scroll(self):
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

            BrowserSession._scroll_container = _smooth_scroll_container
            BrowserSession._smooth_scroll_patched = True

    async def start_browser_session(self):
        browser_profile = BrowserProfile(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            user_data_dir=self.user_data_dir,
        )
        self.browser_session = BrowserSession(
            keep_alive=True, headless=self.headless, browser_profile=browser_profile
        )
        await self.browser_session.start()
        self._log("✅ Browser session started.")

    async def execute_step(self, step, state, action_plan, Action):
        serializable_action_plan = create_debug_info(action_plan)
        debug_info = {
            "type": "generic_action",
            "step": step,
            "url": state.url,
            "action_plan": serializable_action_plan,
        }
        await save_and_upload_debug_info(self, debug_info, step)

        if action_plan:
            self._log("🎬 Taking action...")
            await handle_actions(self, action_plan, Action)
            if self.manual:
                input("Press Enter to continue...")
        else:
            self._log("   - No action to take. Ending agent run.")

    def _start_screen_recording(self):
        if self._record_proc:
            return
        if not self.debug_path:
            self._log("Debug path not set, cannot record video.", level="warning")
            return
        os.makedirs(self.debug_path, exist_ok=True)
        output_path = os.path.join(self.debug_path, "_session.mp4")

        if sys.platform == "darwin":
            input_spec = "1:none"
            cmd = [
                "ffmpeg",
                "-y",
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
            self._log(
                "⚠️  Screen recording not supported on this OS. Skipping video capture."
            )
            return

        try:
            self._record_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self._log(f"🎥 Screen recording started → {output_path}")
        except FileNotFoundError:
            self._log("❌ ffmpeg not found. Install ffmpeg to enable screen recording.")
            self._record_proc = None

    def _stop_screen_recording(self):
        if not self._record_proc:
            return
        try:
            self._record_proc.terminate()
            self._record_proc.wait(timeout=10)
            self._log("🎬 Screen recording saved.")
        except Exception:
            self._record_proc.kill()
        finally:
            self._record_proc = None

    async def shutdown(self):
        """Saves memory, makes final decision, and cleans up resources."""
        self._log("\n--- Shutting down and saving state ---")
        if self._record_proc:
            self._stop_screen_recording()
        if self.browser_session:
            await self.browser_session.kill()
        await self.save_memory_and_scores()
        await make_final_purchase_decision(self)
        self.log_final_token_usage()

    async def save_memory_and_scores(self):
        if not self.save_local and not self.save_gcs:
            return
        if not self.debug_path:
            return

        memory_path = os.path.join(self.debug_path, "_memory.json")
        scores_path = os.path.join(self.debug_path, "_semantic_scores.json")

        if self.save_local:
            self.memory.save_to_json(memory_path)
            self._log(f"🧠 Memory saved to {memory_path}")

        if self.save_gcs and self.gcs_manager:
            memory_data = asdict(self.memory)
            await self.gcs_manager.upload_string_to_gcs(
                json.dumps(memory_data, indent=2), f"{self.debug_path}/_memory.json"
            )

        # Calculate and save semantic scores
        all_products = self.memory.products
        top_10_products = all_products[:10]
        semantic_scores_data = {
            "page1_products": self.calculate_scores(all_products),
            "top_10_products": self.calculate_scores(top_10_products),
        }

        if self.save_local:
            with open(scores_path, "w") as f:
                json.dump(semantic_scores_data, f, indent=2)
            self._log(f"📊 Semantic scores saved to {scores_path}")

        if self.save_gcs and self.gcs_manager:
            await self.gcs_manager.upload_string_to_gcs(
                json.dumps(semantic_scores_data, indent=2),
                f"{self.debug_path}/_semantic_scores.json",
            )

    def calculate_scores(self, products: List[ProductMemory]) -> Dict[str, Any]:
        scores = {"HIGHLY RELEVANT": 0, "SOMEWHAT RELEVANT": 0, "NOT RELEVANT": 0}
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

    def log_final_token_usage(self):
        self._log("📊 Final token usage:")
        total_session_cost = 0.0

        for model, usages in self.token_usage.items():
            model_total_cost = 0.0
            self._log(f"\n   Model: {model}")

            analysis = usages.get("analysis", {})
            analysis_total = analysis.get("total_cost", 0.0)
            model_total_cost += analysis_total
            if analysis:
                self._log(
                    f"   - Analysis:"
                    f"\n     Tokens:"
                    f"\n       Input: {analysis.get('input_tokens', 0)} (Image={analysis.get('image_tokens', 0)}, Text={analysis.get('text_tokens', 0)})"
                    f"\n       Output: {analysis.get('output_tokens', 0)}"
                    f"\n       Total: {analysis.get('total_tokens', 0)}"
                    f"\n     Costs:"
                    f"\n       Input Text: ${analysis.get('input_text_cost', 0.0):.6f}"
                    f"\n       Input Image: ${analysis.get('input_image_cost', 0.0):.6f}"
                    f"\n       Input Total: ${analysis.get('input_total_cost', 0.0):.6f}"
                    f"\n       Output: ${analysis.get('output_cost', 0.0):.6f}"
                    f"\n       Total: ${analysis_total:.6f}"
                )

            final = usages.get("final_decision", {})
            final_total = final.get("total_cost", 0.0)
            model_total_cost += final_total
            if final:
                self._log(
                    f"   - Final Decision:"
                    f"\n     Tokens:"
                    f"\n       Input: {final.get('input_tokens', 0)}"
                    f"\n       Output: {final.get('output_tokens', 0)}"
                    f"\n       Total: {final.get('total_tokens', 0)}"
                    f"\n     Costs:"
                    f"\n       Input: ${final.get('input_cost', 0.0):.6f}"
                    f"\n       Output: ${final.get('output_cost', 0.0):.6f}"
                    f"\n       Total: ${final_total:.6f}"
                )

            self._log(f"   - Model Total Cost: ${model_total_cost:.6f}")
            total_session_cost += model_total_cost

        self._log(f"\n   💰 Total Session Cost: ${total_session_cost:.6f}")
        if self.debug_path:
            self._log(
                f"   - Token usage saved to {os.path.join(self.debug_path, '_token_usage.json')}"
            ) 