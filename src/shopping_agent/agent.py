import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin

import click
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.browser.views import BrowserStateSummary
from browser_use.controller.service import Controller
from browser_use.controller.views import GoToUrlAction
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from src.shopping_agent.memory import MemoryModule, ProductMemory

from src.shopping_agent.agent_actions import (
    create_debug_info,
    handle_actions,
    save_and_upload_debug_info,
)
from src.shopping_agent.browser_utils import (
    analyze_product_page,
    make_final_purchase_decision,
    extract_all_listings_from_search,
)
from src.shopping_agent.config import DEFAULT_TASK, MODEL_PRICING, VENDOR_DISCOUNT_GEMINI
from src.shopping_agent.gcs_utils import GCSManager


@dataclass
class EtsyShoppingAgent:
    """
    An agent that shops on Etsy based on a given task and persona.
    """

    task: str = DEFAULT_TASK
    curr_query: Optional[str] = None
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
    gcs_prefix: str = "mission-understanding/optimization-agent"
    gcs_project: str = "etsy-search-ml-dev"

    # Internal state, not initialized by the user
    browser_session: Optional[BrowserSession] = field(init=False, default=None)
    controller: Controller = field(init=False, default_factory=Controller)
    llm: BaseChatModel = field(init=False)
    final_decision_llm: BaseChatModel = field(init=False)
    memory: MemoryModule = field(init=False, default_factory=MemoryModule)
    current_product_name: Optional[str] = field(init=False, default=None)
    visited_listing_ids: set = field(init=False, default_factory=set)
    listings_queue: List[Dict[str, Any]] = field(init=False, default_factory=list)
    current_listing_index: int = field(init=False, default=0)
    listings_extracted: bool = field(init=False, default=False)
    token_usage: Dict[str, Dict[str, Any]] = field(init=False, default_factory=dict)
    _record_proc: Optional[subprocess.Popen] = field(
        init=False, default=None, repr=False
    )
    gcs_manager: Optional[GCSManager] = field(init=False, default=None, repr=False)
    # Time tracking
    _start_time: Optional[float] = field(init=False, default=None, repr=False)
    _start_timestamp: Optional[str] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        """Initializes the agent, handling debug path setup."""
        if not self.curr_query:
            self.curr_query = self.task

        self._setup_llms()
        self._setup_debug_path()
        if self.save_gcs:
            self.gcs_manager = GCSManager(self, project=self.gcs_project)

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
        
        try:
            self._log(f"   - Current URL: {state.url}")

            if "etsy.com" not in state.url:
                return self.navigate_to_etsy_search()

            if "etsy.com/search" in state.url:
                return await self.handle_search_page(state)

            elif "etsy.com/listing" in state.url:
                return await self.handle_listing_page(state, step)

            self._log("   - No specific action decided.")
            return None
            
        except Exception as e:
            self._log(f"   ❌ Error in _think method: {e}")
            # Return a safe fallback action instead of crashing
            if "etsy.com" not in state.url:
                self._log("   - Falling back to Etsy navigation")
                return self.navigate_to_etsy_search()
            else:
                self._log("   - No safe fallback action available")
                return None

    async def handle_search_page(self, state: BrowserStateSummary) -> Optional[Dict[str, Any]]:
        """
        Handles the search page by extracting all listings first, then processing them one by one.
        """
        try:
            # If we haven't extracted listings yet, extract them all first
            if not self.listings_extracted:
                self.listings_queue = await extract_all_listings_from_search(self)
                self.listings_extracted = True
                self.current_listing_index = 0
                
                if not self.listings_queue:
                    self._log("   - No listings found on search page.")
                    return None
                
                self._log(f"   - Ready to process {len(self.listings_queue)} listings sequentially.")
            
            # Process the next listing in the queue
            if self.current_listing_index < len(self.listings_queue):
                listing = self.listings_queue[self.current_listing_index]
                listing_id = listing["listing_id"]
                product_name = listing["product_name"]
                href = listing["href"]
                
                # Skip if we've already visited this listing
                if listing_id in self.visited_listing_ids:
                    self._log(f"   - Skipping already visited listing {self.current_listing_index + 1}/{len(self.listings_queue)}: '{product_name}' (ID: {listing_id})")
                    self.current_listing_index += 1
                    return await self.handle_search_page(state)  # Try next listing
                
                self._log(f"   - Processing listing {self.current_listing_index + 1}/{len(self.listings_queue)}: '{product_name}' (ID: {listing_id})")
                
                # Mark this listing as visited
                self.visited_listing_ids.add(listing_id)
                
                # Move to next listing for next iteration
                self.current_listing_index += 1
                
                # Build full URL
                if href.startswith("/"):
                    href = urljoin("https://www.etsy.com", href)
                
                return {
                    "go_to_url": GoToUrlAction(url=href),
                    "product_name": product_name,
                    "listing_id": listing_id,
                }
            
            # All listings processed
            self._log("   - All listings from search page have been processed.")
            return None
            
        except Exception as e:
            self._log(f"   ❌ Error handling search page: {e}")
            # Try to continue with remaining listings if possible
            if hasattr(self, 'current_listing_index') and hasattr(self, 'listings_queue'):
                if self.current_listing_index < len(self.listings_queue):
                    self.current_listing_index += 1
                    self._log(f"   - Skipping to next listing ({self.current_listing_index}/{len(self.listings_queue)})")
                    return await self.handle_search_page(state)
            self._log("   - No recovery possible from search page error")
            return None

    def navigate_to_etsy_search(self):
        search_query_encoded = quote(self.curr_query)
        search_url = f"https://www.etsy.com/search?q={search_query_encoded}"
        self._log(f"   - Initial navigation. Going to search page for '{self.curr_query}'.")
        return {
            "go_to_url": GoToUrlAction(url=search_url),
            "search_query": self.curr_query,
        }

    async def handle_listing_page(self, state, step):
        try:
            if self.current_product_name:
                self._log(f"   - Analyzing product: {self.current_product_name}")
            
            await analyze_product_page(self, state, step, self.current_product_name)

            # If there's another product in the queue, navigate directly to next product
            if (self.current_listing_index < len(self.listings_queue) and 
                self.listings_extracted):
                
                # Get the next listing
                listing = self.listings_queue[self.current_listing_index]
                listing_id = listing["listing_id"]
                product_name = listing["product_name"]
                href = listing["href"]
                
                self._log(f"   - Navigating directly to next product")
                self._log(f"   - Next product {self.current_listing_index + 1}/{len(self.listings_queue)}: '{product_name}' (ID: {listing_id})")
                
                # Mark this listing as visited and increment counter
                self.visited_listing_ids.add(listing_id)
                self.current_listing_index += 1
                
                # Build full URL
                if href.startswith("/"):
                    href = urljoin("https://www.etsy.com", href)
                
                return {
                    "go_to_url": GoToUrlAction(url=href),
                    "product_name": product_name,
                    "listing_id": listing_id,
                }
            
            # No more products to process, we're done
            self._log("   - No more products to analyze, analysis complete")
            return None
            
        except Exception as e:
            self._log(f"   ❌ Error handling listing page: {e}")
            # If there's an error but we have more products, try to navigate to the next one
            if (self.current_listing_index < len(self.listings_queue) and 
                self.listings_extracted):
                try:
                    listing = self.listings_queue[self.current_listing_index]
                    listing_id = listing["listing_id"]
                    product_name = listing["product_name"]
                    href = listing["href"]
                    
                    self._log(f"   - Error occurred, attempting to continue with next product: '{product_name}' (ID: {listing_id})")
                    
                    # Mark this listing as visited and increment counter
                    self.visited_listing_ids.add(listing_id)
                    self.current_listing_index += 1
                    
                    # Build full URL
                    if href.startswith("/"):
                        href = urljoin("https://www.etsy.com", href)
                    
                    return {
                        "go_to_url": GoToUrlAction(url=href),
                        "product_name": product_name,
                        "listing_id": listing_id,
                    }
                except Exception as recovery_error:
                    self._log(f"   - Recovery also failed: {recovery_error}")
            return None

    async def run(self):
        """Main agent workflow for online shopping on Etsy."""
        self._log("🚀 Starting Etsy shopping agent...")
        
        # Record start time for this agent run
        self._start_time = time.time()
        self._start_timestamp = datetime.now().isoformat()
        
        self.patch_browser_session_scroll()

        if self.record_video:
            self._start_screen_recording()

        await self.start_browser_session()
        
        # Initial health check
        await self._check_browser_session_health()
        
        Action = self.controller.registry.create_action_model()
        step = 0
        health_check_interval = 10  # Check health every 10 steps instead of every step
        while not self.max_steps or step < self.max_steps:
            step += 1
            self._log(f"\n--- Step {step}/{self.max_steps or '∞'} ---")

            # Check browser session health periodically or after errors
            should_check_health = (step % health_check_interval == 0) or (step == 1)
            if should_check_health:
                is_healthy = await self._check_browser_session_health()
                if not is_healthy:
                    self._log("   ⚠️ Browser session unhealthy, attempting restart...")
                    try:
                        await self._restart_browser_session()
                        is_healthy = await self._check_browser_session_health()
                        if not is_healthy:
                            self._log("   ❌ Browser session restart failed, ending agent run")
                            break
                    except Exception as e:
                        self._log(f"   ❌ Browser session restart failed: {e}")
                        break

            try:
                state = await self.browser_session.get_state_summary(
                    cache_clickable_elements_hashes=True
                )

                action_plan = await self._think(state, step)

                await self.execute_step(step, state, action_plan, Action)
                if not action_plan:
                    break
                    
            except Exception as e:
                self._log(f"   ❌ Error in step {step}: {e}")
                # Check if this is a browser-related error
                if "browser" in str(e).lower() or "context" in str(e).lower():
                    self._log("   ⚠️ Browser-related error detected, checking session health...")
                    await self._check_browser_session_health()
                # Continue with next step instead of crashing
                continue

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
        # Browser optimization for speed (keeping JS enabled for functionality)
        browser_args = [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-blink-features=AutomationControlled',
            '--disable-extensions',
            '--disable-plugins',
            '--disable-web-security',
            '--disable-features=TranslateUI',
            '--disable-background-networking',
            '--disable-background-timer-throttling',
            '--disable-client-side-phishing-detection',
            '--disable-default-apps',
            '--disable-hang-monitor',
            '--disable-popup-blocking',
            '--disable-prompt-on-repost',
            '--disable-sync',
            '--disable-domain-reliability',
            '--no-first-run',
            '--no-default-browser-check',
            '--fast-start',
            '--aggressive-cache-discard',
            '--memory-pressure-off',
            '--disable-background-media-suspend',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--disable-field-trial-config',
            '--disable-back-forward-cache',
            '--disable-ipc-flooding-protection'
        ]
        
        browser_profile = BrowserProfile(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            user_data_dir=self.user_data_dir,
            browser_args=browser_args,  # Add optimization flags
            user_agent="Mozilla/4.0 (compatible; Catchpoint)",
        )
        self.browser_session = BrowserSession(
            keep_alive=True, 
            chromium_sandbox=False,
            headless=self.headless, 
            browser_profile=browser_profile, 
        )
        await self.browser_session.start()
        self._log("✅ Browser session started with optimization flags.")

    async def _restart_browser_session(self):
        """Safely restart the browser session when the context becomes corrupted."""
        self._log("🔄 Restarting browser session...")
        
        # First, try to safely shut down the current session
        if self.browser_session:
            try:
                await self.browser_session.kill()
                self._log("   - Old browser session terminated")
            except Exception as e:
                self._log(f"   - Warning: Error terminating old session: {e}")
        
        # Clear the session reference
        self.browser_session = None
        
        # Add a small delay to let system resources free up
        await asyncio.sleep(0.5)
        
        # Create a new session
        try:
            await self.start_browser_session()
            self._log("   - New browser session started successfully")
        except Exception as e:
            self._log(f"   - Error starting new browser session: {e}")
            self.browser_session = None
            raise

    async def _check_browser_session_health(self):
        """Check the health of the browser session and log its state."""
        if not self.browser_session:
            self._log("   🔍 Browser session health: Session is None")
            return False
        
        try:
            # Check if browser context exists
            if not hasattr(self.browser_session, 'browser_context'):
                self._log("   🔍 Browser session health: No browser_context attribute")
                return False
            
            if self.browser_session.browser_context is None:
                self._log("   🔍 Browser session health: Browser context is None")
                return False
            
            # Try to get current state with required parameter
            state = await self.browser_session.get_state_summary(
                cache_clickable_elements_hashes=True
            )
            if state:
                active_tabs = len(state.tabs) if state.tabs else 0
                self._log(f"   🔍 Browser session health: OK (URL: {state.url}, {active_tabs} tabs)")
                return True
            else:
                self._log("   🔍 Browser session health: Could not get state summary")
                return False
                
        except Exception as e:
            self._log(f"   🔍 Browser session health: Error checking health - {e}")
            return False

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
        """Saves memory, makes final decision, calculates timing, and cleans up resources."""
        self._log("\n--- Shutting down and saving state ---")
        
        # Stop screen recording first
        if self._record_proc:
            self._stop_screen_recording()
        
        # Enhanced browser session cleanup
        if self.browser_session:
            try:
                self._log("🔄 Cleaning up browser session...")
                
                # First, try to close all tabs gracefully
                try:
                    state = await self.browser_session.get_state_summary(
                        cache_clickable_elements_hashes=True
                    )
                    if state and state.tabs:
                        for tab in state.tabs:
                            if tab.page_id != 0:  # Don't close the main tab
                                try:
                                    await self.browser_session.close_tab(tab.page_id)
                                except Exception as e:
                                    self._log(f"   - Warning: Could not close tab {tab.page_id}: {e}")
                except Exception as e:
                    self._log(f"   - Warning: Could not get state for tab cleanup: {e}")
                
                # Kill the browser session
                await self.browser_session.kill()
                self._log("   - Browser session terminated successfully")
                
            except Exception as e:
                self._log(f"   - Error during browser cleanup: {e}")
                # Force cleanup if normal shutdown fails
                try:
                    if hasattr(self.browser_session, 'browser') and self.browser_session.browser:
                        await self.browser_session.browser.close()
                    self._log("   - Forced browser cleanup completed")
                except Exception as force_error:
                    self._log(f"   - Force cleanup also failed: {force_error}")
            finally:
                self.browser_session = None
                self._log("   - Browser session reference cleared")
        
        # Save memory and scores
        await self.save_memory_and_scores()
        
        # Make final purchase decision
        await make_final_purchase_decision(self)
        
        # Log final token usage
        self.log_final_token_usage()
        
        # Calculate and save timing data
        await self._save_timing_data()

    async def _save_timing_data(self):
        """Calculate and save timing data to _time_taken.json"""
        if not self._start_time or not self.debug_path:
            return
            
        if not (self.save_local or self.save_gcs):
            return

        # Record end time and calculate duration
        end_time = time.time()
        end_timestamp = datetime.now().isoformat()
        total_time_seconds = end_time - self._start_time
        
        # Create timing data
        timing_data = {
            "start_time": self._start_timestamp,
            "end_time": end_timestamp,
            "total_time_seconds": round(total_time_seconds, 2),
            "total_time_formatted": f"{int(total_time_seconds // 3600):02d}:{int((total_time_seconds % 3600) // 60):02d}:{int(total_time_seconds % 60):02d}"
        }
        
        # Save timing data to _time_taken.json
        if self.save_local:
            try:
                time_file_path = os.path.join(self.debug_path, "_time_taken.json")
                with open(time_file_path, "w") as f:
                    json.dump(timing_data, f, indent=2)
                self._log(f"⏱️  Total time taken: {timing_data['total_time_formatted']} ({timing_data['total_time_seconds']} seconds)")
                self._log(f"📄 Timing data saved to _time_taken.json")
            except Exception as e:
                self._log(f"⚠️  Failed to save timing data: {e}", level="error")
        
        # Upload to GCS if enabled
        if self.save_gcs and self.gcs_manager:
            try:
                await self.gcs_manager.upload_string_to_gcs(
                    json.dumps(timing_data, indent=2), f"{self.debug_path}/_time_taken.json"
                )
            except Exception as e:
                self._log(f"⚠️  Failed to upload timing data to GCS: {e}", level="error")

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

        if 'gemini' in self.model_name.lower():
            total_session_cost_after_discount = total_session_cost * (1 - VENDOR_DISCOUNT_GEMINI)
            self._log(f"\n   💰 Total Session Cost (after discount): ${total_session_cost_after_discount:.6f}")
        else:
            self._log(f"\n   💰 Total Session Cost: ${total_session_cost:.6f}")
        
        if self.debug_path:
            self._log(
                f"   - Token usage saved to {os.path.join(self.debug_path, '_token_usage.json')}"
            ) 