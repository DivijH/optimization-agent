import asyncio
import base64
import json
import os
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

# from browser_use.browser.views import BrowserStateSummary

if TYPE_CHECKING:
    from src.shopping_agent.agent import EtsyShoppingAgent


# async def save_and_upload_screenshots(agent: "EtsyShoppingAgent", step: int):
#     """Save and upload screenshots to local and GCS."""
#     if not agent.browser_session:
#         return
    
#     # Remove highlights to get a screenshot
#     try:
#         await agent.browser_session.remove_highlights()
#     except Exception:
#         pass
    
#     plain_screenshot_b64 = await agent.browser_session.take_screenshot()
#     if (
#         (agent.save_local or agent.save_gcs)
#         and plain_screenshot_b64
#         and agent.debug_path
#     ):
#         if agent.save_local:
#             image_path = os.path.join(
#                 agent.debug_path, f"screenshot_step_{step}.png"
#             )
#             try:
#                 with open(image_path, "wb") as f:
#                     f.write(base64.b64decode(plain_screenshot_b64))
#                 agent._log(f"   - Saved debug screenshot to {image_path}")
#             except Exception as e:
#                 agent._log(f"   - Failed to save debug screenshot: {e}")

#         # Upload plain screenshot
#         if agent.save_gcs and agent.gcs_manager:
#             gcs_image_path = f"{agent.debug_path}/screenshot_step_{step}.png"
#             await agent.gcs_manager.upload_string_to_gcs(
#                 base64.b64decode(plain_screenshot_b64),
#                 gcs_image_path,
#                 content_type="image/png",
#             )


def create_debug_info(action_plan: Optional[Dict[str, Any]]) -> Optional[dict]:
    if action_plan is None:
        return None
    serializable_action_plan = {}
    for k, v in action_plan.items():
        if is_dataclass(v):
            serializable_action_plan[k] = asdict(v)
        else:
            serializable_action_plan[k] = str(v)
    return serializable_action_plan


async def save_and_upload_debug_info(
    agent: "EtsyShoppingAgent", debug_info: Dict, step
):
    if not agent.debug_path:
        return
    
    # If step starts with "_", use it as a custom filename
    if isinstance(step, str) and step.startswith("_"):
        filename = step if step.endswith('.json') else f"{step}.json"
        debug_file_path = os.path.join(agent.debug_path, filename)
        gcs_debug_path = f"{agent.debug_path}/{filename}"
        log_message = f"   - Saved custom debug info to {debug_file_path}"
    else:
        # Standard debug step format
        filename = f"debug_step_{step}.json"
        debug_file_path = os.path.join(agent.debug_path, filename)
        gcs_debug_path = f"{agent.debug_path}/{filename}"
        log_message = f"   - Saved generic action debug info to {debug_file_path}"

    should_create_debug = not (agent.save_local and os.path.exists(debug_file_path))

    if should_create_debug:
        if agent.save_local:
            try:
                with open(debug_file_path, "w") as f:
                    json.dump(debug_info, f, indent=2)
                agent._log(log_message)
            except Exception as e:
                agent._log(f"   - Failed to save debug info: {e}")

        if agent.save_gcs and agent.gcs_manager:
            await agent.gcs_manager.upload_string_to_gcs(
                json.dumps(debug_info, indent=2), gcs_debug_path
            )


async def handle_actions(agent: "EtsyShoppingAgent", action_plan, Action):
    actions_to_perform = []
    if "go_to_url" in action_plan:
        action = action_plan["go_to_url"]
        actions_to_perform.append(Action(go_to_url=action))
        agent._log(f"   - Navigating to {action.url}")
        if "search_query" in action_plan:
            # agent.history.append(f"searched_{action_plan['search_query']}")
            agent.memory.add_search_query(action_plan["search_query"])
    if "input_text" in action_plan:
        action = action_plan["input_text"]
        actions_to_perform.append(Action(input_text=action))
        # agent.history.append(f"searched_{action_plan['search_query']}")
    if "send_keys" in action_plan:
        action = action_plan["send_keys"]
        actions_to_perform.append(Action(send_keys=action))
        agent._log(f"   - Pressing '{action.keys}'")
    # if "click_element_by_index" in action_plan:
    #     action = action_plan["click_element_by_index"]
    #     actions_to_perform.append(Action(click_element_by_index=action))
    #     if "product_name" in action_plan:
    #         if "listing_id" in action_plan:
    #             agent.history.append(f"clicked_listing_{action_plan['listing_id']}")
    #         else:
    #             agent.history.append(f"clicked_product_{action_plan['product_name']}")
    #         agent.current_product_name = action_plan["product_name"]
    #     agent._log(f"   - Clicking element at index {action.index}")
    # Tab actions no longer needed with direct URL navigation optimization
    # if "open_tab" in action_plan:
    #     action = action_plan["open_tab"]
    #     actions_to_perform.append(Action(open_tab=action))
    #     if "product_name" in action_plan:
    #         # if "listing_id" in action_plan:
    #         #     agent.history.append(f"opened_listing_{action_plan['listing_id']}")
    #         # else:
    #         #     agent.history.append(f"clicked_product_{action_plan['product_name']}")
    #         agent.current_product_name = action_plan["product_name"]
    #     agent._log(f"   - Opening new tab with {action.url}")
    # if "close_tab" in action_plan:
    #     action = action_plan["close_tab"]
    #     actions_to_perform.append(Action(close_tab=action))
    #     # Only clear current_product_name if we're not opening another tab in the same action
    #     # (which happens in the optimized flow when switching directly between products)
    #     if "open_tab" not in action_plan:
    #         agent.current_product_name = None
    #     agent._log(f"   - Closing tab {action.page_id}")
    # if "switch_tab" in action_plan:
    #     action = action_plan["switch_tab"]
    #     actions_to_perform.append(Action(switch_tab=action))
    #     agent._log(f"   - Switching to tab {action.page_id}")
    # 
    # Handle product name setting for direct URL navigation
    # if "product_name" in action_plan and "go_to_url" in action_plan:
    #     agent.current_product_name = action_plan["product_name"]
    # if "scroll_down" in action_plan:
    #     action = action_plan["scroll_down"]
    #     actions_to_perform.append(Action(scroll_down=action))
    #     agent._log("   - Scrolling down the page")

    async def _execute_action_with_retry(action, max_retries=3):
        """Execute an action with retry logic for browser context errors."""
        for attempt in range(max_retries):
            try:
                if not agent.browser_session:
                    agent._log("   ❌ Browser session is None, skipping action")
                    return None
                
                # Check if browser context is still valid
                if hasattr(agent.browser_session, 'browser_context') and agent.browser_session.browser_context is None:
                    agent._log("   ❌ Browser context is None, attempting to restart session")
                    await agent._restart_browser_session()
                    if not agent.browser_session:
                        agent._log("   ❌ Failed to restart browser session")
                        return None
                
                result = await agent.controller.act(
                    action=action, browser_session=agent.browser_session
                )
                agent._log(f"   ✔️ Action result: {result.extracted_content}")
                return result
                
            except Exception as e:
                error_msg = str(e)
                if "Failed to connect to or create a new BrowserContext" in error_msg or "browser=None" in error_msg:
                    agent._log(f"   ⚠️ Browser context error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    if attempt < max_retries - 1:
                        # Wait before retry
                        await asyncio.sleep(0.3)
                        # Try to restart the browser session
                        try:
                            await agent._restart_browser_session()
                        except Exception as restart_error:
                            agent._log(f"   ❌ Failed to restart browser session: {restart_error}")
                    else:
                        agent._log(f"   ❌ Max retries reached for browser context error. Skipping action.")
                        return None
                else:
                    agent._log(f"   ❌ Action failed with unexpected error: {error_msg}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.2)
                    else:
                        agent._log(f"   ❌ Max retries reached for action error. Skipping action.")
                        return None
        
        return None

    for action in actions_to_perform:
        result = await _execute_action_with_retry(action)
        if result is not None:
            await asyncio.sleep(0.05)
        else:
            agent._log(f"   ⚠️ Action failed after retries, continuing with next action") 