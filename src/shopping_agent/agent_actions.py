import asyncio
import base64
import json
import os
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from browser_use.browser.views import BrowserStateSummary

if TYPE_CHECKING:
    from src.shopping_agent.agent import EtsyShoppingAgent


async def save_and_upload_screenshots(
    agent: "EtsyShoppingAgent", state: BrowserStateSummary, step: int
):
    if (agent.save_local or agent.save_gcs) and state.screenshot and agent.debug_path:
        # Save highlighted screenshot
        if agent.save_local:
            image_path = os.path.join(
                agent.debug_path, f"screenshot_step_{step}_with_boxes.png"
            )
            try:
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(state.screenshot))
                agent._log(f"   - Saved highlighted debug screenshot to {image_path}")
            except Exception as e:
                agent._log(f"   - Failed to save highlighted debug screenshot: {e}")

        # Upload highlighted screenshot
        if agent.save_gcs and agent.gcs_manager:
            gcs_image_path = (
                f"{agent.debug_path}/screenshot_step_{step}_with_boxes.png"
            )
            await agent.gcs_manager.upload_string_to_gcs(
                base64.b64decode(state.screenshot),
                gcs_image_path,
                content_type="image/png",
            )

    # Save plain screenshot
    if not agent.browser_session:
        return
    plain_screenshot_b64 = await agent.browser_session.take_screenshot()
    if (
        (agent.save_local or agent.save_gcs)
        and plain_screenshot_b64
        and agent.debug_path
    ):
        if agent.save_local:
            plain_image_path = os.path.join(
                agent.debug_path, f"screenshot_step_{step}_plain.png"
            )
            try:
                with open(plain_image_path, "wb") as f:
                    f.write(base64.b64decode(plain_screenshot_b64))
                agent._log(f"   - Saved plain debug screenshot to {plain_image_path}")
            except Exception as e:
                agent._log(f"   - Failed to save plain debug screenshot: {e}")

        # Upload plain screenshot
        if agent.save_gcs and agent.gcs_manager:
            gcs_plain_path = f"{agent.debug_path}/screenshot_step_{step}_plain.png"
            await agent.gcs_manager.upload_string_to_gcs(
                base64.b64decode(plain_screenshot_b64),
                gcs_plain_path,
                content_type="image/png",
            )


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
    agent: "EtsyShoppingAgent", debug_info: Dict, step: int
):
    if not agent.debug_path:
        return
    debug_file_path = os.path.join(agent.debug_path, f"debug_step_{step}.json")

    should_create_debug = not (agent.save_local and os.path.exists(debug_file_path))

    if should_create_debug:
        if agent.save_local:
            try:
                with open(debug_file_path, "w") as f:
                    json.dump(debug_info, f, indent=2)
                agent._log(f"   - Saved generic action debug info to {debug_file_path}")
            except Exception as e:
                agent._log(f"   - Failed to save generic action debug info: {e}")

        if agent.save_gcs and agent.gcs_manager:
            gcs_debug_path = f"{agent.debug_path}/debug_step_{step}.json"
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
            agent.history.append(f"searched_{action_plan['search_query']}")
            agent.memory.add_search_query(action_plan["search_query"])
    if "input_text" in action_plan:
        action = action_plan["input_text"]
        actions_to_perform.append(Action(input_text=action))
        agent.history.append(f"searched_{action_plan['search_query']}")
    if "send_keys" in action_plan:
        action = action_plan["send_keys"]
        actions_to_perform.append(Action(send_keys=action))
        agent._log(f"   - Pressing '{action.keys}'")
    if "click_element_by_index" in action_plan:
        action = action_plan["click_element_by_index"]
        actions_to_perform.append(Action(click_element_by_index=action))
        if "product_name" in action_plan:
            if "listing_id" in action_plan:
                agent.history.append(f"clicked_listing_{action_plan['listing_id']}")
            else:
                agent.history.append(f"clicked_product_{action_plan['product_name']}")
            agent.current_product_name = action_plan["product_name"]
        agent._log(f"   - Clicking element at index {action.index}")
    if "open_tab" in action_plan:
        action = action_plan["open_tab"]
        actions_to_perform.append(Action(open_tab=action))
        if "product_name" in action_plan:
            if "listing_id" in action_plan:
                agent.history.append(f"opened_listing_{action_plan['listing_id']}")
            else:
                agent.history.append(f"clicked_product_{action_plan['product_name']}")
            agent.current_product_name = action_plan["product_name"]
        agent._log(f"   - Opening new tab with {action.url}")
    if "close_tab" in action_plan:
        action = action_plan["close_tab"]
        actions_to_perform.append(Action(close_tab=action))
        agent.current_product_name = None
        agent._log(f"   - Closing tab {action.page_id}")
    if "switch_tab" in action_plan:
        action = action_plan["switch_tab"]
        actions_to_perform.append(Action(switch_tab=action))
        agent._log(f"   - Switching to tab {action.page_id}")
    if "scroll_down" in action_plan:
        action = action_plan["scroll_down"]
        actions_to_perform.append(Action(scroll_down=action))
        agent._log("   - Scrolling down the page")

    for action in actions_to_perform:
        if agent.browser_session:
            result = await agent.controller.act(
                action=action, browser_session=agent.browser_session
            )
            agent._log(f"   ✔️ Action result: {result.extracted_content}")
            await asyncio.sleep(2) 