import asyncio
import json
import os
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from src.shopping_agent.agent import EtsyShoppingAgent


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
            agent.memory.add_search_query(action_plan["search_query"])
    if "input_text" in action_plan:
        action = action_plan["input_text"]
        actions_to_perform.append(Action(input_text=action))
    if "send_keys" in action_plan:
        action = action_plan["send_keys"]
        actions_to_perform.append(Action(send_keys=action))
        agent._log(f"   - Pressing '{action.keys}'")

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