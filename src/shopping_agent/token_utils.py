import asyncio
import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.shopping_agent.config import IMAGE_TOKEN_PERCENTAGE, MODEL_PRICING, VENDOR_DISCOUNT_GEMINI

if TYPE_CHECKING:
    from src.shopping_agent.agent import EtsyShoppingAgent


async def save_token_usage(agent: "EtsyShoppingAgent"):
    """Calculates total cost and saves token usage to a JSON file in real-time."""
    if not agent.save_local and not agent.save_gcs:
        return

    # Calculate total session cost by summing all model costs
    total_session_cost = 0.0
    for model_usage in agent.token_usage.values():
        # Add analysis costs
        if "analysis" in model_usage:
            total_session_cost += model_usage["analysis"]["total_cost"]
        # Add final decision costs
        if "final_decision" in model_usage:
            total_session_cost += model_usage["final_decision"]["total_cost"]

    # Apply vendor discount for Gemini models
    total_session_cost_after_discount = total_session_cost
    vendor_discount_applied = 0.0
    if 'gemini' in agent.model_name.lower():
        total_session_cost_after_discount = total_session_cost * (1 - VENDOR_DISCOUNT_GEMINI)
        vendor_discount_applied = VENDOR_DISCOUNT_GEMINI

    token_usage_data = {
        "models": agent.token_usage,
        "total_session_cost": total_session_cost,
        "total_session_cost_after_discount": total_session_cost_after_discount,
        "vendor_discount_applied": vendor_discount_applied,
        "vendor_discount_percentage": f"{vendor_discount_applied * 100:.0f}%" if vendor_discount_applied > 0 else "0%",
    }

    # Save locally if enabled
    if agent.save_local and agent.debug_path:
        token_usage_path = os.path.join(agent.debug_path, "_token_usage.json")

        def _write_file_sync():
            # This function contains the blocking file I/O
            os.makedirs(agent.debug_path, exist_ok=True)
            with open(token_usage_path, "w") as f:
                json.dump(token_usage_data, f, indent=2)

        try:
            # Run the synchronous file write operation in a separate thread
            await asyncio.to_thread(_write_file_sync)
        except Exception as e:
            agent._log(
                f"   - Failed to save token usage locally in real-time: {e}",
                level="error",
            )

    # Upload to GCS if enabled
    if agent.save_gcs and agent.gcs_manager:
        gcs_file_path = f"{agent.debug_path or 'debug_run'}/_token_usage.json"
        await agent.gcs_manager.upload_string_to_gcs(
            json.dumps(token_usage_data, indent=2), gcs_file_path
        )


async def update_token_usage(
    agent: "EtsyShoppingAgent",
    model_name: str,
    usage_metadata: Optional[Dict[str, Any]],
    usage_type: str = "analysis",
):
    """Updates the token usage count for a given model."""
    if not usage_metadata:
        return

    input_tokens = usage_metadata.get("input_tokens", 0)
    output_tokens = usage_metadata.get("output_tokens", 0)
    total_tokens = usage_metadata.get("total_tokens", 0)

    if not total_tokens:
        return

    # Initialize model usage stats if not exists
    if model_name not in agent.token_usage:
        agent.token_usage[model_name] = {
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
                "total_cost": 0.0,
            },
            "final_decision": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
            },
        }

    # Update the appropriate usage type
    usage_data = agent.token_usage[model_name][usage_type]
    usage_data["input_tokens"] += input_tokens
    usage_data["output_tokens"] += output_tokens
    usage_data["total_tokens"] += total_tokens

    # Calculate costs
    if model_name in MODEL_PRICING:
        price_per_million = MODEL_PRICING[model_name]

        if usage_type == "analysis":
            # Calculate image and text tokens
            image_tokens = int(input_tokens * IMAGE_TOKEN_PERCENTAGE)
            text_tokens = input_tokens - image_tokens
            usage_data["image_tokens"] += image_tokens
            usage_data["text_tokens"] += text_tokens

            # Calculate input costs (text and image)
            input_text_cost = (text_tokens / 1_000_000) * price_per_million["input"]
            input_image_cost = (image_tokens / 1_000_000) * price_per_million["input"]
            output_cost = (output_tokens / 1_000_000) * price_per_million["output"]

            # Accumulate costs
            usage_data["input_text_cost"] += input_text_cost
            usage_data["input_image_cost"] += input_image_cost
            usage_data["input_total_cost"] += input_text_cost + input_image_cost
            usage_data["output_cost"] += output_cost
            usage_data["total_cost"] += input_text_cost + input_image_cost + output_cost
        else:  # for final_decision
            input_cost = (input_tokens / 1_000_000) * price_per_million["input"]
            output_cost = (output_tokens / 1_000_000) * price_per_million["output"]

            # Accumulate costs
            usage_data["input_cost"] += input_cost
            usage_data["output_cost"] += output_cost
            usage_data["total_cost"] += input_cost + output_cost

    # Log the usage details
    if usage_type == "analysis":
        agent._log(
            f"   - Token usage for {model_name} (analysis): "
            f"Input={input_tokens} (Image={usage_data['image_tokens']}, Text={usage_data['text_tokens']}), "
            f"Output={output_tokens}, Total={total_tokens}"
        )
        agent._log(
            f"   - Cost breakdown for {model_name} (analysis):"
            f"\n     Input Text: ${usage_data['input_text_cost']:.6f}"
            f"\n     Input Image: ${usage_data['input_image_cost']:.6f}"
            f"\n     Input Total: ${usage_data['input_total_cost']:.6f}"
            f"\n     Output: ${usage_data['output_cost']:.6f}"
            f"\n     Total: ${usage_data['total_cost']:.6f}"
        )
    else:  # for final_decision
        agent._log(
            f"   - Token usage for {model_name} (final_decision): "
            f"Input={input_tokens}, Output={output_tokens}, Total={total_tokens}"
        )
        agent._log(
            f"   - Cost breakdown for {model_name} (final_decision):"
            f"\n     Input: ${usage_data['input_cost']:.6f}"
            f"\n     Output: ${usage_data['output_cost']:.6f}"
            f"\n     Total: ${usage_data['total_cost']:.6f}"
        )

    # Save token usage to file in real-time
    await save_token_usage(agent) 