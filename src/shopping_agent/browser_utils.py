import asyncio
import base64
import json
import re
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urljoin
from browser_use.browser.views import BrowserStateSummary
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from src.shopping_agent.memory import ProductMemory
from src.shopping_agent.prompts import FINAL_DECISION_PROMPT, PRODUCT_ANALYSIS_PROMPT
from src.shopping_agent.token_utils import update_token_usage
from src.shopping_agent.agent_actions import save_and_upload_debug_info

if TYPE_CHECKING:
    from src.shopping_agent.agent import EtsyShoppingAgent


def find_search_bar(state: BrowserStateSummary) -> Optional[int]:
    """Finds the search bar element on the page."""
    for index, element in state.selector_map.items():
        if (
            element.attributes.get("name") == "search_query"
            or element.attributes.get("id") == "global-enhancements-search-query"
            or element.attributes.get("aria-label", "").lower() == "search"
        ):
            return index
    return None


def extract_product_name_from_url(href: str) -> Optional[str]:
    """Extract product name from Etsy listing URL.

    URL pattern: /listing/<listing_id>/<product-name-with-dashes>...
    Example: /listing/719316991/giant-halloween-spider-photo-props
    """
    try:
        # Pattern to match /listing/<number>/<product-name-part>
        match = re.search(r"/listing/\d+/([^/?]+)", href)
        if match:
            product_slug = match.group(1)
            # Convert dashes to spaces and title case
            product_name = product_slug.replace("-", " ").title()
            return product_name
        return None
    except Exception:
        return None


async def scroll_and_collect(
    agent: "EtsyShoppingAgent",
    max_scrolls: int = 10,
    stop_at_bottom: bool = True,
    delay: float = 1.0,
) -> tuple[str, List[str]]:
    """Scrolls the current page multiple times, concatenating *all* text and viewport screenshots.
    Only scrolls the main webpage, not embedded scrollable elements like product galleries.
    """
    if not agent.browser_session:
        raise RuntimeError("Browser session not initialised - call run() first.")

    screenshots: List[str] = []
    text_chunks: List[str] = []

    try:
        await agent.browser_session.remove_highlights()
    except Exception:
        pass

    page = await agent.browser_session.get_current_page()

    MAIN_PAGE_SCROLL_JS = """(dy) => {
        const scrollOptions = { top: dy, behavior: 'smooth' };
        if (document.scrollingElement) {
            document.scrollingElement.scrollBy(scrollOptions);
        } else {
            const mainElement = document.documentElement || document.body;
            mainElement.scrollBy(scrollOptions);
        }
    }"""

    for i in range(max_scrolls):
        try:
            screenshot_b64 = await agent.browser_session.take_screenshot(full_page=False)
            screenshots.append(screenshot_b64)
        except Exception as e:
            agent._log(f"   - Failed to take screenshot during scroll capture: {e}")

        try:
            inner_text: str = await agent.browser_session.execute_javascript(
                "() => document.body.innerText"
            )
            if inner_text:
                text_chunks.append(inner_text)
        except Exception as e:
            agent._log(f"   - Failed to capture text during scroll capture: {e}")

        try:
            _, pixels_below = await agent.browser_session.get_scroll_info(page)
        except Exception:
            pixels_below = 0

        if stop_at_bottom and pixels_below <= 0 and i > 0:
            break

        try:
            dy = int(agent.viewport_height * 0.9)
            await page.evaluate(MAIN_PAGE_SCROLL_JS, dy)
        except Exception as e:
            agent._log(f"   - Failed to perform smooth scroll: {e}")
            try:
                await agent.browser_session._scroll_container(dy)
            except Exception:
                pass

        await asyncio.sleep(delay)

    full_text = "\n".join(text_chunks)
    return full_text, screenshots


async def analyze_product_page(
    agent: "EtsyShoppingAgent",
    state: BrowserStateSummary,
    step: int,
    product_name: Optional[str] = None,
) -> Optional[ProductMemory]:
    """Analyzes a product page and adds the product to memory."""
    agent._log("   - On a product page. Analyzing product details.")

    if agent.memory.get_product_by_url(state.url):
        agent._log("   - Already analyzed this product.")
        return None

    _, screenshots_b64 = await scroll_and_collect(
        agent, max_scrolls=3, stop_at_bottom=True, delay=0.3
    )
    screenshots_b64 = screenshots_b64[:3]  # cap at first 3 images
    parser = JsonOutputParser()

    system_prompt = PRODUCT_ANALYSIS_PROMPT
    user_prompt_text = f"Searched Query: {agent.task}\nCurrent Date: {datetime.now().strftime('%B %d, %Y')}"
    image_payloads = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        for b64 in screenshots_b64
    ]

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[{"type": "text", "text": user_prompt_text}, *image_payloads]),
    ]

    try:
        ai_message = await agent.llm.ainvoke(messages)
        if hasattr(ai_message, "usage_metadata") and ai_message.usage_metadata:

            await update_token_usage(
                agent, agent.llm.model_name, ai_message.usage_metadata, usage_type="analysis"
            )
        analysis_response = parser.parse(ai_message.content)

        product_memory = ProductMemory(
            product_name=product_name,
            url=state.url,
            price=analysis_response.get("price"),
            pros=analysis_response.get("pros", []),
            cons=analysis_response.get("cons", []),
            summary=analysis_response.get("summary", ""),
            semantic_score=analysis_response.get("semantic_score", ""),
        )

        agent.memory.add_product(product_memory)
        agent._log(f"   - Added '{product_memory.product_name}' to memory.")

        scroll_image_paths = []
        if (agent.save_local or agent.save_gcs) and agent.debug_path:
            os.makedirs(agent.debug_path, exist_ok=True)
            for idx, b64 in enumerate(screenshots_b64):
                img_path = os.path.join(
                    agent.debug_path, f"screenshot_step_{step}_scroll_{idx}.png"
                )
                try:
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(b64))
                    scroll_image_paths.append(img_path)
                except Exception as e:
                    agent._log(f"   - Failed to save scroll screenshot {idx}: {e}")

            if agent.save_gcs and agent.gcs_manager:
                for idx, b64 in enumerate(screenshots_b64):
                    gcs_scroll_path = (
                        f"{agent.debug_path}/screenshot_step_{step}_scroll_{idx}.png"
                    )
                    await agent.gcs_manager.upload_string_to_gcs(
                        base64.b64decode(b64), gcs_scroll_path, content_type="image/png"
                    )

        debug_info = {
            "type": "product_analysis",
            "input_url": state.url,
            "product_name": product_name,
            "prompt": {"system": system_prompt, "user_text": user_prompt_text},
            "scroll_screenshots": scroll_image_paths,
            "output": analysis_response,
        }

        if agent.save_local and agent.debug_path:
            file_path = os.path.join(agent.debug_path, f"debug_step_{step}.json")
            try:
                with open(file_path, "w") as f:
                    json.dump(debug_info, f, indent=2)
                agent._log(f"   - Saved product analysis debug info to {file_path}")
            except Exception as e:
                agent._log(f"   - Failed to save product analysis debug info: {e}")

        if agent.save_gcs and agent.gcs_manager:
            gcs_debug_path = f"{agent.debug_path}/debug_step_{step}.json"
            await agent.gcs_manager.upload_string_to_gcs(
                json.dumps(debug_info, indent=2), gcs_debug_path
            )

        return product_memory

    except Exception as e:
        agent._log(f"   - An error occurred during product analysis: {e}")
        return None

async def extract_all_listings_from_search(agent: "EtsyShoppingAgent") -> List[Dict[str, Any]]:
    """
    Extracts all product listings from the search results page by parsing the HTML directly.
    Returns a list of all available listings to be processed sequentially.
    """
    
    all_listings = []
    agent._log("🔍 Extracting all listings from search page HTML...")
    
    # Get the full HTML content of the page
    if not agent.browser_session:
        agent._log("   - No browser session available")
        return []
    
    try:
        page = await agent.browser_session.get_current_page()
        html_content = await page.content()
        agent._log(f"   - Retrieved HTML content: ({len(html_content)} characters)")
    except Exception as e:
        agent._log(f"   - Failed to get HTML content: {e}")
        return []
    
    # Parse HTML to find all product listings
    # Look for elements with data-listing-id and href containing /listing/
    listing_pattern = r'<a[^>]*?data-listing-id="(\d+)"[^>]*?href="([^"]*?/listing/[^"]*?)"[^>]*?>(.*?)</a>'
    matches = re.findall(listing_pattern, html_content, re.DOTALL)
    
    agent._log(f"   - Found {len(matches)} potential listings in HTML")
    
    for listing_id, href, inner_html in matches:
        # Skip if href is empty or invalid
        if not href or not href.strip():
            continue
            
        # Skip if already visited
        if listing_id in agent.visited_listing_ids:
            continue
            
        # Skip if already in our collection
        if any(listing["listing_id"] == listing_id for listing in all_listings):
            continue
            
        # Build full URL
        full_url = urljoin("https://www.etsy.com", href) if href.startswith("/") else href
        
        # Skip if already analyzed
        if agent.memory.get_product_by_url(full_url):
            continue
            
        # Extract product name from URL
        product_name = extract_product_name_from_url(href)
        if not product_name:
            # Try to extract from title attribute or inner text
            title_match = re.search(r'title="([^"]*)"', inner_html)
            if title_match:
                product_name = title_match.group(1).strip()
            else:
                # Extract text content from inner HTML
                text_content = re.sub(r'<[^>]*>', '', inner_html)
                text_content = re.sub(r'\s+', ' ', text_content).strip()
                product_name = text_content[:100] if text_content else f"Product {listing_id}"
        
        # Check for ads
        inner_html_lower = inner_html.lower()
        if ("ad from etsy seller" in inner_html_lower or 
            "ad by etsy seller" in inner_html_lower or
            "advertisement" in inner_html_lower or
            "sponsored" in inner_html_lower):
            continue
        
        all_listings.append({
            "product_name": product_name,
            "href": href,
            "listing_id": listing_id,
            "full_url": full_url,
        })
    
    # Save extraction debug information
    if (agent.save_local or agent.save_gcs) and agent.debug_path:
        
        extraction_debug_info = {
            "type": "html_listing_extraction",
            "total_listings_found": len(all_listings),
            "html_content_length": len(html_content),
            "raw_matches_found": len(matches),
            "listings_summary": [
                {
                    "listing_id": listing["listing_id"],
                    "product_name": listing["product_name"],
                    "href": listing["href"]
                }
                for listing in all_listings
            ]
        }
        
        await save_and_upload_debug_info(agent, extraction_debug_info, "_extracted_listings")
        agent._log("   - 💾 Saved HTML extraction debug info")
    
    agent._log(f"✅ Extracted {len(all_listings)} total listings from HTML")
    return all_listings


async def make_final_purchase_decision(agent: "EtsyShoppingAgent"):
    """Use the LLM to decide which products to finally purchase based on memory."""
    if not agent.memory.products:
        agent._log("🤷 No products were analyzed, skipping final purchase decision.")
        return

    relevant_products = [
        p
        for p in agent.memory.products
        if p.semantic_score.upper() in ["HIGHLY RELEVANT", "SOMEWHAT RELEVANT"]
    ]

    if not relevant_products:
        agent._log(
            "🤷 No relevant products found after filtering, skipping final purchase decision."
        )
        return

    product_descriptions = ""
    for idx, product in enumerate(relevant_products, start=1):
        summary = product.summary if product.summary else "-"
        price = (
            f"Price: ${product.price:.2f}"
            if product.price is not None
            else "Price: Not Available"
        )
        desc = f"{idx}. {product.product_name.title()}\nSummary: {summary}\n{price}"
        product_descriptions += desc + "\n\n"

    user_prompt_text = f"""
Searched Query: {agent.task}

Here are the products that have been analyzed:
{product_descriptions.strip()}
""".strip()

    messages = [
        SystemMessage(content=FINAL_DECISION_PROMPT),
        HumanMessage(content=user_prompt_text),
    ]

    agent._log("🔮 Asking LLM for final purchase recommendations...")
    try:
        ai_message = await agent.final_decision_llm.ainvoke(messages)
        if hasattr(ai_message, "usage_metadata") and ai_message.usage_metadata:
            await update_token_usage(
                agent,
                agent.final_decision_llm.model_name,
                ai_message.usage_metadata,
                usage_type="final_decision",
            )
        response_content = ai_message.content.strip()
        
        # Strip markdown code blocks if present
        if response_content.startswith("```json"):
            response_content = response_content[7:]  # Remove "```json"
        elif response_content.startswith("```"):
            response_content = response_content[3:]   # Remove "```"
        
        if response_content.endswith("```"):
            response_content = response_content[:-3]  # Remove trailing "```"
        
        response_content = response_content.strip()
        
        try:
            decision_json = json.loads(response_content)
            if "total_cost" not in decision_json or not isinstance(
                decision_json.get("total_cost"), (float, int)
            ):
                total_cost = 0.0
                product_price_map = {
                    p.product_name.lower(): p.price
                    for p in relevant_products
                    if p.price is not None
                }
                recommendations = decision_json.get("recommendations", [])
                for rec in recommendations:
                    product_name = rec.get("product_name", "").lower()
                    if product_name in product_price_map:
                        total_cost += product_price_map[product_name]
                decision_json["total_cost"] = round(total_cost, 2)

        except json.JSONDecodeError:
            decision_json = {"raw_response": response_content}

        agent._log("💡 Final purchase decision:")
        agent._log(json.dumps(decision_json, indent=2))

        if (agent.save_local or agent.save_gcs) and agent.debug_path:
            debug_info = {
                "type": "final_purchase_decision",
                "prompt": {
                    "system": FINAL_DECISION_PROMPT,
                    "user_text": user_prompt_text,
                },
                "output": decision_json,
            }

            if agent.save_local:
                decision_path = os.path.join(
                    agent.debug_path, "_final_purchase_decision.json"
                )
                with open(decision_path, "w") as f:
                    json.dump(debug_info, f, indent=2)
                agent._log(f"   - Final decision saved to {decision_path}")

            if agent.save_gcs and agent.gcs_manager:
                gcs_decision_path = (
                    f"{agent.debug_path}/_final_purchase_decision.json"
                )
                await agent.gcs_manager.upload_string_to_gcs(
                    json.dumps(debug_info, indent=2), gcs_decision_path
                )

    except Exception as e:
        agent._log(f"   - An error occurred during final purchase decision making: {e}") 