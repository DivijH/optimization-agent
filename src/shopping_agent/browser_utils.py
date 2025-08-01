import base64
import json
import re
import os
import io
import requests
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urljoin
from browser_use.browser.views import BrowserStateSummary
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from PIL import Image

from src.shopping_agent.memory import ProductMemory
from src.shopping_agent.prompts import FINAL_DECISION_PROMPT, PRODUCT_ANALYSIS_PROMPT
from src.shopping_agent.token_utils import update_token_usage
from src.shopping_agent.agent_actions import save_and_upload_debug_info
from src.shopping_agent.page_parser import extract_product_data

if TYPE_CHECKING:
    from src.shopping_agent.agent import EtsyShoppingAgent


async def download_and_resize_image(image_url: str, max_width: int = 400, quality: int = 70) -> Optional[str]:
    """Downloads an image from URL and resizes it to low resolution, returns base64."""
    try:
        # Download the image
        response = requests.get(image_url, stream=True, timeout=4)
        response.raise_for_status()
        
        # Open and resize the image
        img = Image.open(io.BytesIO(response.content))
        
        # Convert to RGB if necessary (for JPEG compatibility)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        # Calculate new dimensions maintaining aspect ratio
        width, height = img.size
        if width > max_width:
            new_height = int(height * (max_width / width))
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        img_bytes = buffer.getvalue()
        
        return base64.b64encode(img_bytes).decode('utf-8')
    
    except Exception as e:
        print(f"Failed to download and resize image from {image_url}: {e}")
        return None




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

    # Extract comprehensive product data (reviews, ratings, shipping, image URL, and product name)
    product_data = await extract_product_data(agent)
    
    # Use the product name extracted from HTML instead of the URL-based one
    extracted_product_name = product_data.get("product_name", product_name or "Unknown Product")
    agent._log(f"   - Extracted product name: {extracted_product_name}")
    
    # Download and resize the product image instead of taking screenshot
    product_images_b64 = []
    product_image_url = product_data.get("product_image_url")
    if product_image_url:
        agent._log(f"   - Downloading product image from: {product_image_url}")
        product_image_b64 = await download_and_resize_image(product_image_url, max_width=400, quality=60)
        if product_image_b64:
            product_images_b64 = [product_image_b64]
            agent._log("   - Successfully downloaded and resized product image")
        else:
            agent._log("   - Failed to download product image, falling back to screenshot")
            try:
                await agent.browser_session.remove_highlights()
            except Exception:
                pass
            screenshot_b64 = await agent.browser_session.take_screenshot(full_page=False)
            product_images_b64 = [screenshot_b64]
    else:
        agent._log("   - No product image URL found, falling back to screenshot")
        try:
            await agent.browser_session.remove_highlights()
        except Exception:
            pass
        screenshot_b64 = await agent.browser_session.take_screenshot(full_page=False)
        product_images_b64 = [screenshot_b64]
    
    parser = JsonOutputParser()

    system_prompt = PRODUCT_ANALYSIS_PROMPT
    
    # Include rating, reviews, price, seller, and shipping in the user prompt if available
    product_info_text = ""
    
    # Price information section
    if product_data.get("product_price") != "N/A" and product_data.get("product_price"):
        product_info_text += f"Product Price: {product_data['product_price']}\n\n"
    
    # Seller information section
    seller_name = product_data.get("seller_name")
    if seller_name and seller_name != "Unknown Seller":
        product_info_text += f"Seller: {seller_name}\n\n"
    
    # Rating and reviews section
    if product_data["total_rating"] != "N/A" or product_data["total_reviews"] != "N/A" or product_data["individual_reviews"]:
        product_info_text += f"Product Rating Information:\n"
        if product_data["total_rating"] != "N/A":
            product_info_text += f"Overall Rating: {product_data['total_rating']}"
        if product_data["total_reviews"] != "N/A":
            product_info_text += f" {product_data['total_reviews']}\n"
        
        if product_data["individual_reviews"]:
            product_info_text += f"\nFew Customer Reviews:\n"
            for i, review in enumerate(product_data["individual_reviews"][:10], 1):  # Limit to first 10 reviews
                rating_text = f"({review['rating']}/5 stars)" if review['rating'] != "Rating not found" else ""
                product_info_text += f"{i}. {rating_text} {review['text']}\n"
    
    # Shipping information section
    shipping_policy = product_data.get("shipping_policy", {})
    if shipping_policy and (shipping_policy.get("estimated_delivery") != "Not found" or shipping_policy.get("shipping_cost") != "Not found"):
        product_info_text += f"\nShipping & Delivery Information:\n"
        if shipping_policy.get("estimated_delivery") != "Not found":
            product_info_text += f"Estimated Delivery: {shipping_policy['estimated_delivery']}\n"
        if shipping_policy.get("shipping_cost") != "Not found":
            product_info_text += f"Shipping Cost: {shipping_policy['shipping_cost']}\n"
    
    user_prompt_text = f"Searched Query: {agent.task}\nProduct Name: {extracted_product_name}\nCurrent Date: {datetime.now().strftime('%B %d, %Y')}\n\n{product_info_text.strip()}"
    image_payloads = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in product_images_b64
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

        # Use the extracted price from webpage instead of asking the model
        extracted_price_float = product_data.get("product_price_float")
        
        product_memory = ProductMemory(
            product_name=extracted_product_name,
            url=state.url,
            price=extracted_price_float,  # Use extracted price instead of model response
            summary=analysis_response.get("summary", ""),
            semantic_score=analysis_response.get("semantic_score", ""),
        )

        agent.memory.add_product(product_memory)
        agent._log(f"   - Added '{product_memory.product_name}' to memory.")

        image_paths = []
        if (agent.save_local or agent.save_gcs) and agent.debug_path:
            os.makedirs(agent.debug_path, exist_ok=True)
            for b64 in product_images_b64:
                img_path = os.path.join(
                    agent.debug_path, f"product_image_step_{step}.jpg"
                )
                try:
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(b64))
                    image_paths.append(img_path)
                except Exception as e:
                    agent._log(f"   - Failed to save product image: {e}")

            if agent.save_gcs and agent.gcs_manager:
                for b64 in product_images_b64:
                    gcs_image_path = (
                        f"{agent.debug_path}/product_image_step_{step}.jpg"
                    )
                    await agent.gcs_manager.upload_string_to_gcs(
                        base64.b64decode(b64), gcs_image_path, content_type="image/jpeg"
                    )

        debug_info = {
            "type": "product_analysis",
            "input_url": state.url,
            "product_name": extracted_product_name,
            "prompt": {"system": system_prompt, "user_text": user_prompt_text},
            "product_images": image_paths,
            "product_data": product_data,
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
            
            # Always calculate total cost based on recommendations
            total_cost = 0.0
            product_price_map = {
                p.product_name.lower(): p.price
                for p in relevant_products
                if p.price is not None
            }
            recommendations = decision_json.get("recommendations", [])
            
            agent._log(f"💰 Calculating total cost for {len(recommendations)} recommended products:")

            for product_name in recommendations:
                product_found = False
                for product_name_in_map in product_price_map.keys():
                    if product_name.lower() in product_name_in_map:
                        total_cost += product_price_map[product_name_in_map]
                        agent._log(f"   - '{product_name}': ${product_price_map[product_name_in_map]:.2f}")
                        product_found = True
                        break
                if not product_found:
                    agent._log(f"   - '{product_name}': Price not found")
            
            # Always add the calculated total cost to the response
            decision_json["total_cost"] = round(total_cost, 2)
            agent._log(f"   - Total calculated cost: ${decision_json['total_cost']:.2f}")

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