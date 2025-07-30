"""
HTML parsing utilities for extracting product information from Etsy pages.
"""

import re
from typing import TYPE_CHECKING, Any, Dict, Optional
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from src.shopping_agent.agent import EtsyShoppingAgent


async def extract_product_data(agent: "EtsyShoppingAgent") -> Dict[str, Any]:
    """Extracts comprehensive product data from the current page including reviews, ratings, shipping info, and product name."""
    try:
        # Get the page HTML
        page_html = await agent.browser_session.execute_javascript(
            "() => document.documentElement.outerHTML"
        )
        
        soup = BeautifulSoup(page_html, 'html.parser')
        
        # Extract all product data
        review_data = _extract_reviews_and_ratings(soup, agent)
        shipping_data = _extract_shipping_info(soup, agent)
        image_data = _extract_product_image(soup, agent)
        price_data = _extract_product_price(soup, agent)
        product_name = extract_product_name_from_html(soup, agent)
        seller_name = _extract_seller_name(soup, agent)
        
        return {
            **review_data,
            **shipping_data,
            **image_data,
            **price_data,
            "product_name": product_name,
            "seller_name": seller_name
        }
    
    except Exception as e:
        agent._log(f"   - Failed to extract product data: {e}")
        return {
            "total_rating": "N/A",
            "total_reviews": "N/A", 
            "individual_reviews": [],
            "shipping_policy": {},
            "product_image_url": None,
            "product_price": "N/A",
            "product_name": "Unknown Product",
            "seller_name": "Unknown Seller"
        }


def _extract_reviews_and_ratings(soup: BeautifulSoup, agent: "EtsyShoppingAgent") -> Dict[str, Any]:
    """Extracts reviews and rating information from the soup."""
    try:
        # Extract rating information
        reviews_container = soup.find('div', class_='reviews-section') or soup.find('section', class_='reviews') or soup
        
        # Find the tag containing the overall rating text, e.g., "5 out of 5"
        total_rating_tag = reviews_container.find('p', class_='wt-text-title-large')
        total_rating = total_rating_tag.get_text(strip=True) if total_rating_tag else "N/A"

        # Find the tag containing the total number of reviews text, e.g., "(44 reviews)"
        total_reviews_tag = reviews_container.find('p', class_='wt-text-body-large')
        total_reviews = total_reviews_tag.get_text(strip=True) if total_reviews_tag else "N/A"
        
        # Extract individual review cards
        review_cards = soup.find_all('div', class_='review-card')

        if not review_cards:
            agent._log("   - No review cards found. The HTML structure might have changed.")
        
        extracted_reviews = []
        
        for card in review_cards:
            # Extract the review text
            review_text_tag = card.find('p', id=lambda x: x and x.startswith('review-preview-toggle-'))
            review_text = review_text_tag.get_text(strip=True) if review_text_tag else "N/A"
            
            # Extract the individual rating
            rating_span = card.find('span', {'data-stars-svg-container': True})
            if rating_span:
                rating_input = rating_span.find('input', {'name': 'initial-rating'})
                if rating_input and 'value' in rating_input.attrs:
                    rating = rating_input['value']
                else:
                    rating = "Rating not found"
            else:
                rating = "Rating not found"

            if review_text != "N/A":
                extracted_reviews.append({
                    "rating": rating,
                    "text": review_text
                })

        agent._log(f"   - Extracted rating: {total_rating}, total reviews: {total_reviews}")
        agent._log(f"   - Extracted {len(extracted_reviews)} individual review texts")
        
        return {
            "total_rating": total_rating,
            "total_reviews": total_reviews,
            "individual_reviews": extracted_reviews
        }
    
    except Exception as e:
        agent._log(f"   - Failed to extract reviews and ratings: {e}")
        return {
            "total_rating": "N/A",
            "total_reviews": "N/A", 
            "individual_reviews": []
        }


def _extract_shipping_info(soup: BeautifulSoup, agent: "EtsyShoppingAgent") -> Dict[str, Any]:
    """Extracts shipping and delivery information from the soup."""
    try:
        shipping_policy = {}
        shipping_section = soup.find('div', {'id': 'shipping-and-returns-div'})
        
        if shipping_section:
            # Using specific selectors for more stability instead of indexing
            
            # Find estimated delivery using its data attribute
            delivery_li = shipping_section.find('li', {'data-shipping-estimated-delivery': True})
            shipping_policy['estimated_delivery'] = delivery_li.get_text(strip=True, separator=' ').replace('\n', ' ') if delivery_li else "Not found"
            shipping_policy['estimated_delivery'] = shipping_policy['estimated_delivery'].split('Your order should arrive by this date if you buy today. To calculate')[0]

            # Find shipping cost using its data attribute
            cost_span = shipping_section.find('span', {'currency-value': True})
            cost_li = cost_span.find_parent('li') if cost_span else None
            shipping_policy['shipping_cost'] = cost_li.get_text(strip=True, separator=' ').replace('\n', ' ') if cost_li else "Not found"
        else:
            agent._log("   - No shipping section found")
            shipping_policy['estimated_delivery'] = "Not found"
            shipping_policy['shipping_cost'] = "Not found"
        
        agent._log(f"   - Extracted shipping info: delivery={shipping_policy.get('estimated_delivery')}, cost={shipping_policy.get('shipping_cost')}")
        
        return {
            "shipping_policy": shipping_policy
        }
    
    except Exception as e:
        agent._log(f"   - Failed to extract shipping info: {e}")
        return {
            "shipping_policy": {}
        }


def _extract_product_image(soup: BeautifulSoup, agent: "EtsyShoppingAgent") -> Dict[str, Any]:
    """Extracts the main product image URL from the soup."""
    try:
        main_image_tag = soup.find('img', {'data-carousel-first-image': True})
        if main_image_tag:
            # Prefer the high-resolution zoom image URL
            product_image_url = main_image_tag.get('data-src-zoom-image', main_image_tag.get('src'))
            agent._log(f"   - Extracted product image URL: {product_image_url}")
            return {
                "product_image_url": product_image_url
            }
        else:
            agent._log("   - No main product image found")
            return {
                "product_image_url": None
            }
    
    except Exception as e:
        agent._log(f"   - Failed to extract product image: {e}")
        return {
            "product_image_url": None
        }


def parse_price_to_float(price_string: str) -> Optional[float]:
    """
    Parses a price string like '$25.99', '€30.00', '£15.50' etc. into a float.
    Returns None if parsing fails.
    """
    if not price_string or price_string == "N/A":
        return None
    
    try:
        # Remove currency symbols, commas, and extra whitespace
        # This regex finds numbers with optional decimal places
        price_match = re.search(r'[\d,]+\.?\d*', price_string.replace(',', ''))
        if price_match:
            price_float = float(price_match.group())
            return price_float
        return None
    except (ValueError, AttributeError):
        return None


def _extract_product_price(soup: BeautifulSoup, agent: "EtsyShoppingAgent") -> Dict[str, Any]:
    """Extracts the product price from the soup."""
    try:
        # Find the price component using data-appears-component-name
        price_component = soup.find('div', {'data-appears-component-name': 'price'})
        
        if price_component:
            price_element = price_component.find('p', class_='wt-text-title-larger')
            if price_element:
                # Clean up the text, removing the screen-reader-only part.
                price_text = price_element.get_text(strip=True).replace('Price:', '')
                cleaned_price = price_text.replace('+', '').strip()
                
                # Parse the price to float
                price_float = parse_price_to_float(cleaned_price)
                
                agent._log(f"   - Successfully extracted price: {cleaned_price} (parsed: {price_float})")
                return {
                    "product_price": cleaned_price,
                    "product_price_float": price_float
                }
        
        agent._log("   - No price element found")
        return {
            "product_price": "N/A",
            "product_price_float": None
        }
    
    except Exception as e:
        agent._log(f"   - Failed to extract product price: {e}")
        return {
            "product_price": "N/A",
            "product_price_float": None
        }


def extract_product_name_from_html(soup: BeautifulSoup, agent: "EtsyShoppingAgent") -> str:
    """Extracts the product name from the HTML content of the product page."""
    try:
        # Find the H1 tag with the product title
        name_element = soup.find('h1', {'data-buy-box-listing-title': 'true'})
        if name_element:
            product_name = name_element.get_text(strip=True)
            agent._log("   - Successfully extracted product name from H1 tag.")
            return product_name
        
        agent._log("   - No product name found in HTML content")
        return "Unknown Product"
        
    except Exception as e:
        agent._log(f"   - Failed to extract product name from HTML: {e}")
        return "Unknown Product" 


def _extract_seller_name(soup: BeautifulSoup, agent: "EtsyShoppingAgent") -> str:
    """Extracts the seller name from the soup."""
    try:
        seller_link = soup.find('a', href=lambda href: href and "ref=shop-header-name" in href)
        
        # Check if the link was found.
        if seller_link:
            # The seller's name is the text within the link.
            # .strip() removes any leading/trailing whitespace.
            seller_name = seller_link.get_text(strip=True)
            agent._log(f"   - Successfully extracted seller name: {seller_name}")
            return seller_name
        
        agent._log("   - No seller name found")
        return "Unknown Seller"
        
    except Exception as e:
        agent._log(f"   - Failed to extract seller name: {e}")
        return "Unknown Seller" 