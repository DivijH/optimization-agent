#!/usr/bin/env python3
"""
Single Listing Analyzer for Etsy Shopping Agent

This script analyzes a single Etsy listing for a given query and saves the semantic score.
Edit the global variables QUERY and LISTING_URL to change what you want to analyze.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from src.shopping_agent.browser_utils import download_and_resize_image
from src.shopping_agent.config import MODEL_PRICING
from src.shopping_agent.page_parser import extract_product_data
from src.shopping_agent.prompts import PRODUCT_ANALYSIS_PROMPT
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession

# =============================================================================
# GLOBAL VARIABLES - EDIT THESE TO CHANGE WHAT YOU WANT TO ANALYZE
# =============================================================================

# The search query you want to test against
QUERY = "winnie the pooh quilt kit"

# The Etsy listing URL you want to analyze
LISTING_ID = "1902732239"
LISTING_URL = f"https://www.etsy.com/listing/{LISTING_ID}"

# Model to use for analysis
MODEL_NAME = "global-gemini-2.5-flash"

# Temperatures to test (will run analysis 5 times with these different temperatures)
TEMPERATURES = [0, 0.25, 0.5, 0.75, 1.0]

# Output directory for results
OUTPUT_DIR = "debug_single_listing_analysis"

# =============================================================================
# END GLOBAL VARIABLES
# =============================================================================

# Set up LiteLLM credentials
try:
    os.environ["OPENAI_API_KEY"] = open("../keys/litellm.key").read().strip()
except FileNotFoundError:
    raise Exception(
        "litellm.key file not found. It is expected in optimization-agent/src/keys/litellm.key"
    )
os.environ["OPENAI_API_BASE"] = "https://litellm.litellm.kn.ml-platform.etsy-mlinfra-dev.etsycloud.com"


class SingleListingAnalyzer:
    """Analyzes a single Etsy listing for semantic relevance to a query."""
    
    def __init__(self, query: str, listing_url: str, model_name: str = "global-gemini-2.5-flash", output_dir: str = "single_listing_analysis", temperature: float = 0.7):
        self.query = query
        self.listing_url = listing_url
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(temperature=temperature, model=model_name)
        self.browser_session: Optional[BrowserSession] = None
        self.token_usage: Dict[str, Dict[str, Any]] = {}
        
        # Create query-specific subdirectory
        query_safe = self._sanitize_filename(query)
        self.output_dir = os.path.join(output_dir, query_safe)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate model
        if model_name not in MODEL_PRICING:
            raise ValueError(f"Model '{model_name}' not found in MODEL_PRICING. Available models: {list(MODEL_PRICING.keys())}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Convert query string to safe directory name."""
        import re
        # Replace spaces with underscores
        safe_name = filename.replace(" ", "_")
        # Remove or replace unsafe characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '', safe_name)
        # Limit length to avoid filesystem issues
        safe_name = safe_name[:50]
        # Remove trailing dots and spaces
        safe_name = safe_name.strip('. ')
        return safe_name if safe_name else "unknown_query"
    
    def _log(self, message: str, level: str = "info"):
        """Logs a message to stdout."""
        print(message)
    
    async def start_browser_session(self):
        """Start a browser session for web scraping."""
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
        ]
        
        browser_profile = BrowserProfile(
            viewport={"width": 1920, "height": 1080},
            browser_args=browser_args,
            user_agent="Mozilla/4.0 (compatible; Catchpoint)",
        )
        
        self.browser_session = BrowserSession(
            keep_alive=True, 
            chromium_sandbox=False,
            headless=False,
            browser_profile=browser_profile, 
        )
        await self.browser_session.start()
        self._log("✅ Browser session started.")
    
    async def navigate_to_listing(self):
        """Navigate to the listing URL."""
        if not self.browser_session:
            raise RuntimeError("Browser session not started")
        
        # Get the current page and navigate to the listing
        page = await self.browser_session.get_current_page()
        await page.goto(self.listing_url)
        self._log(f"🔗 Navigated to: {self.listing_url}")
        
        # Wait a moment for the page to load
        await asyncio.sleep(2)
    
    async def extract_product_data_from_page(self) -> Dict[str, Any]:
        """Extract product data from the current page using existing parser."""
        return await extract_product_data(self)
    
    async def analyze_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the product using the LLM."""
        self._log("🤔 Analyzing product with LLM...")
        
        # Download and resize product image
        product_images_b64 = []
        product_image_url = product_data.get("product_image_url")
        if product_image_url:
            self._log(f"   - Downloading product image from: {product_image_url}")
            product_image_b64 = await download_and_resize_image(product_image_url, max_width=400, quality=60)
            if product_image_b64:
                product_images_b64 = [product_image_b64]
                self._log("   - Successfully downloaded and resized product image")
            else:
                self._log("   - Failed to download product image")
        
        # Build the product info text
        product_info_text = ""
        
        # Price information
        if product_data.get("product_price") != "N/A" and product_data.get("product_price"):
            product_info_text += f"Product Price: {product_data['product_price']}\n\n"
        
        # Seller information
        seller_name = product_data.get("seller_name")
        if seller_name and seller_name != "Unknown Seller":
            product_info_text += f"Seller: {seller_name}\n\n"
        
        # Rating and reviews
        if product_data["total_rating"] != "N/A" or product_data["total_reviews"] != "N/A" or product_data["individual_reviews"]:
            product_info_text += f"Product Rating Information:\n"
            if product_data["total_rating"] != "N/A":
                product_info_text += f"Overall Rating: {product_data['total_rating']}"
            if product_data["total_reviews"] != "N/A":
                product_info_text += f" {product_data['total_reviews']}\n"
            
            if product_data["individual_reviews"]:
                product_info_text += f"\nFew Customer Reviews:\n"
                for i, review in enumerate(product_data["individual_reviews"][:10], 1):
                    rating_text = f"({review['rating']}/5 stars)" if review['rating'] != "Rating not found" else ""
                    product_info_text += f"{i}. {rating_text} {review['text']}\n"
        
        # Shipping information
        shipping_policy = product_data.get("shipping_policy", {})
        if shipping_policy and (shipping_policy.get("estimated_delivery") != "Not found" or shipping_policy.get("shipping_cost") != "Not found"):
            product_info_text += f"\nShipping & Delivery Information:\n"
            if shipping_policy.get("estimated_delivery") != "Not found":
                product_info_text += f"Estimated Delivery: {shipping_policy['estimated_delivery']}\n"
            if shipping_policy.get("shipping_cost") != "Not found":
                product_info_text += f"Shipping Cost: {shipping_policy['shipping_cost']}\n"
        
        # Create the user prompt
        user_prompt_text = f"Searched Query: {self.query}\nProduct Name: {product_data['product_name']}\nCurrent Date: {datetime.now().strftime('%B %d, %Y')}\n\n{product_info_text.strip()}"
        
        # Create image payloads
        image_payloads = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in product_images_b64
        ]
        
        # Create messages
        messages = [
            SystemMessage(content=PRODUCT_ANALYSIS_PROMPT),
            HumanMessage(content=[{"type": "text", "text": user_prompt_text}, *image_payloads]),
        ]
        
        try:
            # Get LLM response
            ai_message = await self.llm.ainvoke(messages)
            
            # Track token usage
            if hasattr(ai_message, "usage_metadata") and ai_message.usage_metadata:
                await self._update_token_usage(ai_message.usage_metadata)
            
            # Parse response
            parser = JsonOutputParser()
            analysis_response = parser.parse(ai_message.content)
            
            self._log(f"✅ Analysis complete!")
            self._log(f"   - Summary: {analysis_response.get('summary', 'N/A')}")
            self._log(f"   - Semantic Score: {analysis_response.get('semantic_score', 'N/A')}")
            
            return {
                "analysis": analysis_response,
                "product_data": product_data,
                "prompt": {
                    "system": PRODUCT_ANALYSIS_PROMPT,
                    "user_text": user_prompt_text
                },
                "query": self.query,
                "listing_url": self.listing_url,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            self._log(f"❌ Analysis failed: {e}")
            return {
                "error": str(e),
                "product_data": product_data,
                "query": self.query,
                "listing_url": self.listing_url,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _update_token_usage(self, usage_metadata: Dict[str, Any]):
        """Update token usage tracking."""
        input_tokens = usage_metadata.get("input_tokens", 0)
        output_tokens = usage_metadata.get("output_tokens", 0)
        total_tokens = usage_metadata.get("total_tokens", 0)

        if not total_tokens:
            return

        # Initialize model usage if not exists
        if self.model_name not in self.token_usage:
            self.token_usage[self.model_name] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
            }

        # Update usage
        usage_data = self.token_usage[self.model_name]
        usage_data["input_tokens"] += input_tokens
        usage_data["output_tokens"] += output_tokens
        usage_data["total_tokens"] += total_tokens

        # Calculate costs
        if self.model_name in MODEL_PRICING:
            price_per_million = MODEL_PRICING[self.model_name]
            input_cost = (input_tokens / 1_000_000) * price_per_million["input"]
            output_cost = (output_tokens / 1_000_000) * price_per_million["output"]
            total_cost = input_cost + output_cost
            usage_data["total_cost"] += total_cost

            self._log(f"   - Token usage: Input={input_tokens}, Output={output_tokens}, Total={total_tokens}")
            self._log(f"   - Cost: ${total_cost:.6f}")
    
    def save_results(self, results: Dict[str, Any], temperature: float):
        """Save analysis results to files with temperature suffix."""
        temp_suffix = f"_temp_{temperature:.2f}".replace(".", "_")
        
        # Save main results
        results_file = os.path.join(self.output_dir, f"analysis_results{temp_suffix}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        self._log(f"💾 Results saved to: {results_file}")
        
        # Save semantic score separately for easy access
        semantic_score_data = {
            "query": self.query,
            "listing_url": self.listing_url,
            "product_name": results.get("product_data", {}).get("product_name", "Unknown"),
            "semantic_score": results.get("analysis", {}).get("semantic_score", "N/A"),
            "summary": results.get("analysis", {}).get("summary", "N/A"),
            "temperature": temperature,
            "timestamp": results.get("timestamp"),
        }
        
        score_file = os.path.join(self.output_dir, f"semantic_score{temp_suffix}.json")
        with open(score_file, "w") as f:
            json.dump(semantic_score_data, f, indent=2)
        self._log(f"🎯 Semantic score saved to: {score_file}")
        
        return semantic_score_data
    
    def save_combined_results(self, all_results: List[Dict[str, Any]], all_semantic_scores: List[Dict[str, Any]]):
        """Save combined results from all temperature analyses."""
        # Save combined semantic scores summary
        combined_summary = {
            "query": self.query,
            "listing_url": self.listing_url,
            "product_name": all_semantic_scores[0].get("product_name", "Unknown") if all_semantic_scores else "Unknown",
            "model": self.model_name,
            "temperatures_tested": TEMPERATURES,
            "analyses": all_semantic_scores,
            "score_distribution": self._calculate_score_distribution(all_semantic_scores),
            "timestamp": datetime.now().isoformat()
        }
        
        summary_file = os.path.join(self.output_dir, "combined_analysis_summary.json")
        with open(summary_file, "w") as f:
            json.dump(combined_summary, f, indent=2)
        self._log(f"📊 Combined summary saved to: {summary_file}")
        
        # Save token usage summary
        if self.token_usage:
            token_file = os.path.join(self.output_dir, "total_token_usage.json")
            with open(token_file, "w") as f:
                json.dump(self.token_usage, f, indent=2)
            self._log(f"💰 Total token usage saved to: {token_file}")
    
    def _calculate_score_distribution(self, all_semantic_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate distribution of semantic scores across temperatures."""
        scores = [score.get("semantic_score", "N/A") for score in all_semantic_scores]
        distribution = {
            "HIGHLY RELEVANT": scores.count("HIGHLY RELEVANT"),
            "SOMEWHAT RELEVANT": scores.count("SOMEWHAT RELEVANT"), 
            "NOT RELEVANT": scores.count("NOT RELEVANT"),
            "N/A": scores.count("N/A"),
            "total": len(scores)
        }
        
        # Calculate percentages
        if distribution["total"] > 0:
            for key in ["HIGHLY RELEVANT", "SOMEWHAT RELEVANT", "NOT RELEVANT", "N/A"]:
                distribution[f"{key.lower().replace(' ', '_')}_percentage"] = (distribution[key] / distribution["total"]) * 100
        
        return distribution
    
    async def cleanup(self):
        """Clean up browser session."""
        if self.browser_session:
            try:
                await self.browser_session.kill()
                self._log("🧹 Browser session cleaned up.")
            except Exception as e:
                self._log(f"⚠️ Error during cleanup: {e}")
    
    async def run_single_analysis(self, temperature: float, product_data: Dict[str, Any]):
        """Run a single analysis with a specific temperature."""
        # Update the LLM with new temperature
        self.temperature = temperature
        self.llm = ChatOpenAI(temperature=temperature, model=self.model_name)
        
        self._log(f"🎯 Running analysis with temperature {temperature}")
        
        # Analyze with LLM
        results = await self.analyze_product(product_data)
        
        # Save results
        semantic_score_data = self.save_results(results, temperature)
        
        return results, semantic_score_data

    async def run(self):
        """Run the complete analysis 5 times with different temperatures."""
        all_results = []
        all_semantic_scores = []
        
        try:
            self._log("🚀 Starting multi-temperature single listing analysis...")
            self._log(f"   - Query: {self.query}")
            self._log(f"   - Listing URL: {self.listing_url}")
            self._log(f"   - Model: {self.model_name}")
            self._log(f"   - Temperatures: {TEMPERATURES}")
            self._log(f"   - Output Directory: {self.output_dir}")
            
            # Start browser and navigate (only once)
            await self.start_browser_session()
            await self.navigate_to_listing()
            
            # Extract product data (only once)
            self._log("📊 Extracting product data...")
            product_data = await self.extract_product_data_from_page()
            
            # Run analysis with each temperature
            for i, temperature in enumerate(TEMPERATURES, 1):
                self._log(f"\n--- Analysis {i}/{len(TEMPERATURES)} (Temperature: {temperature}) ---")
                
                results, semantic_score_data = await self.run_single_analysis(temperature, product_data)
                all_results.append(results)
                all_semantic_scores.append(semantic_score_data)
                
                # Small delay between analyses
                if i < len(TEMPERATURES):
                    await asyncio.sleep(1)
            
            # Save combined results
            self.save_combined_results(all_results, all_semantic_scores)
            
            self._log("\n✅ All analyses complete!")
            return all_results, all_semantic_scores
            
        except Exception as e:
            self._log(f"❌ Error during analysis: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main function to run the analyzer."""
    analyzer = SingleListingAnalyzer(
        query=QUERY,
        listing_url=LISTING_URL,
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR
    )
    
    try:
        all_results, all_semantic_scores = await analyzer.run()
        
        # Print final summary
        print("\n" + "="*70)
        print("📋 MULTI-TEMPERATURE ANALYSIS SUMMARY")
        print("="*70)
        print(f"Query: {QUERY}")
        print(f"Product: {all_semantic_scores[0].get('product_name', 'Unknown') if all_semantic_scores else 'Unknown'}")
        print(f"Model: {MODEL_NAME}")
        print(f"Temperatures tested: {TEMPERATURES}")
        print("\n🎯 SEMANTIC SCORES BY TEMPERATURE:")
        
        for i, (temp, score_data) in enumerate(zip(TEMPERATURES, all_semantic_scores)):
            print(f"  Temperature {temp:4.2f}: {score_data.get('semantic_score', 'N/A')}")
        
        # Calculate and show distribution
        scores = [score.get("semantic_score", "N/A") for score in all_semantic_scores]
        highly_relevant = scores.count("HIGHLY RELEVANT")
        somewhat_relevant = scores.count("SOMEWHAT RELEVANT")
        not_relevant = scores.count("NOT RELEVANT")
        
        print(f"\n📊 SCORE DISTRIBUTION:")
        print(f"  HIGHLY RELEVANT:   {highly_relevant}/{len(scores)} ({(highly_relevant/len(scores)*100):.1f}%)")
        print(f"  SOMEWHAT RELEVANT: {somewhat_relevant}/{len(scores)} ({(somewhat_relevant/len(scores)*100):.1f}%)")
        print(f"  NOT RELEVANT:      {not_relevant}/{len(scores)} ({(not_relevant/len(scores)*100):.1f}%)")
        
        # Show price if available
        if all_results and all_results[0].get('product_data', {}).get('product_price'):
            print(f"\n💰 Price: {all_results[0]['product_data']['product_price']}")
        
        print("="*70)
        query_safe = analyzer._sanitize_filename(QUERY)
        print(f"📁 Results saved in: {OUTPUT_DIR}/{query_safe}/")
        
    except KeyboardInterrupt:
        print("\n👋 Analysis interrupted by user.")
    except Exception as e:
        print(f"\n💥 Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())