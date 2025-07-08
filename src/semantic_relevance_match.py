import json
import re
import requests
import os
import time
from tqdm import tqdm
from collections import deque
from typing import List, Dict, Any

# Define the global parameter for the memory file
MEMORY_FILE_PATH = os.path.join(os.path.dirname(__file__), 'debug_run', '_memory.json')
RESULTS_FILE_PATH = os.path.join(os.path.dirname(__file__), 'debug_run', '_semantic_comparison_results.json')
MODEL_ENDPOINT = "https://dev-ml-platform.etsycloud.com/barista/smu-semrel-unified-serve-v2/v1/models/semrel-unified-serve-v2:predict"

class RateLimiter:
    """Rate limiter to ensure we don't exceed a specified RPS."""
    
    def __init__(self, max_rps: int = 50, verbose: bool = False):
        self.max_rps = max_rps
        self.requests = deque()
        self.verbose = verbose
    
    def wait_if_needed(self):
        """Wait if necessary to maintain the rate limit."""
        current_time = time.time()
        
        # Remove requests older than 1 second
        while self.requests and current_time - self.requests[0] > 1.0:
            self.requests.popleft()
        
        # If we've made too many requests in the last second, wait
        if len(self.requests) >= self.max_rps:
            sleep_time = 1.0 - (current_time - self.requests[0])
            if sleep_time > 0:
                if self.verbose:
                    print(f"Rate limiting: Sleeping for {sleep_time:.3f}s to maintain {self.max_rps} RPS limit")
                time.sleep(sleep_time)
                # Clean up old requests after sleeping
                current_time = time.time()
                while self.requests and current_time - self.requests[0] > 1.0:
                    self.requests.popleft()
        
        # Record this request
        self.requests.append(time.time())

# Global rate limiter instance
rate_limiter = RateLimiter(max_rps=50, verbose=True)

def classify_relevance(scores_relevant: float, scores_irrelevant: float, scores_partial: float) -> str:
    """Classify relevance based on the highest probability score."""
    scores = {
        'HIGHLY RELEVANT': scores_relevant,
        'SOMEWHAT RELEVANT': scores_irrelevant,
        'NOT RELEVANT': scores_partial
    }
    return max(scores, key=scores.get)

def extract_listing_id_from_url(url: str) -> str:
    """Extracts the listing ID from an Etsy product URL."""
    match = re.search(r'/listing/(\d+)/', url)
    if match:
        return match.group(1)
    return ""

def get_model_prediction(query: str, listing_id: str) -> Dict[str, Any]:
    """Calls the semantic relevance model and returns the prediction."""
    # Apply rate limiting before making the request
    rate_limiter.wait_if_needed()
    
    payload = {
        "signature_name": "score_listings",
        "inputs": {
            "query": query,
            "listing_ids": [int(listing_id)]
        }
    }
    try:
        response = requests.post(MODEL_ENDPOINT, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling model endpoint for listing {listing_id}: {e}")
        return {}

def save_results(search_query: str, listing_id: str, url: str, stored_summary: str, stored_score: str, predicted_classification: str, scores_relevant: float, scores_irrelevant: float, scores_partial: float, match: bool, high_mismatch: bool):
    if os.path.exists(RESULTS_FILE_PATH):
        with open(RESULTS_FILE_PATH, 'r') as f:
            results = json.load(f)
    else:
        results = []
    
    result = {
        "search_query": search_query,
        "listing_id": listing_id,
        "url": url,
        "stored_summary": stored_summary,
        "stored_score": stored_score,
        "predicted_classification": predicted_classification,
        "scores_relevant": scores_relevant,
        "scores_irrelevant": scores_irrelevant,
        "scores_partial": scores_partial,
        "match": match,
        "high_mismatch": high_mismatch
    }
    results.append(result)
    with open(RESULTS_FILE_PATH, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    """
    Goes through the memory.json file, compares semantic scores with the live model.
    """
    # Remove existing results file if it exists to start fresh
    if os.path.exists(RESULTS_FILE_PATH):
        os.remove(RESULTS_FILE_PATH)
        print(f"Removed existing results file: {RESULTS_FILE_PATH}")
    
    if not os.path.exists(MEMORY_FILE_PATH):
        print(f"Memory file not found at: {MEMORY_FILE_PATH}")
        return

    with open(MEMORY_FILE_PATH, 'r') as f:
        memory_data = json.load(f)

    search_query = memory_data.get("search_queries", [""])[0]
    if not search_query:
        print("No search query found in memory file.")
        return

    print(f"Using search query: '{search_query}'")
    print(f"Rate limiting enabled: Maximum {rate_limiter.max_rps} requests per second")
    
    products: List[Dict[str, Any]] = memory_data.get("products", [])
    matches = 0
    high_mismatches = 0

    for product in tqdm(products):
        listing_id = extract_listing_id_from_url(product.get('url', ''))
        stored_score = product.get("semantic_score", "N/A")
        
        # Listing ID not present in the URL
        if not listing_id:
            save_results(
                search_query=search_query,
                listing_id=None,
                url=product.get('url', ''),
                stored_summary=product.get('summary', ''),
                stored_score=stored_score,
                predicted_classification=None,
                scores_relevant=None,
                scores_irrelevant=None,
                scores_partial=None,
                match=False,
                high_mismatch=False
            )
            continue

        prediction = get_model_prediction(search_query, listing_id)
        
        if prediction:
            prediction = prediction['outputs']
            
            # Extract all the scores with safety checks for empty lists
            scores_relevant_list = prediction.get('scores_relevant', [])
            scores_irrelevant_list = prediction.get('scores_irrelevant', [])
            scores_partial_list = prediction.get('scores_partial', [])
            
            # Check if all score lists have at least one element
            if not (scores_relevant_list and scores_irrelevant_list and scores_partial_list):
                save_results(
                    search_query=search_query,
                    listing_id=listing_id,
                    url=product.get('url', ''),
                    stored_summary=product.get('summary', ''),
                    stored_score=stored_score,
                    predicted_classification=None,
                    scores_relevant=None,
                    scores_irrelevant=None,
                    scores_partial=None,
                    match=False,
                    high_mismatch=False
                )
                continue
            
            scores_relevant = scores_relevant_list[0]
            scores_irrelevant = scores_irrelevant_list[0]
            scores_partial = scores_partial_list[0]
            
            # Classify based on probabilities
            predicted_classification = classify_relevance(scores_relevant, scores_irrelevant, scores_partial)
            if predicted_classification == stored_score:
                matches += 1
            elif (predicted_classification == "HIGHLY RELEVANT" and stored_score == "NOT RELEVANT") or (predicted_classification == "NOT RELEVANT" and stored_score == "HIGHLY RELEVANT"):
                high_mismatches += 1

            # Store result
            save_results(
                search_query=search_query,
                listing_id=listing_id,
                url=product.get('url', ''),
                stored_summary=product.get('summary', ''),
                stored_score=stored_score,
                predicted_classification=predicted_classification,
                scores_relevant=scores_relevant,
                scores_irrelevant=scores_irrelevant,
                scores_partial=scores_partial,
                match=predicted_classification == stored_score,
                high_mismatch=(predicted_classification == "HIGHLY RELEVANT" and stored_score == "NOT RELEVANT") or (predicted_classification == "NOT RELEVANT" and stored_score == "HIGHLY RELEVANT")
            )

    print(f"\n\nResults saved to: {RESULTS_FILE_PATH}")
    print(f"Summary:")
    print(f"  Total products: {len(products)}")
    print(f"  Matches: {matches} : {matches / len(products) * 100:.2f}%")
    print(f"  Mismatches: {len(products) - matches} : {((len(products) - matches) / len(products) * 100):.2f}%")
    print(f"  High mismatches: {high_mismatches} : {high_mismatches / len(products) * 100:.2f}%")

if __name__ == "__main__":
    main()
    # output = get_model_prediction("wood backscratcher", "785973487")
    # print(output['outputs'])
    # print(output.keys())
