import json
import re
from pathlib import Path
import click
import sys
from typing import List
import os
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

CURRENT_DIR = Path(__file__).resolve().parent
project_root = str(CURRENT_DIR.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.shopping_agent.config import VENDOR_DISCOUNT_GEMINI

# Set up OpenAI credentials
try:
    key_path = CURRENT_DIR / "keys" / "litellm.key"
    os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()
except FileNotFoundError:
    raise Exception(
        "litellm.key file not found. It is expected in optimization-agent/src/keys/litellm.key"
    )
os.environ["OPENAI_API_BASE"] = "https://litellm.litellm.kn.ml-platform.etsy-mlinfra-dev.etsycloud.com"


def extract_listing_id_from_url(url: str) -> str:
    """Extract listing ID from Etsy URL."""
    if not url:
        return ""
    
    # Pattern to match Etsy listing URLs
    # Examples: 
    # https://www.etsy.com/listing/4295128956/higonokami-black-leather-pocket-knife
    # https://www.etsy.com/listing/1234567890/product-name?query=params
    match = re.search(r'/listing/(\d+)/', url)
    if match:
        return match.group(1)
    return ""


def score_to_numeric(semantic_score: str) -> int:
    """Convert semantic score to numeric value."""
    score_upper = semantic_score.upper().strip()
    if score_upper == "HIGHLY RELEVANT":
        return 2
    elif score_upper == "SOMEWHAT RELEVANT":
        return 1
    elif score_upper == "NOT RELEVANT":
        return 0
    else:
        return 0  # Default to 0 for unknown scores


def process_agent_results(debug_root: Path, model_name: str = "openai/o4-mini", temperature: float = 0.7):
    """
    Analyzes all agent runs in a directory, aggregates key metrics,
    and saves a summary CSV file.
    """
    print("\n" + "="*80)
    print(" " * 25 + "Aggregating Agent Run Data")
    print("="*80 + "\n")

    agent_dirs = sorted([d for d in debug_root.iterdir() if d.is_dir() and d.name.startswith('agent_')])

    if not agent_dirs:
        print(f"No agent debug directories found in '{debug_root}'. Nothing to process.")
        return

    print(f"Found {len(agent_dirs)} agent directories. Processing...")

    # Processing Token Usage
    process_token_usage(agent_dirs)

    # Processing Semantic Scores
    process_semantic_scores(agent_dirs)

    # Processing Final Decision
    process_final_decision(agent_dirs)

    # Processing Listing Rankings
    process_listing_rankings(agent_dirs)

    # Generate a sumamry from all the agents
    generate_summary(agent_dirs, model_name, temperature)


def process_token_usage(agent_dirs: List[Path]):
    """
    Process token usage data from all agent directories.
    """
    token_usage_data = {
        'models': {},
        'total_session_cost': 0.0,
        'total_session_cost_after_discount': 0.0,
        'vendor_discount_applied': 0.0,
        'vendor_discount_percentage': '0%',
    }
    
    total_discount_applied = 0.0
    agents_with_discount = 0
    
    for agent_dir in agent_dirs:
        token_file = agent_dir / "_token_usage.json"
        if not token_file.exists():
            print(f"  - Skipping {agent_dir.name}: `_token_usage.json` not found.")
            continue
        with open(token_file, 'r') as f:
            data = json.load(f)

        token_usage_data['total_session_cost'] += data['total_session_cost']
        token_usage_data['total_session_cost_after_discount'] += data.get('total_session_cost_after_discount', data['total_session_cost'])
        
        # Track vendor discount
        agent_discount = data.get('vendor_discount_applied', 0.0)
        if agent_discount > 0:
            total_discount_applied += agent_discount
            agents_with_discount += 1
        
        for model_name in data['models'].keys():
            if model_name not in token_usage_data['models']:
                token_usage_data['models'][model_name] = data['models'][model_name]
            else:
                token_usage_data['models'][model_name]['analysis']['input_tokens'] += data['models'][model_name]['analysis']['input_tokens']
                token_usage_data['models'][model_name]['analysis']['output_tokens'] += data['models'][model_name]['analysis']['output_tokens']
                token_usage_data['models'][model_name]['analysis']['total_tokens'] += data['models'][model_name]['analysis']['total_tokens']
                token_usage_data['models'][model_name]['analysis']['image_tokens'] += data['models'][model_name]['analysis']['image_tokens']
                token_usage_data['models'][model_name]['analysis']['text_tokens'] += data['models'][model_name]['analysis']['text_tokens']
                token_usage_data['models'][model_name]['analysis']['input_text_cost'] += data['models'][model_name]['analysis']['input_text_cost']
                token_usage_data['models'][model_name]['analysis']['input_image_cost'] += data['models'][model_name]['analysis']['input_image_cost']
                token_usage_data['models'][model_name]['analysis']['input_total_cost'] += data['models'][model_name]['analysis']['input_total_cost']
                token_usage_data['models'][model_name]['analysis']['output_cost'] += data['models'][model_name]['analysis']['output_cost']
                token_usage_data['models'][model_name]['analysis']['total_cost'] += data['models'][model_name]['analysis']['total_cost']
                token_usage_data['models'][model_name]['final_decision']['input_tokens'] += data['models'][model_name]['final_decision']['input_tokens']
                token_usage_data['models'][model_name]['final_decision']['output_tokens'] += data['models'][model_name]['final_decision']['output_tokens']
                token_usage_data['models'][model_name]['final_decision']['total_tokens'] += data['models'][model_name]['final_decision']['total_tokens']
                token_usage_data['models'][model_name]['final_decision']['input_cost'] += data['models'][model_name]['final_decision']['input_cost']
                token_usage_data['models'][model_name]['final_decision']['output_cost'] += data['models'][model_name]['final_decision']['output_cost']
                token_usage_data['models'][model_name]['final_decision']['total_cost'] += data['models'][model_name]['final_decision']['total_cost']

    # Calculate average costs
    token_usage_data['avg_total_cost'] = token_usage_data['total_session_cost'] / len(agent_dirs)
    token_usage_data['avg_total_cost_after_discount'] = token_usage_data['total_session_cost_after_discount'] / len(agent_dirs)
    
    # Calculate vendor discount statistics
    if agents_with_discount > 0:
        token_usage_data['vendor_discount_applied'] = total_discount_applied / agents_with_discount
        token_usage_data['vendor_discount_percentage'] = f"{token_usage_data['vendor_discount_applied'] * 100:.0f}%"
    
    # Add metadata about vendor discount
    token_usage_data['vendor_discount_metadata'] = {
        'agents_with_discount': agents_with_discount,
        'total_agents': len(agent_dirs),
        'discount_rate': VENDOR_DISCOUNT_GEMINI,
        'total_savings': token_usage_data['total_session_cost'] - token_usage_data['total_session_cost_after_discount']
    }

    output_file = agent_dirs[0].parent / "_token_usage.json"
    with open(output_file, 'w') as f:
        json.dump(token_usage_data, f, indent=4)
    print(f"  - Token usage aggregated and saved to: {output_file}")


def process_semantic_scores(agent_dirs: List[Path]):
    """
    Process semantic scores data from all agent directories.
    """
    aggregated_scores = {
        'page1_products': {
            'total': 0,
            'highly_relevant': 0,
            'somewhat_relevant': 0,
            'not_relevant': 0
        },
        'top_10_products': {
            'total': 0,
            'highly_relevant': 0,
            'somewhat_relevant': 0,
            'not_relevant': 0
        },
        'agent_count': 0,
        'agents_with_data': []
    }
    
    for agent_dir in agent_dirs:
        semantic_file = agent_dir / "_semantic_scores.json"
        if not semantic_file.exists():
            print(f"  - Skipping {agent_dir.name}: `_semantic_scores.json` not found.")
            continue
            
        try:
            with open(semantic_file, 'r') as f:
                data = json.load(f)
            
            # Aggregate page1_products data
            if 'page1_products' in data:
                page1 = data['page1_products']
                aggregated_scores['page1_products']['total'] += page1.get('total', 0)
                aggregated_scores['page1_products']['highly_relevant'] += page1.get('highly_relevant', 0)
                aggregated_scores['page1_products']['somewhat_relevant'] += page1.get('somewhat_relevant', 0)
                aggregated_scores['page1_products']['not_relevant'] += page1.get('not_relevant', 0)
            
            # Aggregate top_10_products data
            if 'top_10_products' in data:
                top10 = data['top_10_products']
                aggregated_scores['top_10_products']['total'] += top10.get('total', 0)
                aggregated_scores['top_10_products']['highly_relevant'] += top10.get('highly_relevant', 0)
                aggregated_scores['top_10_products']['somewhat_relevant'] += top10.get('somewhat_relevant', 0)
                aggregated_scores['top_10_products']['not_relevant'] += top10.get('not_relevant', 0)
            
            aggregated_scores['agents_with_data'].append(agent_dir.name)
            
        except json.JSONDecodeError as e:
            print(f"  - Error reading {agent_dir.name}: {e}")
        except Exception as e:
            print(f"  - Unexpected error with {agent_dir.name}: {e}")
    
    aggregated_scores['agent_count'] = len(aggregated_scores['agents_with_data'])
    
    # Calculate percentages
    for category in ['page1_products', 'top_10_products']:
        total = aggregated_scores[category]['total']
        if total > 0:
            aggregated_scores[category]['highly_relevant_pct'] = round(
                (aggregated_scores[category]['highly_relevant'] / total) * 100, 2
            )
            aggregated_scores[category]['somewhat_relevant_pct'] = round(
                (aggregated_scores[category]['somewhat_relevant'] / total) * 100, 2
            )
            aggregated_scores[category]['not_relevant_pct'] = round(
                (aggregated_scores[category]['not_relevant'] / total) * 100, 2
            )
    
    # Save aggregated semantic scores
    output_file = agent_dirs[0].parent / "_semantic_scores.json"
    with open(output_file, 'w') as f:
        json.dump(aggregated_scores, f, indent=4)
    
    print(f"  - Semantic scores aggregated and saved to: {output_file}")


def process_final_decision(agent_dirs: List[Path]):
    """
    Process final purchase decision data from all agent directories.
    """
    aggregated_decisions = {
        'total_agents': 0,
        'agents_with_decisions': [],
        'purchase_statistics': {
            'total_purchases': 0,
            'total_cost': 0.0,
            'average_cost_per_session': 0.0,
            'agents_who_purchased': 0,
            'agents_who_did_not_purchase': 0
        },
        'product_recommendations': {},
        'reasoning_themes': [],
        'individual_decisions': []
    }
    
    for agent_dir in agent_dirs:
        decision_file = agent_dir / "_final_purchase_decision.json"
        if not decision_file.exists():
            print(f"    - Skipping {agent_dir.name}: `_final_purchase_decision.json` not found.")
            continue
            
        try:
            with open(decision_file, 'r') as f:
                data = json.load(f)
            
            if 'output' in data:
                output = data['output']
                agent_decision = {
                    'agent': agent_dir.name,
                    'reasoning': output.get('reasoning', ''),
                    'recommendations': output.get('recommendations', []),
                    'total_cost': output.get('total_cost', 0.0),
                    'purchased': len(output.get('recommendations', [])) > 0
                }
                
                aggregated_decisions['individual_decisions'].append(agent_decision)
                aggregated_decisions['agents_with_decisions'].append(agent_dir.name)
                
                # Update purchase statistics
                if agent_decision['purchased']:
                    aggregated_decisions['purchase_statistics']['agents_who_purchased'] += 1
                    aggregated_decisions['purchase_statistics']['total_purchases'] += len(agent_decision['recommendations'])
                    aggregated_decisions['purchase_statistics']['total_cost'] += agent_decision['total_cost']
                else:
                    aggregated_decisions['purchase_statistics']['agents_who_did_not_purchase'] += 1
                
                # Collect reasoning for thematic analysis
                if agent_decision['reasoning']:
                    aggregated_decisions['reasoning_themes'].append(agent_decision['reasoning'])
            else:
                print(f"  - Warning: {agent_dir.name} missing 'output' field")
                
        except json.JSONDecodeError as e:
            print(f"  - Error reading {agent_dir.name}: {e}")
        except Exception as e:
            print(f"  - Unexpected error with {agent_dir.name}: {e}")
    
    # Calculate final statistics
    total_agents_with_decisions = len(aggregated_decisions['agents_with_decisions'])
    aggregated_decisions['total_agents'] = total_agents_with_decisions
    
    if total_agents_with_decisions > 0:
        aggregated_decisions['purchase_statistics']['average_cost_per_session'] = round(
            aggregated_decisions['purchase_statistics']['total_cost'] / total_agents_with_decisions, 2
        )
        aggregated_decisions['purchase_statistics']['purchase_rate_pct'] = round(
            (aggregated_decisions['purchase_statistics']['agents_who_purchased'] / total_agents_with_decisions) * 100, 2
        )
    
    # Save aggregated final decisions
    output_file = agent_dirs[0].parent / "_final_purchase_decision.json"
    with open(output_file, 'w') as f:
        json.dump(aggregated_decisions, f, indent=4)
    
    print(f"  - Final purchase decisions aggregated and saved to: {output_file}")


def process_listing_rankings(agent_dirs: List[Path]):
    """
    Process listing rankings by extracting product URLs, calculating average semantic scores,
    and ranking all products from all agents.
    """
    print("  - Processing listing rankings...")
    
    # Dictionary to store listing data: listing_id -> {agent_scores, product_info}
    listings_data = defaultdict(lambda: {
        'listing_id': '',
        'product_name': '',
        'url': '',
        'price': None,
        'total_score': 0,
        'avg_score': 0.0,
        'agent_count': 0,
        'agent_scores': {}  # agent_name -> score
    })

    total_products_processed = 0
    agents_processed = 0

    for agent_dir in agent_dirs:
        agent_name = agent_dir.name
        
        # Find all debug step files with product analysis
        debug_files = sorted([f for f in agent_dir.glob("debug_step_*.json")])
        agent_products = 0
        
        for debug_file in debug_files:
            try:
                with open(debug_file, 'r') as f:
                    data = json.load(f)
                
                # Check if this is a product analysis step
                if data.get('type') == 'product_analysis' and 'output' in data:
                    url = data.get('input_url', '')
                    listing_id = extract_listing_id_from_url(url)
                    
                    if not listing_id:
                        continue  # Skip if we can't extract listing ID
                    
                    output = data['output']
                    semantic_score = output.get('semantic_score', '')
                    numeric_score = score_to_numeric(semantic_score)
                    
                    # Store or update listing data
                    listing_info = listings_data[listing_id]
                    if not listing_info['listing_id']:  # First time seeing this listing
                        listing_info['listing_id'] = listing_id
                        listing_info['product_name'] = data.get('product_name', '')
                        listing_info['url'] = url
                        listing_info['price'] = output.get('price')
                    
                    # Add agent's score and details
                    listing_info['agent_scores'][agent_name] = {
                        'semantic_score': semantic_score,
                        'numeric_score': numeric_score
                    }
                    
                    agent_products += 1
                    total_products_processed += 1
                    
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                continue
        
        agents_processed += 1

    # Calculate averages and sort by score
    for listing_id, listing_info in listings_data.items():
        scores = [score_data['numeric_score'] for score_data in listing_info['agent_scores'].values()]
        listing_info['total_score'] = sum(scores)
        listing_info['avg_score'] = round(sum(scores) / len(scores), 3) if scores else 0.0
        listing_info['agent_count'] = len(scores)
    
    # Sort listings by average score (descending)
    sorted_listings = sorted(
        listings_data.values(), 
        key=lambda x: (x['avg_score'], x['agent_count']), 
        reverse=True
    )
    
    # Prepare final output
    result = {
        'metadata': {
            'total_agents_processed': agents_processed,
            'total_products_analyzed': total_products_processed,
            'unique_listings': len(listings_data),
            'scoring_system': {
                'HIGHLY RELEVANT': 2,
                'SOMEWHAT RELEVANT': 1,
                'NOT RELEVANT': 0
            }
        },
        'listings': sorted_listings
    }
    
    # Save results
    output_file = agent_dirs[0].parent / "_listing_order.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  - Listing rankings processed and saved to: {output_file}")


def generate_summary(agent_dirs: List[Path], model_name: str = "openai/o4-mini", temperature: float = 0.7):
    """
    Generate a summary from all the agents using LLM.
    """
    print('Generating summary...')

    with open(agent_dirs[0].parent / "_final_purchase_decision.json", 'r') as f:
        final_purchase_decision_data = json.load(f)
    
    feedbacks = []
    for decision in final_purchase_decision_data['individual_decisions']:
        feedbacks.append(decision['reasoning'])

    if not feedbacks:
        print("  - No feedbacks found. Skipping summary generation.")
        return

    # Load the prompt from file
    prompt_file = CURRENT_DIR / "summary_prompt.txt"
    try:
        with open(prompt_file, 'r') as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        raise Exception(f"⚠️ Error: Prompt file not found at {prompt_file}.")

    # Create LLM with specified model and temperature
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    
    # Format feedbacks for the prompt
    feedbacks_text = "\n\n".join([f"Agent Feedback {i+1}: {feedback}" for i, feedback in enumerate(feedbacks)])
    
    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(content=f"Here are the feedbacks from {len(feedbacks)} shopping agents:\n\n{feedbacks_text}")

    print(f"  - Calling LLM ({model_name}) to generate summary...")
    try:
        response = llm.invoke([system_message, human_message])
        summary = response.content
    except Exception as e:
        print(f"  - Error calling LLM: {e}")
        summary = f"Error generating summary: {e}"

    summary_data = {
        "summary": summary,
        "model_used": model_name,
        "temperature": temperature,
        "total_agents": len(final_purchase_decision_data['individual_decisions']),
        "purchase_rate_pct": final_purchase_decision_data['purchase_statistics'].get('purchase_rate_pct', 0),
        "average_cost_per_session": final_purchase_decision_data['purchase_statistics']['average_cost_per_session'],
        "reasoning_themes": final_purchase_decision_data['reasoning_themes']
    }

    summary_file = agent_dirs[0].parent / "_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"  - Summary generated and saved to: {summary_file}")


@click.command()
@click.option(
    "--debug-root",
    "debug_root",
    default=str(CURRENT_DIR / "debug_run"),
    show_default=True,
    help="Directory containing agent data to process.",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--model",
    "model_name",
    default="global-gemini-2.5-flash",
    show_default=True,
    help="LLM model to use for generating summary.",
)
@click.option(
    "--temperature",
    "temperature",
    default=0.7,
    show_default=True,
    help="Temperature for the LLM.",
)
def main(debug_root: Path, model_name: str, temperature: float):
    """
    Process agent results from a specified debug directory.
    """
    if not debug_root.exists():
        click.echo(f"Error: Directory '{debug_root.resolve()}' does not exist.", err=True)
        return

    process_agent_results(debug_root, model_name)


if __name__ == "__main__":
    main() 