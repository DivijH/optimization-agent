#!/usr/bin/env python3
"""
Visualization and Analysis for Genetic Algorithm Optimization Results.

This script analyzes optimization results and provides insights into the
genetic algorithm's performance and evolution patterns.
"""

import json
import sys
from pathlib import Path
from typing import Dict
import click

# Add current directory to path for imports
CURRENT_DIR = Path(__file__).resolve().parent
project_root = str(CURRENT_DIR.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_optimization_results(results_path: Path) -> Dict:
    """Load optimization results from JSON file."""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing results file: {e}")
        sys.exit(1)


def show_original_query_context(results: Dict) -> None:
    """Show the original query that was being optimized."""
    print("🎯 Original Query Context")
    print("=" * 50)
    
    generation_history = results.get("generation_history", [])
    
    if generation_history:
        # Get the first generation (generation 0)
        first_generation = generation_history[0]
        
        # Find individuals without parent queries (original queries)
        original_individuals = [
            ind for ind in first_generation 
            if not ind.get("parent_queries") or len(ind.get("parent_queries", [])) == 0
        ]
        
        if original_individuals:
            if len(original_individuals) == 1:
                original_query = original_individuals[0]["query"]
                print(f"Original query: '{original_query}'")
                print(f"This analysis shows how the genetic algorithm evolved from this starting point.")
            else:
                print(f"Found {len(original_individuals)} original queries in the initial population:")
                for i, ind in enumerate(original_individuals, 1):
                    print(f"  {i}. '{ind['query']}'")
                print(f"This analysis shows how the genetic algorithm evolved from these starting points.")
        else:
            # Fallback: show all queries from first generation
            print("Initial population queries:")
            for i, ind in enumerate(first_generation, 1):
                print(f"  {i}. '{ind['query']}'")
            print(f"This analysis shows how the genetic algorithm evolved from these starting points.")
    else:
        print("❌ No generation history found in results.")
    
    print()


def show_performance_metrics(results: Dict) -> None:
    """Show timing, token usage, and cost information."""
    print("⏱️  Performance Metrics")
    print("=" * 50)
    
    config = results.get("optimization_config", {})
    summary = results.get("summary", {})
    
    # Time information
    total_time = summary.get("total_time_seconds")
    if total_time:
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        print(f"Total optimization time: {minutes}m {seconds}s ({total_time:.2f}s)")
    
    # Token usage
    token_usage = summary.get("token_usage", {})
    if token_usage:
        total_tokens = token_usage.get("total_tokens", 0)
        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)
        
        print(f"Token usage:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Input tokens: {input_tokens:,}")
        print(f"  Output tokens: {output_tokens:,}")
    
    # Cost information
    cost_info = summary.get("cost_info", {})
    if cost_info:
        total_cost = cost_info.get("total_cost", 0)
        input_cost = cost_info.get("input_cost", 0)
        output_cost = cost_info.get("output_cost", 0)
        
        print(f"Cost breakdown:")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Input cost: ${input_cost:.4f}")
        print(f"  Output cost: ${output_cost:.4f}")
    
    # Performance per generation
    avg_time_per_gen = summary.get("avg_time_per_generation")
    if avg_time_per_gen:
        print(f"Average time per generation: {avg_time_per_gen:.2f}s")
    
    print()


def analyze_evolution_trends(results: Dict) -> None:
    """Analyze and display evolution trends across generations."""
    print("📈 Evolution Trends Analysis")
    print("=" * 50)
    
    generation_history = results.get("generation_history", [])
    if not generation_history:
        print("❌ No generation history found in results.")
        return
    
    print(f"Total generations: {len(generation_history)}")
    print(f"Population size: {len(generation_history[0]) if generation_history else 0}")
    print()
    
    # Track best fitness across generations
    best_fitness_per_gen = []
    avg_fitness_per_gen = []
    
    for gen_num, generation in enumerate(generation_history):
        fitness_scores = [ind["fitness_score"] for ind in generation]
        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        
        best_fitness_per_gen.append(best_fitness)
        avg_fitness_per_gen.append(avg_fitness)
        
        best_individual = max(generation, key=lambda x: x["fitness_score"])
        print(f"Generation {gen_num}:")
        print(f"  Best fitness: {best_fitness:.3f}")
        print(f"  Average fitness: {avg_fitness:.3f}")
        print(f"  Best query: '{best_individual['query']}'")
        print()
    
    # Show improvement trend
    if len(best_fitness_per_gen) > 1:
        initial_best = best_fitness_per_gen[0]
        final_best = best_fitness_per_gen[-1]
        improvement = ((final_best - initial_best) / max(initial_best, 0.001)) * 100
        
        print(f"🚀 Overall Improvement: {improvement:.1f}%")
        print(f"   Initial best: {initial_best:.3f}")
        print(f"   Final best: {final_best:.3f}")
        print()


def analyze_query_diversity(results: Dict) -> None:
    """Analyze diversity of queries in each generation."""
    print("🌈 Query Diversity Analysis")
    print("=" * 50)
    
    generation_history = results.get("generation_history", [])
    
    for gen_num, generation in enumerate(generation_history):
        queries = [ind["query"] for ind in generation]
        unique_queries = set(queries)
        
        # Count word overlap
        all_words = set()
        for query in queries:
            all_words.update(query.lower().split())
        
        print(f"Generation {gen_num}:")
        print(f"  Unique queries: {len(unique_queries)}/{len(queries)}")
        print(f"  Unique words: {len(all_words)}")
        print(f"  Queries: {queries}")
        print()


def analyze_genetic_operations(results: Dict) -> None:
    """Analyze the impact of genetic operations."""
    print("🧬 Genetic Operations Analysis")
    print("=" * 50)
    
    generation_history = results.get("generation_history", [])
    
    # Track parent-child relationships
    parent_count = {}
    mutation_count = 0
    crossover_count = 0
    
    for gen_num, generation in enumerate(generation_history):
        if gen_num == 0:
            continue  # Skip initial generation
        
        for individual in generation:
            parent_queries = individual.get("parent_queries", [])
            
            if len(parent_queries) == 1:
                mutation_count += 1
            elif len(parent_queries) == 2:
                crossover_count += 1
            
            for parent in parent_queries:
                parent_count[parent] = parent_count.get(parent, 0) + 1
    
    print(f"Total mutations: {mutation_count}")
    print(f"Total crossovers: {crossover_count}")
    print()
    
    if parent_count:
        print("Most successful parent queries:")
        sorted_parents = sorted(parent_count.items(), key=lambda x: x[1], reverse=True)
        for parent, count in sorted_parents[:5]:
            print(f"  '{parent}': {count} offspring")
        print()


def analyze_semantic_performance(results: Dict) -> None:
    """Analyze semantic relevance performance for both whole page and top 10."""
    print("🎯 Semantic Performance Analysis")
    print("=" * 50)
    
    generation_history = results.get("generation_history", [])
    
    for gen_num, generation in enumerate(generation_history):
        whole_page_scores = []
        top10_scores = []
        purchase_scores = []
        
        for individual in generation:
            semantic_rel = individual.get("semantic_relevance", {})
            purchase_stats = individual.get("purchase_stats", {})
            
            if semantic_rel:
                # Whole page semantic relevance
                whole_page = semantic_rel.get("whole_page", {})
                if whole_page:
                    highly_relevant = whole_page.get("highly_relevant", 0)
                    total = whole_page.get("total", 1)
                    whole_page_score = highly_relevant / max(total, 1)
                    whole_page_scores.append(whole_page_score)
                
                # Top 10 semantic relevance
                top10 = semantic_rel.get("top10_products", {})
                if top10:
                    highly_relevant = top10.get("highly_relevant", 0)
                    total = top10.get("total", 1)
                    top10_score = highly_relevant / max(total, 1)
                    top10_scores.append(top10_score)
            
            if purchase_stats:
                stats = purchase_stats.get("purchase_statistics", {})
                agents_purchased = stats.get("agents_who_purchased", 0)
                total_agents = purchase_stats.get("total_agents", 1)
                purchase_score = agents_purchased / max(total_agents, 1)
                purchase_scores.append(purchase_score)
        
        print(f"Generation {gen_num}:")
        if whole_page_scores:
            avg_whole_page = sum(whole_page_scores) / len(whole_page_scores)
            print(f"  Avg whole page semantic relevance: {avg_whole_page:.3f}")
        
        if top10_scores:
            avg_top10 = sum(top10_scores) / len(top10_scores)
            print(f"  Avg top 10 semantic relevance: {avg_top10:.3f}")
        
        if purchase_scores:
            avg_purchase = sum(purchase_scores) / len(purchase_scores)
            print(f"  Avg purchase rate: {avg_purchase:.3f}")
        
        print()


def show_best_queries(results: Dict) -> None:
    """Show the best performing queries with enhanced semantic relevance details."""
    print("🏆 Best Performing Queries")
    print("=" * 50)
    
    best_individual = results.get("best_individual", {})
    
    if best_individual:
        print(f"Overall best query: '{best_individual['query']}'")
        print(f"Fitness score: {best_individual['fitness_score']:.3f}")
        print(f"Generation: {best_individual['generation']}")
        print()
        
        semantic_rel = best_individual.get("semantic_relevance", {})
        if semantic_rel:
            print("Semantic performance:")
            
            # Show whole page results
            whole_page = semantic_rel.get("whole_page", {})
            if whole_page:
                print(f"  Whole page results:")
                print(f"    Highly relevant: {whole_page.get('highly_relevant', 0)}")
                print(f"    Somewhat relevant: {whole_page.get('somewhat_relevant', 0)}")
                print(f"    Not relevant: {whole_page.get('not_relevant', 0)}")
                print(f"    Total products: {whole_page.get('total', 0)}")
                print()
            
            # Show top 10 results
            top10 = semantic_rel.get("top10_products", {})
            if top10:
                print(f"  Top 10 results:")
                print(f"    Highly relevant: {top10.get('highly_relevant', 0)}")
                print(f"    Somewhat relevant: {top10.get('somewhat_relevant', 0)}")
                print(f"    Not relevant: {top10.get('not_relevant', 0)}")
                print(f"    Total products: {top10.get('total', 0)}")
                print()
        
        purchase_stats = best_individual.get("purchase_stats", {})
        if purchase_stats:
            stats = purchase_stats.get("purchase_statistics", {})
            print(f"Purchase performance:")
            print(f"  Agents who purchased: {stats.get('agents_who_purchased', 0)}")
            print(f"  Total agents: {purchase_stats.get('total_agents', 0)}")
            print(f"  Total purchases: {stats.get('total_purchases', 0)}")
            print()
    
    # Show top queries from final generation
    generation_history = results.get("generation_history", [])
    if generation_history:
        final_generation = generation_history[-1]
        sorted_final = sorted(final_generation, key=lambda x: x["fitness_score"], reverse=True)
        
        print("Top 3 queries from final generation:")
        for i, individual in enumerate(sorted_final[:3], 1):
            print(f"  {i}. '{individual['query']}' - Fitness: {individual['fitness_score']:.3f}")
        print()


def generate_summary_report(results: Dict) -> None:
    """Generate a comprehensive summary report."""
    print("📊 Optimization Summary Report")
    print("=" * 60)
    
    config = results.get("optimization_config", {})
    summary = results.get("summary", {})
    
    print(f"Configuration:")
    print(f"  Population size: {config.get('population_size', 'N/A')}")
    print(f"  Generations: {config.get('n_generations', 'N/A')}")
    print(f"  Mutation rate: {config.get('mutation_rate', 'N/A')}")
    print(f"  Crossover rate: {config.get('crossover_rate', 'N/A')}")
    print(f"  Whole page semantic weight: {config.get('whole_page_semantic_weight', 'N/A')}")
    print(f"  Top10 semantic weight: {config.get('top10_semantic_weight', 'N/A')}")
    print(f"  Purchase weight: {config.get('purchase_weight', 'N/A')}")
    print()
    
    print(f"Results:")
    print(f"  Best fitness: {summary.get('best_fitness', 'N/A'):.3f}")
    print(f"  Best query: '{summary.get('best_query', 'N/A')}'")
    
    improvement = summary.get("improvement", {})
    rel_improvement = improvement.get("relative_improvement", 0)
    print(f"  Improvement: {rel_improvement:.1f}%")
    print()


@click.command()
@click.option(
    "--results-path",
    type=click.Path(exists=True, path_type=Path),
    default=CURRENT_DIR / "debug_ga" / "optimization_results.json",
    show_default=True,
    help="Path to the optimization results JSON file."
)
@click.option(
    "--analysis",
    type=click.Choice(["all", "evolution", "diversity", "genetics", "semantic", "best", "summary"]),
    default="all",
    show_default=True,
    help="Type of analysis to perform."
)
def main(results_path: Path, analysis: str):
    """
    Analyze and visualize genetic algorithm optimization results.
    
    This tool provides insights into how the genetic algorithm evolved
    queries and the performance improvements achieved.
    """
    
    print("🧬 Genetic Algorithm Results Analyzer")
    print("=" * 60)
    print(f"Analyzing results from: {results_path}")
    print()
    
    results = load_optimization_results(results_path)
    
    # Always show context and performance metrics first
    if analysis == "all":
        show_original_query_context(results)
        show_performance_metrics(results)
        generate_summary_report(results)
        analyze_evolution_trends(results)
        analyze_query_diversity(results)
        analyze_genetic_operations(results)
        analyze_semantic_performance(results)
        show_best_queries(results)
    elif analysis == "evolution":
        show_original_query_context(results)
        analyze_evolution_trends(results)
    elif analysis == "diversity":
        analyze_query_diversity(results)
    elif analysis == "genetics":
        analyze_genetic_operations(results)
    elif analysis == "semantic":
        analyze_semantic_performance(results)
    elif analysis == "best":
        show_original_query_context(results)
        show_best_queries(results)
    elif analysis == "summary":
        show_original_query_context(results)
        show_performance_metrics(results)
        generate_summary_report(results)
    
    print("✨ Analysis complete!")


if __name__ == "__main__":
    main() 