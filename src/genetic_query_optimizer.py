#!/usr/bin/env python3
"""
Genetic Algorithm for Etsy Query Optimization

This module implements a genetic algorithm that evolves shopping queries to maximize:
1. Semantic relevance (primary): Semantic relevance score for the first page of products
2. Purchase decisions (secondary): Total amount of money spent by agents

The algorithm uses the existing analyze_query.py infrastructure to evaluate fitness.
"""

import asyncio
import json
import random
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import logging
import subprocess
import os
import time
import re
from dataclasses import dataclass, asdict
import click

# Add current directory to path for imports
CURRENT_DIR = Path(__file__).resolve().parent
project_root = str(CURRENT_DIR.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up OpenAI credentials
try:
    key_path = CURRENT_DIR / "keys" / "litellm.key"
    os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()
except FileNotFoundError:
    raise Exception(
        "litellm.key file not found. It is expected in optimization-agent/src/keys/litellm.key"
    )
os.environ["OPENAI_API_BASE"] = "https://litellm.litellm.kn.ml-platform.etsy-mlinfra-dev.etsycloud.com"

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from genetic_prompts import GENERATE_INITIAL_POPULATION_PROMPT

@dataclass
class QueryIndividual:
    """Represents a single query individual in the genetic algorithm population."""
    query: str
    fitness_score: float = 0.0
    semantic_relevance: Dict = None
    purchase_stats: Dict = None
    generation: int = 0
    parent_queries: List[str] = None
    
    def __post_init__(self):
        if self.semantic_relevance is None:
            self.semantic_relevance = {}
        if self.purchase_stats is None:
            self.purchase_stats = {}
        if self.parent_queries is None:
            self.parent_queries = []


@dataclass
class GeneticAlgorithmConfig:
    """Configuration for the genetic algorithm."""
    population_size: int = 6
    n_generations: int = 4
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elitism_count: int = 2
    n_agents: int = 2  # Number of agents per query evaluation
    concurrency: int = 1  # Number of concurrent evaluations
    max_steps: int = 15  # Max steps per agent
    semantic_weight: float = 0.8  # Weight for semantic relevance in fitness
    purchase_weight: float = 0.2  # Weight for purchase decisions in fitness


class GeneticQueryOptimizer:
    """Genetic Algorithm for optimizing Etsy shopping queries."""
    
    def __init__(self, config: GeneticAlgorithmConfig, base_debug_path: Path):
        self.config = config
        self.base_debug_path = base_debug_path
        self.generation_history: List[List[QueryIndividual]] = []
        self.best_individual: Optional[QueryIndividual] = None
        
        # Set up logging
        self.setup_logging()
        
        # Initialize LLM for query generation
        self.llm = ChatOpenAI(
            model_name="openai/o4-mini",
            temperature=0.7
        )
    
    def setup_logging(self):
        """Set up logging for the genetic algorithm."""
        log_file = self.base_debug_path / "genetic_algorithm.log"
        self.base_debug_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("GeneticQueryOptimizer")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
    
    async def generate_initial_population(self, original_query: str) -> List[QueryIndividual]:
        """Generate initial population of query variations."""
        self.logger.info(f"Generating initial population from: '{original_query}'")
        
        population = [QueryIndividual(query=original_query, generation=0)]
        
        # Generate variations using LLM
        system_prompt = GENERATE_INITIAL_POPULATION_PROMPT
        
        user_prompt = f"""Original query: "{original_query}"

        Generate {self.config.population_size - 1} diverse variations of this query. Return them as a JSON list of strings.
        
        Example format:
        ["variation 1", "variation 2", "variation 3"]
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            variations_text = response.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', variations_text, re.DOTALL)
            if json_match:
                variations = json.loads(json_match.group())
                for variation in variations[:self.config.population_size - 1]:
                    if isinstance(variation, str) and variation.strip():
                        population.append(QueryIndividual(
                            query=variation.strip(),
                            generation=0,
                            parent_queries=[original_query]
                        ))
            
        except Exception as e:
            self.logger.warning(f"Failed to generate LLM variations: {e}")
            # Fallback: generate simple variations
            population.extend(self._generate_simple_variations(original_query))
        
        # Ensure we have the right population size
        while len(population) < self.config.population_size:
            population.append(QueryIndividual(
                query=self._mutate_query(original_query),
                generation=0,
                parent_queries=[original_query]
            ))
        
        population = population[:self.config.population_size]
        
        self.logger.info(f"Generated population of {len(population)} queries:")
        for i, individual in enumerate(population):
            self.logger.info(f"  {i+1}. '{individual.query}'")
        
        return population
    
    def _generate_simple_variations(self, original_query: str) -> List[QueryIndividual]:
        """Generate simple rule-based variations as fallback."""
        variations = []
        words = original_query.split()
        
        # Add adjectives
        adjectives = ["vintage", "handmade", "unique", "custom", "artisan", "modern", "classic"]
        for adj in adjectives[:2]:
            variations.append(QueryIndividual(
                query=f"{adj} {original_query}",
                generation=0,
                parent_queries=[original_query]
            ))
        
        # Word reordering
        if len(words) > 1:
            variations.append(QueryIndividual(
                query=" ".join(words[::-1]),
                generation=0,
                parent_queries=[original_query]
            ))
        
        return variations[:3]
    
    async def evaluate_fitness(self, individual: QueryIndividual, generation: int) -> QueryIndividual:
        """Evaluate the fitness of a query individual."""
        self.logger.info(f"Evaluating query: '{individual.query}'")
        
        # Create unique debug directory for this evaluation
        eval_debug_path = self.base_debug_path / f"gen_{generation}" / f"query_{abs(hash(individual.query)) % 10000}"
        eval_debug_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run analyze_query.py for this query
            cmd = [
                sys.executable, str(CURRENT_DIR / "analyze_query.py"),
                "--task", individual.query,
                "--n-agents", str(self.config.n_agents),
                "--concurrency", str(self.config.concurrency),
                "--max-steps", str(self.config.max_steps),
                "--debug-root", str(eval_debug_path),
                "--headless",
                "--model-name", "openai/o4-mini",
                "--seed", "42"  # For reproducibility
            ]
            
            # Run the evaluation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                self.logger.error(f"Query evaluation failed: {result.stderr}")
                individual.fitness_score = 0.0
                return individual
            
            # Parse results
            fitness_score = self._parse_evaluation_results(eval_debug_path, individual)
            individual.fitness_score = fitness_score
            
            self.logger.info(f"Query '{individual.query}' fitness: {fitness_score:.3f}")
            return individual
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Query evaluation timed out: '{individual.query}'")
            individual.fitness_score = 0.0
            return individual
        except Exception as e:
            self.logger.error(f"Error evaluating query '{individual.query}': {e}")
            individual.fitness_score = 0.0
            return individual
    
    def _parse_evaluation_results(self, debug_path: Path, individual: QueryIndividual) -> float:
        """Parse the results from analyze_query.py and calculate fitness score."""
        try:
            # Parse semantic scores
            semantic_file = debug_path / "_semantic_scores.json"
            purchase_file = debug_path / "_final_purchase_decision.json"
            
            semantic_score = 0.0
            purchase_score = 0.0
            
            if semantic_file.exists():
                with open(semantic_file, 'r') as f:
                    semantic_data = json.load(f)
                
                # Calculate semantic relevance score
                page1_data = semantic_data.get('page1_products', {})
                highly_relevant = page1_data.get('highly_relevant', 0)
                somewhat_relevant = page1_data.get('somewhat_relevant', 0)
                total_products = page1_data.get('total', 1)
                
                # Weighted semantic score
                semantic_score = (highly_relevant * 2 + somewhat_relevant * 1) / max(total_products, 1)
                
                individual.semantic_relevance = semantic_data
            
            if purchase_file.exists():
                with open(purchase_file, 'r') as f:
                    purchase_data = json.load(f)
                
                # Calculate purchase score
                purchase_stats = purchase_data.get('purchase_statistics', {})
                total_purchases = purchase_stats.get('total_purchases', 0)
                agents_who_purchased = purchase_stats.get('agents_who_purchased', 0)
                total_agents = purchase_data.get('total_agents', 1)
                
                # Purchase rate score
                purchase_score = agents_who_purchased / max(total_agents, 1)
                
                individual.purchase_stats = purchase_data
            
            # Combined fitness score
            fitness = (
                self.config.semantic_weight * semantic_score +
                self.config.purchase_weight * purchase_score
            )
            
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error parsing results: {e}")
            return 0.0
    
    def select_parents(self, population: List[QueryIndividual]) -> List[QueryIndividual]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Tournament selection
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.fitness_score)
            parents.append(winner)
        
        return parents
    
    async def crossover(self, parent1: QueryIndividual, parent2: QueryIndividual, generation: int) -> QueryIndividual:
        """Create offspring through crossover of two parent queries."""
        if random.random() > self.config.crossover_rate:
            return random.choice([parent1, parent2])
        
        # LLM-based crossover
        system_prompt = """You are an expert at combining shopping queries to create new, potentially better variations.
        Given two parent queries, create a new query that combines the best aspects of both while maintaining search relevance.
        
        Guidelines:
        - The result should be a natural, searchable query
        - Combine meaningful elements from both parents
        - Keep it concise (2-8 words typically)
        - Maintain the original search intent
        """
        
        user_prompt = f"""Parent 1: "{parent1.query}"
        Parent 2: "{parent2.query}"
        
        Create a new query that combines elements from both parents. Return only the new query text, nothing else."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            offspring_query = response.content.strip().strip('"').strip("'")
            
            # Validate the offspring
            if len(offspring_query.split()) > 10:
                # Fallback to simple word combination
                words1 = parent1.query.split()
                words2 = parent2.query.split()
                offspring_query = " ".join(random.sample(words1 + words2, min(4, len(words1 + words2))))
            
        except Exception as e:
            self.logger.warning(f"LLM crossover failed: {e}")
            # Fallback crossover
            words1 = parent1.query.split()
            words2 = parent2.query.split()
            offspring_query = " ".join(random.sample(words1 + words2, min(4, len(words1 + words2))))
        
        return QueryIndividual(
            query=offspring_query,
            generation=generation,
            parent_queries=[parent1.query, parent2.query]
        )
    
    def _mutate_query(self, query: str) -> str:
        """Apply simple mutations to a query."""
        words = query.split()
        
        if not words:
            return query
        
        mutation_type = random.choice(["synonym", "add_word", "remove_word", "reorder"])
        
        if mutation_type == "add_word" and len(words) < 6:
            # Add common modifiers
            modifiers = ["vintage", "handmade", "custom", "unique", "modern", "classic", "artisan"]
            words.insert(random.randint(0, len(words)), random.choice(modifiers))
        
        elif mutation_type == "remove_word" and len(words) > 2:
            words.pop(random.randint(0, len(words) - 1))
        
        elif mutation_type == "reorder" and len(words) > 1:
            random.shuffle(words)
        
        return " ".join(words)
    
    async def mutate(self, individual: QueryIndividual) -> QueryIndividual:
        """Apply mutation to an individual."""
        if random.random() > self.config.mutation_rate:
            return individual
        
        mutated_query = self._mutate_query(individual.query)
        
        return QueryIndividual(
            query=mutated_query,
            generation=individual.generation,
            parent_queries=[individual.query]
        )
    
    async def evolve_generation(self, population: List[QueryIndividual], generation: int) -> List[QueryIndividual]:
        """Evolve one generation of the population."""
        self.logger.info(f"\n=== Generation {generation} ===")
        
        # Evaluate fitness for all individuals
        evaluated_population = []
        for individual in population:
            if individual.fitness_score == 0.0:  # Only evaluate if not already evaluated
                individual = await self.evaluate_fitness(individual, generation)
            evaluated_population.append(individual)
        
        # Sort by fitness
        evaluated_population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Log generation results
        self.logger.info(f"Generation {generation} results:")
        for i, individual in enumerate(evaluated_population):
            self.logger.info(f"  {i+1}. '{individual.query}' - Fitness: {individual.fitness_score:.3f}")
        
        # Update best individual
        if self.best_individual is None or evaluated_population[0].fitness_score > self.best_individual.fitness_score:
            self.best_individual = evaluated_population[0]
            self.logger.info(f"New best individual: '{self.best_individual.query}' - Fitness: {self.best_individual.fitness_score:.3f}")
        
        # Save generation history
        self.generation_history.append(evaluated_population.copy())
        
        if generation >= self.config.n_generations - 1:
            return evaluated_population
        
        # Create next generation
        next_generation = []
        
        # Elitism: keep best individuals
        next_generation.extend(evaluated_population[:self.config.elitism_count])
        
        # Generate new individuals through crossover and mutation
        parents = self.select_parents(evaluated_population)
        
        while len(next_generation) < self.config.population_size:
            parent1, parent2 = random.sample(parents, 2)
            offspring = await self.crossover(parent1, parent2, generation + 1)
            offspring = await self.mutate(offspring)
            next_generation.append(offspring)
        
        return next_generation[:self.config.population_size]
    
    async def run_optimization(self, original_query: str) -> QueryIndividual:
        """Run the complete genetic algorithm optimization."""
        self.logger.info(f"Starting genetic algorithm optimization for query: '{original_query}'")
        start_time = time.time()
        
        # Generate initial population
        population = await self.generate_initial_population(original_query)
        
        # Evolve through generations
        for generation in range(self.config.n_generations):
            population = await self.evolve_generation(population, generation)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        self.logger.info(f"\nOptimization completed in {elapsed_time:.2f} seconds")
        self.logger.info(f"Best query found: '{self.best_individual.query}'")
        self.logger.info(f"Best fitness score: {self.best_individual.fitness_score:.3f}")
        
        # Save final results
        self._save_optimization_results()
        
        return self.best_individual
    
    def _save_optimization_results(self):
        """Save optimization results to files."""
        results_file = self.base_debug_path / "optimization_results.json"
        
        results = {
            "optimization_config": asdict(self.config),
            "best_individual": asdict(self.best_individual),
            "generation_history": [
                [asdict(individual) for individual in generation]
                for generation in self.generation_history
            ],
            "summary": {
                "total_generations": len(self.generation_history),
                "best_fitness": self.best_individual.fitness_score if self.best_individual else 0.0,
                "best_query": self.best_individual.query if self.best_individual else "",
                "improvement": self._calculate_improvement()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_file}")
    
    def _calculate_improvement(self) -> Dict:
        """Calculate improvement metrics."""
        if not self.generation_history:
            return {}
        
        initial_best = max(self.generation_history[0], key=lambda x: x.fitness_score)
        final_best = self.best_individual
        
        return {
            "initial_best_fitness": initial_best.fitness_score,
            "final_best_fitness": final_best.fitness_score if final_best else 0.0,
            "absolute_improvement": (final_best.fitness_score - initial_best.fitness_score) if final_best else 0.0,
            "relative_improvement": ((final_best.fitness_score - initial_best.fitness_score) / max(initial_best.fitness_score, 0.001)) * 100 if final_best else 0.0
        }


# CLI Interface
@click.command()
@click.option(
    "--query",
    type=str,
    required=True,
    help="The original query to optimize."
)
@click.option(
    "--population-size",
    type=int,
    default=8,
    show_default=True,
    help="Size of the population in each generation."
)
@click.option(
    "--generations",
    type=int,
    default=5,
    show_default=True,
    help="Number of generations to evolve."
)
@click.option(
    "--mutation-rate",
    type=float,
    default=0.3,
    show_default=True,
    help="Probability of mutation (0.0-1.0)."
)
@click.option(
    "--crossover-rate",
    type=float,
    default=0.7,
    show_default=True,
    help="Probability of crossover (0.0-1.0)."
)
@click.option(
    "--n-agents",
    type=int,
    default=2,
    show_default=True,
    help="Number of agents to use for each query evaluation."
)
@click.option(
    "--max-steps",
    type=int,
    default=15,
    show_default=True,
    help="Maximum steps for each agent."
)
@click.option(
    "--debug-root",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=CURRENT_DIR / "genetic_optimization",
    show_default=True,
    help="Root directory for optimization debug files."
)
@click.option(
    "--semantic-weight",
    type=float,
    default=0.8,
    show_default=True,
    help="Weight for semantic relevance in fitness calculation (0.0-1.0)."
)
@click.option(
    "--purchase-weight", 
    type=float,
    default=0.2,
    show_default=True,
    help="Weight for purchase decisions in fitness calculation (0.0-1.0)."
)
def main(
    query: str,
    population_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    n_agents: int,
    max_steps: int,
    debug_root: Path,
    semantic_weight: float,
    purchase_weight: float
):
    """
    Genetic Algorithm for optimizing Etsy shopping queries.
    
    This tool evolves a given query over multiple generations to find variations
    that perform better in terms of semantic relevance and purchase decisions.
    """
    
    # Validate weights
    if abs(semantic_weight + purchase_weight - 1.0) > 0.001:
        raise click.BadParameter("semantic-weight and purchase-weight must sum to 1.0")
    
    # Clean debug directory
    if debug_root.exists():
        if click.confirm(f'Debug path {debug_root} already exists. Remove it and start fresh?'):
            shutil.rmtree(debug_root)
        else:
            print('Continuing with existing debug directory...')
    
    debug_root.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = GeneticAlgorithmConfig(
        population_size=population_size,
        n_generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        n_agents=n_agents,
        max_steps=max_steps,
        semantic_weight=semantic_weight,
        purchase_weight=purchase_weight
    )
    
    # Run optimization
    optimizer = GeneticQueryOptimizer(config, debug_root)
    
    async def run_async():
        return await optimizer.run_optimization(query)
    
    best_query = asyncio.run(run_async())
    
    print(f"\n{'='*60}")
    print(f"🧬 GENETIC ALGORITHM OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Original query: '{query}'")
    print(f"Best query found: '{best_query.query}'")
    print(f"Fitness improvement: {optimizer._calculate_improvement().get('relative_improvement', 0):.1f}%")
    print(f"Final fitness score: {best_query.fitness_score:.3f}")
    print(f"Results saved to: {debug_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 