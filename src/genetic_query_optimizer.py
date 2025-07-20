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
import math
import random
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import logging
import os
import time
import re
from dataclasses import dataclass, asdict
import click
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Add current directory to path for imports
CURRENT_DIR = Path(__file__).resolve().parent
project_root = str(CURRENT_DIR.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from genetic_prompts import GENERATE_INITIAL_POPULATION_PROMPT
from analyze_query import run_analyze_query
from shopping_agent.config import MODEL_PRICING, VENDOR_DISCOUNT_GEMINI

# Set up OpenAI credentials
try:
    key_path = CURRENT_DIR / "keys" / "litellm.key"
    os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()
except FileNotFoundError:
    raise Exception(
        "litellm.key file not found. It is expected in optimization-agent/src/keys/litellm.key"
    )
os.environ["OPENAI_API_BASE"] = "https://litellm.litellm.kn.ml-platform.etsy-mlinfra-dev.etsycloud.com"

@dataclass
class QueryIndividual:
    """Represents a single query individual in the genetic algorithm population."""
    query: str
    fitness_score: float = -1.0  # -1.0 indicates not evaluated yet
    semantic_relevance: Dict = None
    purchase_stats: Dict = None
    generation: int = 0
    parent_queries: List[str] = None
    cost_info: Dict = None  # Cost information for this evaluation
    
    def __post_init__(self):
        if self.semantic_relevance is None:
            self.semantic_relevance = {}
        if self.purchase_stats is None:
            self.purchase_stats = {}
        if self.parent_queries is None:
            self.parent_queries = []
        if self.cost_info is None:
            self.cost_info = {}


@dataclass
class GeneticAlgorithmConfig:
    """Configuration for the genetic algorithm."""
    population_size: int = 5
    n_generations: int = 4
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    n_agents: int = 5  # Number of agents per query evaluation
    concurrency: int = 5  # Number of concurrent evaluations
    max_steps: Optional[int] = None  # Max steps per agent (None for unlimited)
    page1_semantic_weight: float = 0.4  # Weight for page 1 semantic relevance
    top10_semantic_weight: float = 0.5  # Weight for top 10 semantic relevance
    purchase_weight: float = 0.1  # Weight for purchase decision rate
    headless: bool = True  # Whether to run browsers in headless mode
    model_name: str = "global-gemini-2.5-flash"  # Model name for genetic algorithm LLM operations


class GeneticQueryOptimizer:
    """Genetic Algorithm for optimizing Etsy shopping queries."""
    
    def __init__(self, config: GeneticAlgorithmConfig, base_debug_path: Path):
        self.config = config
        self.base_debug_path = base_debug_path

        self.generation_history: List[List[QueryIndividual]] = []
        self.best_individual: Optional[QueryIndividual] = None
        self.original_query: Optional[str] = None  # Store the original query being optimized
        
        # Token tracking
        self.token_usage: Dict[str, Dict[str, any]] = {}  # Track token usage for genetic algorithm LLM calls
        self.total_optimization_cost: float = 0.0
        self.fitness_evaluation_costs: List[Dict] = []  # Track costs from fitness evaluations
        
        # Set up logging
        self.setup_logging()
        
        # Initialize LLM for query generation
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
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
    
    def _update_token_usage(self, model_name: str, usage_metadata: Dict, operation_type: str):
        """Update token usage tracking for genetic algorithm LLM calls."""
        if not usage_metadata:
            return
            
        input_tokens = usage_metadata.get("input_tokens", 0)
        output_tokens = usage_metadata.get("output_tokens", 0)
        
        # Initialize model tracking if not exists
        if model_name not in self.token_usage:
            self.token_usage[model_name] = {
                "operations": {},
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0
            }
        
        # Initialize operation tracking if not exists
        if operation_type not in self.token_usage[model_name]["operations"]:
            self.token_usage[model_name]["operations"][operation_type] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0
            }
        
        # Update operation-specific tracking
        op_data = self.token_usage[model_name]["operations"][operation_type]
        op_data["input_tokens"] += input_tokens
        op_data["output_tokens"] += output_tokens
        op_data["total_tokens"] += input_tokens + output_tokens
        op_data["calls"] += 1
        
        # Calculate costs
        if model_name in MODEL_PRICING:
            price_per_million = MODEL_PRICING[model_name]
            input_cost = (input_tokens / 1_000_000) * price_per_million["input"]
            output_cost = (output_tokens / 1_000_000) * price_per_million["output"]
            
            op_data["input_cost"] += input_cost
            op_data["output_cost"] += output_cost
            op_data["total_cost"] += input_cost + output_cost
        
        # Update model-level totals
        self.token_usage[model_name]["total_input_tokens"] += input_tokens
        self.token_usage[model_name]["total_output_tokens"] += output_tokens
        self.token_usage[model_name]["total_cost"] += op_data["total_cost"]
        
        # Update optimization total
        self.total_optimization_cost += op_data["total_cost"]
        
        self.logger.info(f"Token usage - {operation_type}: {input_tokens} input, {output_tokens} output tokens (${op_data['total_cost']:.4f})")
    
    def _aggregate_fitness_evaluation_costs(self, eval_debug_path: Path) -> Dict:
        """Aggregate token usage from fitness evaluation (analyze_query results)."""
        cost_data = {
            "total_cost": 0.0,
            "total_cost_after_discount": 0.0,
            "agent_costs": []
        }
        
        # Check for aggregated token usage file first
        aggregated_token_file = eval_debug_path / "_token_usage.json"
        if aggregated_token_file.exists():
            with open(aggregated_token_file, 'r') as f:
                data = json.load(f)
                cost_data["total_cost"] = data.get("total_session_cost", 0.0)
                cost_data["total_cost_after_discount"] = data.get("total_session_cost_after_discount", 0.0)
                cost_data["models"] = data.get("models", {})
                return cost_data
        
        # Fallback: aggregate from individual agent files
        agent_dirs = [d for d in eval_debug_path.iterdir() if d.is_dir() and d.name.startswith('agent_')]
        for agent_dir in agent_dirs:
            token_file = agent_dir / "_token_usage.json"
            if token_file.exists():
                with open(token_file, 'r') as f:
                    agent_data = json.load(f)
                    cost_data["agent_costs"].append(agent_data)
                    cost_data["total_cost"] += agent_data.get("total_session_cost", 0.0)
                    cost_data["total_cost_after_discount"] += agent_data.get("total_session_cost_after_discount", 0.0)
        
        return cost_data
    
    async def generate_initial_population(self, original_query: str) -> List[QueryIndividual]:
        """Generate initial population of query variations."""
        self.logger.info(f"Generating initial population from: '{original_query}'")
        
        population = [QueryIndividual(query=original_query, generation=0)]
        
        # Generate variations using LLM
        system_prompt = GENERATE_INITIAL_POPULATION_PROMPT
        user_prompt = f"""
Original query: "{original_query}"
Generate {self.config.population_size - 1} diverse variations of this query. Return them as a JSON list of strings.

Example format:
["variation 1", "variation 2", "variation 3", ...]
        """.strip()
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Track token usage
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                self._update_token_usage(
                    self.llm.model_name,
                    response.usage_metadata,
                    "initial_population_generation"
                )
            
            variations_text = response.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', variations_text, re.DOTALL)
            if json_match:
                variations = json.loads(json_match.group())
                for variation in variations:
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
        for adj in adjectives:
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
        
        return variations[:self.config.population_size]
    
    def _sanitize_query_for_path(self, query: str) -> str:
        """Convert query to filesystem-safe directory name."""
        # Replace spaces with underscores and remove/replace problematic characters
        sanitized = query.lower()
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', sanitized)  # Replace filesystem-unsafe chars
        sanitized = re.sub(r'[^\w\s-]', '', sanitized)  # Remove non-alphanumeric except spaces and hyphens
        sanitized = re.sub(r'[-\s]+', '_', sanitized)  # Replace spaces and multiple hyphens with single underscore
        sanitized = sanitized.strip('_')  # Remove leading/trailing underscores
        
        # Limit length to avoid filesystem issues
        if len(sanitized) > 50:
            sanitized = sanitized[:50].rstrip('_')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = f"query_{abs(hash(query)) % 10000}"
        
        return sanitized
    
    async def evaluate_fitness(self, individual: QueryIndividual, generation: int) -> QueryIndividual:
        """Evaluate the fitness of a query individual."""
        self.logger.info(f"Evaluating query: '{individual.query}'")
        
        # Create unique debug directory for this evaluation
        query_dir_name = self._sanitize_query_for_path(individual.query)
        eval_debug_path = self.base_debug_path / f"gen_{generation}" / query_dir_name
        if os.path.exists(eval_debug_path):
            eval_debug_path = self.base_debug_path / f"gen_{generation}" / query_dir_name + "_" + str(abs(hash(individual.query)) % 10000)
        eval_debug_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run analyze_query directly using imported function
            await run_analyze_query(
                task=self.original_query,  # Use the original query being optimized as the task
                curr_query=individual.query,  # Use the evolved query variant as the current query
                n_agents=self.config.n_agents,
                model_name=self.config.model_name,
                summary_model=None,
                max_steps=self.config.max_steps,
                headless=self.config.headless,
                concurrency=self.config.concurrency,
                debug_root=eval_debug_path,
                width=1920,
                height=1080,
                final_decision_model=None,
                temperature=0.7,
                record_video=False,
                save_local=True,
                save_gcs=True,
                gcs_bucket="training-dev-search-data-jtzn",
                gcs_prefix="smu-agent-optimizer",
                skip_confirmation=True,  # Skip confirmation for automated runs
            )
            
            # Aggregate fitness evaluation costs
            cost_data = self._aggregate_fitness_evaluation_costs(eval_debug_path)
            individual.cost_info = cost_data
            self.fitness_evaluation_costs.append({
                "query": individual.query,
                "generation": generation,
                "cost_data": cost_data
            })
            
            # Parse results
            fitness_score = self._parse_evaluation_results(eval_debug_path, individual)
            individual.fitness_score = fitness_score
            
            self.logger.info(f"Query '{individual.query}' fitness: {fitness_score:.3f}, cost: ${cost_data.get('total_cost_after_discount', 0.0):.4f}")
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
            
            page1_semantic_score = 0.0
            top10_semantic_score = 0.0
            purchase_score = 0.0
            raw_purchase_value = 0.0
            
            if semantic_file.exists():
                with open(semantic_file, 'r') as f:
                    semantic_data = json.load(f)
                
                # Calculate page1 semantic relevance score
                page1_data = semantic_data.get('page1_products', {})
                page1_highly_relevant = page1_data.get('highly_relevant', 0)
                page1_somewhat_relevant = page1_data.get('somewhat_relevant', 0)
                page1_total_products = page1_data.get('total', 1)
                
                # Weighted page1 semantic score
                page1_semantic_score = (page1_highly_relevant * 2 + page1_somewhat_relevant * 1) / (page1_total_products * 2)
                
                # Calculate top10 semantic relevance score
                top10_data = semantic_data.get('top_10_products', {})
                top10_highly_relevant = top10_data.get('highly_relevant', 0)
                top10_somewhat_relevant = top10_data.get('somewhat_relevant', 0)
                top10_total_products = top10_data.get('total', 1)
                
                # Weighted top10 semantic score
                top10_semantic_score = (top10_highly_relevant * 2 + top10_somewhat_relevant * 1) / (top10_total_products * 2)
                
                individual.semantic_relevance = semantic_data
            
            if purchase_file.exists():
                with open(purchase_file, 'r') as f:
                    purchase_data = json.load(f)
                
                # Calculate purchase score
                purchase_stats = purchase_data.get('purchase_statistics', {})
                raw_purchase_value = purchase_stats.get('average_cost_per_session', 0)
                
                # Normalize purchase score to [0, 1) using exponential saturation
                # Scale factor chosen so that $50 per session gives ~0.63 normalized score
                scale_factor = 50.0
                purchase_score = 1 - math.exp(-raw_purchase_value / scale_factor)
                
                individual.purchase_stats = purchase_data
            
            # Combined Fitness Score
            fitness = (
                self.config.purchase_weight * purchase_score +
                self.config.top10_semantic_weight * top10_semantic_score +
                self.config.page1_semantic_weight * page1_semantic_score
            )
            
            # Log individual components for debugging
            self.logger.info(f"  Fitness components - Purchase: {purchase_score:.3f} (${raw_purchase_value:.2f} normalized, weight: {self.config.purchase_weight}), "
                           f"Top10 Semantic: {top10_semantic_score:.3f} (weight: {self.config.top10_semantic_weight}), "
                           f"Page1 Semantic: {page1_semantic_score:.3f} (weight: {self.config.page1_semantic_weight})")
            
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error parsing results: {e}")
            return 0.0
    
    def select_parents(self, population: List[QueryIndividual]) -> List[QueryIndividual]:
        """Select parents for reproduction. For small populations, use top performers directly."""
        # For small populations (<=5), use top 3 to ensure diversity
        if len(population) <= 5:
            return population[:3] # Population is already sorted by fitness (descending), so take top 3
        
        # For larger populations, use tournament selection
        parents = []
        tournament_size = 3
        for _ in range(len(population)):
            tournament = random.sample(population, min(tournament_size, len(population))) # Tournament selection
            winner = max(tournament, key=lambda x: x.fitness_score)
            parents.append(winner)
        
        return parents
    
    async def crossover(self, parent1: QueryIndividual, parent2: QueryIndividual, generation: int) -> QueryIndividual:
        """Create offspring through crossover of two parent queries."""
        
        # Not in cross-over probability
        if random.random() > self.config.crossover_rate:
            return random.choice([parent1, parent2])
        
        # LLM-based crossover
        system_prompt = """
You are an expert at combining shopping queries to create new, potentially better variations.
Given two parent queries, create a new query that combines the best aspects of both while maintaining search relevance.
        
Guidelines:
- The result should be a natural, searchable query
- Combine meaningful elements from both parents
- Keep it concise (2-8 words typically)
- Maintain the original search intent
        """.strip()
        
        user_prompt = f"""
Parent 1: "{parent1.query}"
Parent 2: "{parent2.query}"

Create a new query that combines elements from both parents. Return only the new query text, nothing else.
        """.strip()
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Track token usage
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                self._update_token_usage(
                    self.llm.model_name,
                    response.usage_metadata,
                    "crossover"
                )
            
            offspring_query = response.content.strip().strip('"').strip("'")
            
            # Validate the offspring
            if len(offspring_query.split()) > 10:
                # Fallback to simple word combination
                words1 = parent1.query.split()
                words2 = parent2.query.split()
                offspring_query = " ".join(random.sample(words1 + words2, min(5, len(words1 + words2))))
            
        except Exception as e:
            self.logger.warning(f"LLM crossover failed: {e}")
            # Fallback crossover
            words1 = parent1.query.split()
            words2 = parent2.query.split()
            offspring_query = " ".join(random.sample(words1 + words2, min(5, len(words1 + words2))))
        
        return QueryIndividual(
            query=offspring_query,
            generation=generation,
            parent_queries=[parent1.query, parent2.query]
        )
    
    async def _mutate_query(self, query: str) -> str:
        """Apply LLM-based mutations to a query."""
        if not query.strip():
            return query
        
        # LLM-based mutation prompt
        system_prompt = """
You are an expert at creating subtle variations of shopping queries while maintaining their core meaning and search intent.
        
Guidelines:
- Keep the same general length (don't make it significantly longer or shorter)
- Maintain the original search intent
- Make small but meaningful changes (synonyms, reordering, slight modifications)
- The result should still be a natural, searchable query
- Only output the revised query, nothing else
        """.strip()
        
        user_prompt = f"""
Please revise the following query with no changes to its length and only output the revised version, the query is: 
"{query}" 
        """.strip()
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Track token usage
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                self._update_token_usage(
                    self.llm.model_name,
                    response.usage_metadata,
                    "mutation"
                )
            
            mutated_query = response.content.strip().strip('"').strip("'")
            
            # Validate the mutation
            if not mutated_query or len(mutated_query.split()) > len(query.split()) + 2:
                # Fallback to simple rule-based mutation if LLM produces invalid result
                return self._fallback_mutate_query(query)
            
            return mutated_query
            
        except Exception as e:
            self.logger.warning(f"LLM mutation failed: {e}")
            # Fallback to simple rule-based mutation
            return self._fallback_mutate_query(query)
    
    def _fallback_mutate_query(self, query: str) -> str:
        """Fallback rule-based mutation when LLM fails."""
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

        # Not in mutation probability
        if random.random() > self.config.mutation_rate:
            return individual
        
        # Mutate the query
        mutated_query = await self._mutate_query(individual.query)
        
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
            if individual.fitness_score < 0.0:  # Only evaluate if not already evaluated
                individual = await self.evaluate_fitness(individual, generation)
            evaluated_population.append(individual)
        
        # Sort by fitness
        evaluated_population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Log generation results
        self.logger.info(f"Generation {generation} results:")
        for i, individual in enumerate(evaluated_population):
            cost = individual.cost_info.get('total_cost_after_discount', 0.0) if individual.cost_info else 0.0
            self.logger.info(f"  {i+1}. '{individual.query}' - Fitness: {individual.fitness_score:.3f}, Cost: ${cost:.4f}")
        
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
        
        # All individuals go through crossover and mutation for maximum diversity
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
        
        # Store the original query for use in fitness evaluation
        self.original_query = original_query
        
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
        
        # Calculate total costs
        total_genetic_cost = sum(model_data["total_cost"] for model_data in self.token_usage.values())
        total_fitness_cost = sum(eval_data["cost_data"].get("total_cost_after_discount", 0.0) for eval_data in self.fitness_evaluation_costs)
        
        # Apply vendor discount to genetic algorithm costs if using Gemini
        total_genetic_cost_after_discount = total_genetic_cost
        if any('gemini' in model_name.lower() for model_name in self.token_usage.keys()):
            total_genetic_cost_after_discount = total_genetic_cost * (1 - VENDOR_DISCOUNT_GEMINI)
        
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
            },
            "cost_analysis": {
                "genetic_algorithm_costs": {
                    "total_cost": total_genetic_cost,
                    "total_cost_after_discount": total_genetic_cost_after_discount,
                    "vendor_discount_applied": VENDOR_DISCOUNT_GEMINI if any('gemini' in model_name.lower() for model_name in self.token_usage.keys()) else 0.0,
                    "by_operation": self.token_usage
                },
                "fitness_evaluation_costs": {
                    "total_cost": total_fitness_cost,
                    "by_evaluation": self.fitness_evaluation_costs
                },
                "total_optimization_cost": total_genetic_cost_after_discount + total_fitness_cost,
                "cost_breakdown": {
                    "genetic_algorithm_percentage": (total_genetic_cost_after_discount / (total_genetic_cost_after_discount + total_fitness_cost)) * 100 if (total_genetic_cost_after_discount + total_fitness_cost) > 0 else 0,
                    "fitness_evaluation_percentage": (total_fitness_cost / (total_genetic_cost_after_discount + total_fitness_cost)) * 100 if (total_genetic_cost_after_discount + total_fitness_cost) > 0 else 0
                }
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save a separate token usage file for easy reference
        token_usage_file = self.base_debug_path / "token_usage_summary.json"
        with open(token_usage_file, 'w') as f:
            json.dump(results["cost_analysis"], f, indent=2)
        
        self.logger.info(f"Results saved to: {results_file}")
        self.logger.info(f"Token usage summary saved to: {token_usage_file}")
        
        # Log cost summary
        self.logger.info(f"\n💰 COST SUMMARY:")
        self.logger.info(f"  Genetic Algorithm: ${total_genetic_cost_after_discount:.4f}")
        self.logger.info(f"  Fitness Evaluations: ${total_fitness_cost:.4f}")
        self.logger.info(f"  Total Optimization Cost: ${total_genetic_cost_after_discount + total_fitness_cost:.4f}")
    
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
@click.option("--query", type=str, required=True, help="The original query to optimize.")
@click.option("--population-size", type=int, default=5, show_default=True, help="Size of the population in each generation.")
@click.option("--generations", type=int, default=4, show_default=True, help="Number of generations to evolve.")
@click.option("--mutation-rate", type=float, default=0.1, show_default=True, help="Probability of mutation (0.0-1.0).")
@click.option("--crossover-rate", type=float, default=0.7, show_default=True, help="Probability of crossover (0.0-1.0).")
@click.option("--n-agents", type=int, default=5, show_default=True, help="Number of agents to use for each query evaluation.")
@click.option("--max-steps", type=int, default=None, show_default=True, help="Maximum steps for each agent. Set to None for unlimited.")
@click.option("--debug-root", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=CURRENT_DIR / "debug_ga", show_default=True, help="Root directory for optimization debug files.")
@click.option("--page1-semantic-weight", type=float, default=0.4, show_default=True, help="Weight for page 1 semantic relevance in fitness calculation (0.0-1.0).")
@click.option("--top10-semantic-weight", type=float, default=0.5, show_default=True, help="Weight for top 10 semantic relevance in fitness calculation (0.0-1.0).")
@click.option("--purchase-weight", type=float, default=0.1, show_default=True, help="Weight for purchase decision rate in fitness calculation (0.0-1.0).")
@click.option("--headless/--no-headless", default=True, show_default=True, help="Run browsers in headless mode during fitness evaluations. Use --no-headless to show browser UI for debugging.")
@click.option("--model-name", type=str, default="global-gemini-2.5-flash", show_default=True, help="Model name to use for genetic algorithm LLM operations (population generation, crossover, mutation).")
def main(
    query: str,
    population_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    n_agents: int,
    max_steps: Optional[int],
    debug_root: Path,
    page1_semantic_weight: float,
    top10_semantic_weight: float,
    purchase_weight: float,
    headless: bool,
    model_name: str
):
    """
    Genetic Algorithm for optimizing Etsy shopping queries.
    
    This tool evolves a given query over multiple generations to find variations
    that perform better in terms of semantic relevance and purchase decisions.
    
    Fitness calculation uses configurable weights for:
    - Page 1 semantic relevance (default: 0.4)
    - Top 10 semantic relevance (default: 0.5)
    - Purchase decision rate (default: 0.1)
    
    The three weights must sum to 1.0.
    """
    
    # Validate that weights sum to 1.0
    total_weight = page1_semantic_weight + top10_semantic_weight + purchase_weight
    if abs(total_weight - 1.0) > 0.001:
        raise click.BadParameter(f"All weights must sum to 1.0, got {total_weight:.3f}")
    
    # Validate individual weights are positive
    if page1_semantic_weight < 0 or top10_semantic_weight < 0 or purchase_weight < 0:
        raise click.BadParameter("All weights must be non-negative")
    
    # Clean debug directory
    if debug_root.exists():
        if click.confirm(f'Debug path {debug_root} already exists. Remove it and start fresh?'):
            shutil.rmtree(debug_root)
        else:
            print('Please remove the debug directory or run with a different --debug-root.')
            return
    
    debug_root.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = GeneticAlgorithmConfig(
        population_size=population_size,
        n_generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        n_agents=n_agents,
        max_steps=max_steps,
        page1_semantic_weight=page1_semantic_weight,
        top10_semantic_weight=top10_semantic_weight,
        purchase_weight=purchase_weight,
        headless=headless,
        model_name=model_name
    )
    
    # Run optimization
    optimizer = GeneticQueryOptimizer(config, debug_root)
    
    async def run_async():
        return await optimizer.run_optimization(query)
    
    best_query = asyncio.run(run_async())
    
    # Get final cost information
    total_genetic_cost = sum(model_data["total_cost"] for model_data in optimizer.token_usage.values())
    total_fitness_cost = sum(eval_data["cost_data"].get("total_cost_after_discount", 0.0) for eval_data in optimizer.fitness_evaluation_costs)
    
    # Apply vendor discount to genetic algorithm costs if using Gemini
    total_genetic_cost_after_discount = total_genetic_cost
    if any('gemini' in model_name.lower() for model_name in optimizer.token_usage.keys()):
        total_genetic_cost_after_discount = total_genetic_cost * (1 - VENDOR_DISCOUNT_GEMINI)
    
    total_cost = total_genetic_cost_after_discount + total_fitness_cost
    
    print(f"\n{'='*60}")
    print(f"🧬 GENETIC ALGORITHM OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Original query: '{query}'")
    print(f"Best query found: '{best_query.query}'")
    print(f"Fitness improvement: {optimizer._calculate_improvement().get('relative_improvement', 0):.1f}%")
    print(f"Final fitness score: {best_query.fitness_score:.3f}")
    print(f"💰 Total cost: ${total_cost:.4f}")
    print(f"  - Genetic algorithm: ${total_genetic_cost_after_discount:.4f}")
    print(f"  - Fitness evaluations: ${total_fitness_cost:.4f}")
    print(f"Results saved to: {debug_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 