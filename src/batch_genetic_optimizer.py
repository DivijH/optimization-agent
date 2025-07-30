#!/usr/bin/env python3
"""
Simple batch wrapper for genetic_query_optimizer.py
Reads queries from final_queries.csv and optimizes them sequentially.
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import traceback
import logging

# Add current directory to path for imports
CURRENT_DIR = Path(__file__).resolve().parent
project_root = str(CURRENT_DIR.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from genetic_query_optimizer import GeneticQueryOptimizer, GeneticAlgorithmConfig
from shopping_agent.gcs_utils import upload_string_to_gcs, upload_file_to_gcs


START_INDEX = 0
END_INDEX = 999
POPULATION_SIZE = 5
GENERATIONS = 4
N_AGENTS = 5
CSV_PATH = CURRENT_DIR.parent / "data" / "final_queries.csv"
DEBUG_ROOT = CURRENT_DIR / "batch_run"
HEADLESS = True
MAX_STEPS = None


async def save_batch_results(results, args, logger):
    """Save batch optimization results to JSON and upload to GCS if enabled."""
    batch_summary = {
        "batch_config": {
            "start_index": args.start_index,
            "end_index": args.end_index,
            "population_size": args.population_size,
            "generations": args.generations,
            "n_agents": args.n_agents,
            "max_steps": args.max_steps,
            "headless": args.headless,
            "csv_path": str(args.csv_path),
            "debug_root": str(args.debug_root)
        },
        "results": results,
        "summary": {
            "total_queries": len(results),
            "successful": len([r for r in results if r.get('optimized')]),
            "failed": len([r for r in results if not r.get('optimized')]),
            "avg_improvement": sum(r['improvement'] for r in results if r.get('improvement')) / len([r for r in results if r.get('improvement')]) if any(r.get('improvement') for r in results) else 0.0
        }
    }
    
    # Save locally
    results_file = args.debug_root / "batch_optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    # Upload to GCS if enabled
    if args.save_gcs:
        try:
            await upload_string_to_gcs(
                json.dumps(batch_summary, indent=2),
                str(results_file),
                args.gcs_bucket_name,
                args.gcs_prefix,
                args.gcs_project
            )
            logger.info("✅ Batch results uploaded to GCS successfully")
            print("📊 Batch results uploaded to GCS")
        except Exception as e:
            logger.error(f"❌ Failed to upload batch results to GCS: {e}")
            print(f"❌ Failed to upload batch results to GCS: {e}")
        
        # Also upload the batch optimization log file
        log_file = args.debug_root / "batch_optimization.log"
        if log_file.exists():
            try:
                await upload_file_to_gcs(
                    str(log_file),
                    str(log_file),
                    args.gcs_bucket_name,
                    args.gcs_prefix,
                    args.gcs_project
                )
                logger.info("✅ Batch optimization log uploaded to GCS successfully")
                print("📋 Batch optimization log uploaded to GCS")
            except Exception as e:
                logger.error(f"❌ Failed to upload batch optimization log to GCS: {e}")
                print(f"❌ Failed to upload batch optimization log to GCS: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description='Batch optimize queries from final_queries.csv using genetic algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--start-index', type=int, default=START_INDEX, help='Starting index (0-based, inclusive) in the CSV file')
    parser.add_argument('--end-index', type=int, default=END_INDEX, help='Ending index (0-based, inclusive) in the CSV file')
    parser.add_argument('--population-size', type=int, default=POPULATION_SIZE, help='Size of the population in each generation')
    parser.add_argument('--generations', type=int, default=GENERATIONS, help='Number of generations to evolve')
    parser.add_argument('--n-agents', type=int, default=N_AGENTS, help='Number of agents to use for each query evaluation')
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS, help='Maximum steps for each agent')
    parser.add_argument('--csv-path', type=Path, default=CSV_PATH, help='Path to the CSV file containing queries')
    parser.add_argument('--debug-root', type=Path, default=DEBUG_ROOT, help='Root directory for debug files')
    parser.add_argument('--headless', action='store_true', default=HEADLESS, help='Run browsers in headless mode')
    parser.add_argument('--no-headless', action='store_false', dest='headless', help='Show browser UI (for debugging)')
    parser.add_argument('--no-save-gcs', action='store_false', dest='save_gcs', default=True, help='Disable GCS uploads (GCS is enabled by default)')
    parser.add_argument('--gcs-bucket-name', type=str, default='training-dev-search-data-jtzn', help='GCS bucket name for uploads')
    parser.add_argument('--gcs-prefix', type=str, default='smu-agent-optimizer', help='GCS prefix for uploads')
    parser.add_argument('--gcs-project', type=str, default='etsy-search-ml-dev', help='GCS project for uploads')
    
    args = parser.parse_args()
    
    start_index = args.start_index
    end_index = args.end_index
    population_size = args.population_size
    generations = args.generations
    
    # Load CSV
    df = pd.read_csv(args.csv_path)
    
    # Validate indices
    if start_index < 0 or end_index >= len(df) or start_index > end_index:
        print(f"Invalid index range: {start_index}-{end_index} for CSV with {len(df)} rows")
        return
    
    # Get queries to process
    queries = df.iloc[start_index:end_index + 1]['Query'].tolist()
    
    # Suppress all console logging except our minimal prints
    logging.root.setLevel(logging.CRITICAL + 1)  # Disable root logger
    
    # Specifically silence browser_use and other noisy loggers
    for logger_name in ['browser_use', 'browser_use.telemetry', 'browser_use.telemetry.service']:
        noisy_logger = logging.getLogger(logger_name)
        noisy_logger.setLevel(logging.CRITICAL + 1)
        noisy_logger.propagate = False
        noisy_logger.disabled = True
    
    # Create a file logger for batch processing  
    logger = logging.getLogger('batch_optimizer')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler for detailed logging
    log_file = args.debug_root / "batch_optimization.log"
    args.debug_root.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger (which would print to console)
    logger.propagate = False
    
    print(f"🧬 Batch Genetic Optimizer - Processing {len(queries)} queries (indices {start_index}-{end_index})")
    print(f"📋 Detailed logs: {log_file}")
    logger.info(f"Batch Configuration:")
    logger.info(f"  Population size: {population_size}, Generations: {generations}")
    logger.info(f"  N-agents: {args.n_agents}, Max steps: {args.max_steps}")
    logger.info(f"  Headless: {args.headless}, Save to GCS: {args.save_gcs}")
    logger.info(f"  CSV path: {args.csv_path}")
    logger.info(f"  Debug root: {args.debug_root}")
    
    # Create config
    config = GeneticAlgorithmConfig(
        population_size=population_size,
        n_generations=generations,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        headless=args.headless,
        save_gcs=args.save_gcs,
        gcs_bucket_name=args.gcs_bucket_name,
        gcs_prefix=args.gcs_prefix,
        gcs_project=args.gcs_project
    )
    
    results = []
    for i, query in enumerate(queries):
        current_index = start_index + i
        print(f"\n🔄 [{i+1}/{len(queries)}] Processing query {current_index}: '{query}'")
        
        # Create debug directory for this query
        debug_dir = args.debug_root / f"query_{current_index:04d}"
        
        try:
            # Run optimization
            optimizer = GeneticQueryOptimizer(config, debug_dir)
            best_individual = await optimizer.run_optimization(query)
            
            # Calculate improvement
            improvement = optimizer._calculate_improvement()
            relative_improvement = improvement.get('relative_improvement', 0) if improvement else 0
            
            results.append({
                'index': current_index,
                'original': query,
                'optimized': best_individual.query,
                'fitness': best_individual.fitness_score,
                'improvement': relative_improvement
            })
            
            print(f"✅ Success! Fitness: {best_individual.fitness_score:.3f}, Improvement: {relative_improvement:.1f}%")
            logger.info(f"Query {current_index} optimization completed:")
            logger.info(f"  Original: '{query}'")
            logger.info(f"  Optimized: '{best_individual.query}'")
            logger.info(f"  Fitness: {best_individual.fitness_score:.3f}")
            logger.info(f"  Improvement: {relative_improvement:.1f}%")
            
        except Exception as e:
            print(f"❌ Error optimizing query {current_index}: {e}")
            logger.error(f"Query {current_index} optimization failed:")
            logger.error(f"  Query: '{query}'")
            logger.error(f"  Debug dir: {debug_dir}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            results.append({
                'index': current_index,
                'original': query,
                'optimized': None,
                'fitness': None,
                'improvement': None,
                'error': str(e)
            })
    
    # Print summary
    successful = [r for r in results if r.get('optimized')]
    print(f"\n{'='*60}")
    print(f"🧬 BATCH OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total queries: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(results) - len(successful)}")
    
    if successful:
        avg_improvement = sum(r['improvement'] for r in successful if r['improvement']) / len(successful)
        print(f"📈 Average improvement: {avg_improvement:.1f}%")
        
        # Show best improvements
        best = sorted(successful, key=lambda x: x['improvement'] or 0, reverse=True)[:3]
        print(f"\n🏆 Top improvements:")
        for i, result in enumerate(best, 1):
            print(f"  {i}. '{result['original']}' → '{result['optimized']}' (+{result['improvement']:.1f}%)")
    
    # Final log entry and ensure all logs are flushed
    logger.info(f"Batch optimization completed. Results saved to: {args.debug_root}")
    
    # Flush the logger to ensure all messages are written to file
    for handler in logger.handlers:
        handler.flush()
    
    # Save batch results (including uploading log file to GCS)
    await save_batch_results(results, args, logger)
    print(f"📁 Results saved to: {args.debug_root}")


if __name__ == "__main__":
    asyncio.run(main())