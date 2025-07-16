# Getting Started with Genetic Query Optimization

This guide will help you quickly get started with the genetic algorithm for optimizing Etsy shopping queries.

## 🎯 What This Does

The genetic algorithm automatically improves your Etsy search queries by:

1. **Creating variations** of your original query using AI
2. **Testing each variation** with real shopping agents 
3. **Measuring performance** based on semantic relevance and purchase decisions
4. **Evolving better queries** over multiple generations
5. **Returning the best performing query**

## 🚀 Quick Start (5 minutes)

### 1. Test the System

First, verify everything works:

```bash
cd optimization-agent
python src/test_genetic_optimizer.py
```

You should see:
```
🎉 All tests passed!
✅ The genetic algorithm components are working correctly.
```

### 2. Run Your First Optimization

Start with a simple query:

```bash
python src/genetic_query_optimizer.py --query "vintage jewelry"
```

This will:
- Create 8 query variations
- Test each with 2 shopping agents
- Evolve over 5 generations
- Show you the best optimized query

### 3. View Results

After completion, analyze the results:

```bash
python src/visualize_optimization.py --analysis summary
```

## 📊 Example Session

Here's what a typical optimization looks like:

```bash
# Original query
$ python src/genetic_query_optimizer.py --query "coffee mug"

# Output
🧬 GENETIC ALGORITHM OPTIMIZATION COMPLETE
============================================================
Original query: 'coffee mug'
Best query found: 'handmade ceramic coffee mug'
Fitness improvement: 34.2%
Final fitness score: 0.723
Results saved to: src/genetic_optimization
============================================================
```

## ⚙️ Configuration Examples

### Quick Test (2 minutes)
```bash
python src/genetic_query_optimizer.py \
    --query "wooden spoon" \
    --population-size 4 \
    --generations 2 \
    --n-agents 2
```

### Balanced Optimization (5-10 minutes)
```bash
python src/genetic_query_optimizer.py \
    --query "handmade scarf" \
    --population-size 6 \
    --generations 3 \
    --n-agents 2
```

### Thorough Optimization (15-30 minutes)
```bash
python src/genetic_query_optimizer.py \
    --query "artisan leather bag" \
    --population-size 10 \
    --generations 5 \
    --n-agents 3
```

## 📁 Understanding Results

After optimization, you'll find:

```
src/genetic_optimization/
├── genetic_algorithm.log          # Detailed progress
├── optimization_results.json      # Complete results
└── gen_*/                         # Per-generation data
```

Key metrics to look for:
- **Fitness Score**: 0.0-1.0 (higher is better)
- **Improvement %**: How much better the new query performs
- **Semantic Relevance**: Number of highly relevant products found
- **Purchase Rate**: How often agents decide to buy products

## 🎚️ Key Parameters

| Parameter | Quick Test | Balanced | Thorough |
|-----------|------------|----------|----------|
| `--population-size` | 4 | 6-8 | 10-12 |
| `--generations` | 2 | 3-4 | 5-7 |
| `--n-agents` | 2 | 2-3 | 3-4 |
| Time needed | 2-5 min | 5-15 min | 15-45 min |

## 🔧 Advanced Usage

### Focus on Finding Relevant Products
```bash
python src/genetic_query_optimizer.py \
    --query "vintage poster" \
    --semantic-weight 0.9 \
    --purchase-weight 0.1
```

### Focus on Purchase Decisions  
```bash
python src/genetic_query_optimizer.py \
    --query "kitchen gadget" \
    --semantic-weight 0.6 \
    --purchase-weight 0.4
```

### Custom Output Directory
```bash
python src/genetic_query_optimizer.py \
    --query "art supplies" \
    --debug-root ./my_optimization
```

## 📈 Analyzing Results

### Quick Summary
```bash
python src/visualize_optimization.py --analysis summary
```

### Evolution Trends
```bash
python src/visualize_optimization.py --analysis evolution
```

### Best Queries Found
```bash
python src/visualize_optimization.py --analysis best
```

### Full Analysis
```bash
python src/visualize_optimization.py --analysis all
```

## 🎯 Tips for Success

### Choose Good Starting Queries
- ✅ "wooden kitchen utensils" (specific, clear intent)
- ✅ "vintage jewelry" (clear category)
- ❌ "stuff" (too vague)
- ❌ "the best product ever" (not searchable)

### Adjust Parameters Based on Goals
- **Speed over quality**: Reduce population-size and generations
- **Quality over speed**: Increase population-size and generations  
- **Find products**: Increase semantic-weight
- **Drive purchases**: Increase purchase-weight

### Interpret Results
- **High improvement %**: Algorithm found much better queries
- **Low improvement %**: Original query was already quite good
- **High fitness score (>0.7)**: Query performs very well
- **Low fitness score (<0.3)**: May need different approach

## 🔍 Troubleshooting

### "litellm.key file not found"
Copy your API key to `src/keys/litellm.key`

### Optimization runs slowly
- Reduce `--population-size` and `--generations`
- Reduce `--n-agents` and `--max-steps`

### Low fitness scores
- Try different weight combinations
- Ensure your query is specific and searchable
- Check if enough relevant products exist for your query

### Agent timeouts
- Reduce `--max-steps` (try 10-15)
- Check your internet connection
- Ensure Etsy is accessible

## 📚 Learn More

- Read the [complete documentation](./GENETIC_ALGORITHM_README.md)
- Run examples: `python src/example_optimization.py`
- Test components: `python src/test_genetic_optimizer.py`

## 🤝 Next Steps

1. Start with quick tests to understand the system
2. Try optimizing queries relevant to your use case  
3. Experiment with different parameter combinations
4. Analyze results to understand what makes queries successful
5. Scale up to thorough optimizations for production use

Happy optimizing! 🧬✨ 