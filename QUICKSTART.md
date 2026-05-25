# Quick Start Guide - Policy Intelligence Platform

## Running the Full Pipeline

### Basic Usage
```bash
cd /Users/sahilg/policy-explorer.worktrees/agents-policy-intelligence-refactor
python3 pipeline/build_dataset.py
```

This will:
1. Load macro data (GDP + unemployment from 1997-2025)
2. Load historical bills
3. Fetch Congressional bills from Congress API
4. Classify bills using local Ollama LLM
5. Score each bill through the analog engine
6. Output unified JSON results

---

## Using Engines Directly

### Example 1: Score a Single Bill

```python
import pandas as pd
from pipeline.policy_impact_linker import OutcomeEngine
from pipeline.policy_score import ScoringEngine
from pipeline.policy_embeddings import AnalogMatcher

# Load data
macro = pd.read_parquet("data/processed/policy_dataset.parquet")
historical = pd.read_parquet("data/processed/policy_dataset.parquet")

# Initialize engines
outcome_engine = OutcomeEngine(macro_df=macro)
scorer = ScoringEngine(gdp_weight=0.4, unemployment_weight=-0.3)
matcher = AnalogMatcher(historical_bills_df=historical)

# Define the bill to score
target_policy = {
    "title": "Economic Stimulus Package",
    "policy_type": "spending",
    "direction": "expansionary",
    "intensity": "high",
    "sector": "government"
}

# Step 1: Find historical analogs
analogs = matcher.find_similar_bills(target_policy, top_k=5)
print(f"Found {len(analogs)} analogs")

# Step 2: Estimate macro impacts
impacts = outcome_engine.estimate_directional_impacts(analogs)
print(f"GDP effect: {impacts['gdp_effect']:.3f}")
print(f"Unemployment effect: {impacts['unemployment_effect']:.3f}")

# Step 3: Calculate score and confidence
score_result = scorer.calculate_net_score(impacts, analogs)
print(f"Net score: {score_result['net_score']:.3f}")
print(f"Confidence: {score_result['confidence']:.3f}")
```

---

### Example 2: Batch Processing Multiple Bills

```python
import json
import pandas as pd
from pipeline.policy_impact_linker import OutcomeEngine
from pipeline.policy_score import ScoringEngine
from pipeline.policy_embeddings import AnalogMatcher

# Initialize
macro = pd.read_parquet("data/processed/policy_dataset.parquet")
historical = pd.read_parquet("data/processed/policy_dataset.parquet")

engines = {
    'outcome': OutcomeEngine(macro_df=macro),
    'scorer': ScoringEngine(),
    'matcher': AnalogMatcher(historical_bills_df=historical)
}

# Bills to score
bills = [
    {
        "bill_id": "S.1000",
        "title": "Economic Stimulus Package",
        "policy_type": "spending",
        "direction": "expansionary",
        "intensity": "high",
        "sector": "government"
    },
    {
        "bill_id": "H.R.500",
        "title": "Tax Relief Act",
        "policy_type": "tax",
        "direction": "contractionary",
        "intensity": "medium",
        "sector": "business"
    }
]

# Score each bill
results = []
for bill in bills:
    target = {k: bill[k] for k in ['title', 'policy_type', 'direction', 'intensity', 'sector']}
    
    analogs = engines['matcher'].find_similar_bills(target, top_k=5)
    impacts = engines['outcome'].estimate_directional_impacts(analogs)
    score = engines['scorer'].calculate_net_score(impacts, analogs)
    
    result = {
        "bill_id": bill["bill_id"],
        "title": bill["title"],
        "policy_type": bill["policy_type"],
        "direction": bill["direction"],
        "similar_bills": [
            {
                "bill_id": a.get("bill_id"),
                "title": a.get("title"),
                "similarity_score": round(a.get("similarity_score", 0), 3)
            }
            for a in analogs
        ],
        "estimated_impacts": {
            "gdp_effect": round(impacts['gdp_effect'], 3),
            "unemployment_effect": round(impacts['unemployment_effect'], 3),
            "num_analogs_matched": impacts['num_analogs_matched'],
            "avg_similarity": round(impacts['avg_similarity'], 3)
        },
        "net_score": round(score['net_score'], 3),
        "confidence": round(score['confidence'], 3)
    }
    results.append(result)

# Output results
print(json.dumps(results, indent=2))
```

---

### Example 3: Custom Similarity Weighting

```python
from pipeline.policy_score import ScoringEngine

# Create scorer with custom weights
scorer = ScoringEngine(
    gdp_weight=0.6,        # More emphasis on GDP (was 0.4)
    unemployment_weight=-0.2  # Less emphasis on unemployment (was -0.3)
)

# Use as normal
score_result = scorer.calculate_net_score(impacts, analogs)
```

---

### Example 4: Handling Edge Cases

```python
from pipeline.policy_impact_linker import OutcomeEngine
import pandas as pd

# Initialize with missing data
engine = OutcomeEngine(macro_df=pd.DataFrame())

# This won't crash - returns neutral impacts
impacts = engine.estimate_directional_impacts([
    {"bill_id": "bad_1", "similarity_score": 0.9}  # No date
])

print(impacts)
# Output: {'gdp_effect': 0.0, 'unemployment_effect': 0.0, 'num_analogs_matched': 0, 'avg_similarity': 0.0}
```

---

## Understanding the Output

### Full Result JSON
```json
{
  "bill_id": "S.1000",
  "title": "Economic Stimulus Package",
  "policy_type": "spending",
  "direction": "expansionary",
  "similar_bills": [
    {
      "bill_id": "hist_2010",
      "title": "Recovery Act of 2009",
      "similarity_score": 0.95
    }
  ],
  "estimated_impacts": {
    "gdp_effect": 0.032,
    "unemployment_effect": -0.001,
    "num_analogs_matched": 5,
    "avg_similarity": 0.875
  },
  "net_score": 0.013,
  "confidence": 0.773,
  "explanation": "..."
}
```

### Key Metrics Explained

**net_score** (-1.0 to 1.0)
- Ranges from -1.0 (very negative) to 1.0 (very positive)
- Calculated as: (gdp_effect × 0.4) + (unemployment_effect × -0.3)
- Example: 0.013 means slightly positive economic impact

**confidence** (0.0 to 1.0)
- Measures certainty in the estimate
- Based on quality (similarity) and quantity (number) of analogs
- Example: 0.773 means high confidence (77%)

**gdp_effect** (-1.0 to 1.0)
- Historical GDP percentage change (year-over-year)
- Positive = GDP growth, Negative = GDP contraction
- Example: 0.032 means 3.2% GDP growth from historical precedents

**unemployment_effect** (-1.0 to 1.0)
- Historical unemployment rate change (absolute points)
- Positive = unemployment increased, Negative = unemployment decreased
- Example: -0.001 means unemployment decreased by 0.1 percentage points

---

## Logging and Debugging

### Enable Detailed Logging
```python
import logging

# Set to DEBUG for maximum verbosity
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(name)s] %(levelname)s: %(message)s"
)

# Now run your code
```

### Example Debug Output
```
[pipeline.policy_embeddings] Searching for 5 analogs to: Economic Stimulus Package
[pipeline.policy_embeddings] Found 5 analogs (avg sim: 0.875)
[pipeline.policy_impact_linker] Estimating impacts from 5 analogs
[pipeline.policy_impact_linker] Matched bill hist_2010: GDP 0.027, Unemp -0.001, Sim 0.95
[pipeline.policy_impact_linker] Impact estimation complete: GDP=0.032, Unemp=0.000, Matched=5, AvgSim=0.875
[pipeline.policy_score] Score calculation: GDP_comp=0.013, Unemp_comp=-0.000, Net=0.013, Conf=0.773
```

---

## Performance Tips

1. **Cache Historical Bills**: Load once, reuse
   ```python
   historical = pd.read_parquet("data/processed/policy_dataset.parquet")
   matcher = AnalogMatcher(historical_bills_df=historical)
   # Reuse matcher for multiple bills
   ```

2. **Batch Process**: Score multiple bills efficiently
   ```python
   for bill in bills:
       # Process together to amortize setup cost
   ```

3. **Reduce Top-K**: Use fewer analogs if speed matters more than accuracy
   ```python
   analogs = matcher.find_similar_bills(target, top_k=3)  # Instead of 5
   ```

---

## Common Issues & Solutions

### Issue: No macro data matches
```
[pipeline.policy_impact_linker] No analogs could be matched to macro data
```
**Solution**: Ensure analog bills have dates in 1997-2025 and state is "United States"

### Issue: Empty historical bills
```
[pipeline.policy_embeddings] No historical bills available
```
**Solution**: Load historical data: `historical = pd.read_parquet("data/processed/policy_dataset.parquet")`

### Issue: Ollama connection error
```
requests.exceptions.ConnectionError: Failed to establish connection to localhost:11434
```
**Solution**: Start Ollama service: `ollama serve` (requires Ollama installed locally)

---

## Integration with Your Application

1. Import the engines:
   ```python
   from pipeline.policy_impact_linker import OutcomeEngine
   from pipeline.policy_score import ScoringEngine
   from pipeline.policy_embeddings import AnalogMatcher
   ```

2. Initialize with your data:
   ```python
   outcome_engine = OutcomeEngine(macro_df=your_macro_data)
   scorer = ScoringEngine()
   matcher = AnalogMatcher(historical_bills_df=your_bills)
   ```

3. Score bills:
   ```python
   analogs = matcher.find_similar_bills(target_policy, top_k=5)
   impacts = outcome_engine.estimate_directional_impacts(analogs)
   score = scorer.calculate_net_score(impacts, analogs)
   ```

4. Use results:
   ```python
   result = {
       "bill_id": bill_id,
       "net_score": score['net_score'],
       "confidence": score['confidence'],
       "similar_bills": analogs,
       "estimated_impacts": impacts
   }
   ```

---

## Testing

```bash
# Quick validation test
python3 << 'EOF'
from pipeline.policy_impact_linker import OutcomeEngine
from pipeline.policy_score import ScoringEngine
from pipeline.policy_embeddings import AnalogMatcher
import pandas as pd

macro = pd.read_parquet("data/processed/policy_dataset.parquet")
engine = OutcomeEngine(macro_df=macro)
scorer = ScoringEngine()

print("✓ All engines loaded successfully")
EOF
```

---

## Next Steps

1. Review the architecture in `REFACTORING_SUMMARY.md`
2. Explore the implementation in `IMPLEMENTATION_DETAILS.md`
3. Try the examples above
4. Integrate with your application
5. Customize weights and parameters as needed
6. Add features (caching, visualization, etc.)

Happy scoring! 🎉
