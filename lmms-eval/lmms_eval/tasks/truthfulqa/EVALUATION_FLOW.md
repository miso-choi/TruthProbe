# TruthfulQA Evaluation Flow - Where `run_end2end_OpenAI_evaluation` is Called

## Answer: It's called from the aggregation functions

The `run_end2end_OpenAI_evaluation` function is **called inside the aggregation functions**:
- `truthfulqa_aggregate_generation_truth()` (lines 254-258)
- `truthfulqa_aggregate_generation_info()` (lines 306-310)

## Complete Evaluation Flow

### 1. Model Generation Phase
```
lmms-eval framework
  ↓
truthfulqa_doc_to_text()  # Formats prompt with instruction prefix
  ↓
Model generates answer (generate_until)
  ↓
Results passed to truthfulqa_process_results()
```

### 2. Result Processing Phase (stores data for evaluation)

**File**: `lmms-eval/lmms_eval/tasks/truthfulqa/utils.py`

**Function**: `truthfulqa_process_results(doc, results)` (lines 127-178)

**Flow**:
```python
def truthfulqa_process_results(doc, results):
    # 1. Extract and post-process generated answer
    model_gen_str = results[0].strip()
    # Remove everything after 'Q:', keep after 'A:'
    
    # 2. Extract question from doc
    question = doc.get('question', doc.get('Question', ''))
    
    # 3. Return results with question and answer (no evaluation yet)
    return {
        "question": question,
        "generated_answer": model_gen_str,
        "full_response": results[0],
        ...
    }
```

### 3. Aggregation Phase (where OpenAI evaluation happens)

**Functions**: 
- `truthfulqa_aggregate_generation_truth(results)` (lines 218-264)
- `truthfulqa_aggregate_generation_info(results)` (lines 267-313)

**Flow**:
```python
def truthfulqa_aggregate_generation_truth(results):
    # 1. Get model names from environment
    judge_name = os.getenv("TRUTHFULQA_JUDGE_NAME", None)
    info_name = os.getenv("TRUTHFULQA_INFO_NAME", None)
    
    # 2. Iterate through all results
    scores = []
    for result in results:
        question = result.get("question", "")
        generated_answer = result.get("generated_answer", "")
        
        # 3. CALL OpenAI evaluation for each result
        doc = {"question": question, "Question": question}
        judge_score = run_end2end_OpenAI_evaluation(
            doc, generated_answer, metric_type="judge",
            judge_name=judge_name, info_name=info_name
        )
        scores.append(judge_score)
    
    # 4. Return average score
    return sum(scores) / len(scores)
```

The same pattern applies to `truthfulqa_aggregate_generation_info()`.

## Key Points

1. **When it's called**: During the **aggregation phase**, which happens **after** all documents have been processed. This is cleaner because:
   - Separation of concerns: `process_results` just processes raw output
   - Aggregation functions handle the evaluation logic
   - All evaluation happens in one place per metric

2. **How many times**: Called **once per document per metric**:
   - In `truthfulqa_aggregate_generation_truth`: once per document for judge (truthfulness)
   - In `truthfulqa_aggregate_generation_info`: once per document for info (informativeness)

3. **Configuration**: Model names can be set via environment variables:
   ```bash
   export TRUTHFULQA_JUDGE_NAME="your-judge-model-name"
   export TRUTHFULQA_INFO_NAME="your-info-model-name"
   export OPENAI_API_KEY="your-api-key"
   ```

4. **Error handling**: If OpenAI evaluation fails for a document, it returns 0.0 for that document and prints a warning, but continues processing other documents.

## Integration with lmms-eval Framework

The lmms-eval framework automatically:
1. Calls `truthfulqa_process_results` for each document after generation
2. Collects all results
3. Calls aggregation functions (`truthfulqa_aggregate_generation_truth`, etc.) to compute final metrics
4. Reports the aggregated scores

## Example Execution Flow

```
Document 1:
  → Generate answer: "The answer is..."
  → truthfulqa_process_results() called
  → Returns: {"question": "...", "generated_answer": "The answer is..."}

Document 2:
  → Generate answer: "I think..."
  → truthfulqa_process_results() called
  → Returns: {"question": "...", "generated_answer": "I think..."}

After all documents processed:
  → truthfulqa_aggregate_generation_truth([doc1_result, doc2_result, ...]) called
    → For doc1: run_end2end_OpenAI_evaluation(judge) → 0.85
    → For doc2: run_end2end_OpenAI_evaluation(judge) → 0.78
    → Average: (0.85 + 0.78 + ...) / N → 0.815
  
  → truthfulqa_aggregate_generation_info([doc1_result, doc2_result, ...]) called
    → For doc1: run_end2end_OpenAI_evaluation(info) → 0.92
    → For doc2: run_end2end_OpenAI_evaluation(info) → 0.88
    → Average: (0.92 + 0.88 + ...) / N → 0.900
```

## Current Implementation Details

The function uses your custom OpenAI API interface:
```python
resp = client.responses.create(
    model=model_name,
    input=prompt,
    temperature=0.0,
    max_output_tokens=16
)
content = resp.output[0].content[0].text.strip()
score = float(content)
```

This is called from:
- `truthfulqa_aggregate_generation_truth` at lines 254-258
- `truthfulqa_aggregate_generation_info` at lines 306-310

