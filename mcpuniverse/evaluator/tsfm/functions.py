"""
Evaluation functions for tsfm
"""
import json
from typing import Any
from mcpuniverse.evaluator.functions import eval_func, compare_func, FunctionResult

@eval_func(name="extract_score")
async def extract_score(x: FunctionResult, *args, **kwargs) -> FunctionResult:
    """Extract numerical score from response."""
    if isinstance(x, FunctionResult):
        data = x.result
        if isinstance(data, dict) and 'score' in data:
            return FunctionResult(result=float(data['score']))
        elif isinstance(data, str):
            # Try to extract number from string
            import re
            match = re.search(r'\d+\.?\d*', data)
            if match:
                return FunctionResult(result=float(match.group()))
    raise ValueError("Could not extract score from input")

@eval_func(name="normalize_text")
async def normalize_text(x: FunctionResult, *args, **kwargs) -> FunctionResult:
    """Normalize text for comparison."""
    if isinstance(x, FunctionResult):
        text = str(x.result).lower().strip()
        # Remove extra whitespace
        normalized = ' '.join(text.split())
        return FunctionResult(result=normalized)
    raise ValueError("Could not normalize text")

@compare_func(name="score_threshold")
async def score_threshold(a: Any, b: Any, *args, **kwargs) -> tuple[bool, str]:
    """Check if score meets threshold."""
    if isinstance(a, FunctionResult):
        a = a.result
    if isinstance(b, FunctionResult):
        b = b.result
    
    threshold = float(b)
    score = float(a)
    
    if score >= threshold:
        return True, ""
    return False, f"Score {score} below threshold {threshold}"

@compare_func(name="text_similarity")
async def text_similarity(a: Any, b: Any, *args, **kwargs) -> tuple[bool, str]:
    """Check text similarity using fuzzy matching."""
    from difflib import SequenceMatcher
    
    if isinstance(a, FunctionResult):
        a = a.result
    if isinstance(b, FunctionResult):
        b = b.result
    
    similarity = SequenceMatcher(None, str(a), str(b)).ratio()
    threshold = 0.8  # Default threshold
    
    if len(args) > 2 and args[2]:  # op_args provided
        threshold = float(args[2].get('threshold', 0.8))
    
    if similarity >= threshold:
        return True, ""
    return False, f"Text similarity {similarity:.2f} below threshold {threshold}"