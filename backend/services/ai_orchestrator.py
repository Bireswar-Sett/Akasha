import os

def determine_specialist_model(query: str) -> str:
    """
    Mock implementation of Qwen orchestrator.
    In a real scenario, this would call Qwen API to classify the intent.
    """
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in ["change", "before and after", "difference"]):
        return "M2CD"
    elif any(keyword in query_lower for keyword in ["time", "temporal", "series", "progression"]):
        return "TEOChat"
    elif any(keyword in query_lower for keyword in ["classify", "detect", "feature", "type"]):
        return "GeoVision"
    else:
        # Default to GeoChat for general VQA
        return "GeoChat"

def orchestrate_query(query: str, image_metadata: list = None) -> dict:
    model_choice = determine_specialist_model(query)
    
    return {
        "selected_model": model_choice,
        "confidence": 88.5, # Mock confidence score
        "reasoning": f"Query contains keywords that align with {model_choice} capabilities."
    }
