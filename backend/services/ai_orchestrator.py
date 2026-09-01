def orchestrate_query(query: str, image_metadata: list = None) -> dict:
    return {
        "selected_model": "Qwen-VL",
        "confidence": 99.0,
        "reasoning": "Direct query forwarded to Qwen AI model."
    }

