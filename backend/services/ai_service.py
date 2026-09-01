async def analyze_image(
    image_bytes: bytes,
    query: str,
    mode: str,
):
    """
    Main AI processing entry point.

    For now this is a placeholder.
    Later this function will call:
    - Qwen
    - GeoChat
    - M2CD
    - TeoChat
    """

    return {
        "status": "success",
        "mode": mode,
        "query": query,
        "analysis": (
            "AI service is connected. "
            "Model processing will be added next."
        ),
    }