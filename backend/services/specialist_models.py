import os
from openai import OpenAI

def execute_model(model_name: str, query: str, image_metadata: list = None) -> dict:
    """
    Directly sends user query to AI model (OpenAI for now, easily swappable to Qwen API).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    qwen_api_key = os.getenv("QWEN_API_KEY")

    try:
        # Check if Qwen API key is provided, or fallback to OpenAI
        if qwen_api_key and qwen_api_key != "your_qwen_api_key_here":
            client = OpenAI(
                api_key=qwen_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            model_to_use = "qwen-max"
        elif api_key:
            client = OpenAI(api_key=api_key)
            model_to_use = "gpt-3.5-turbo"
        else:
            return {
                "model_used": model_name or "Qwen-VL",
                "response": "⚠️ No API key configured. Please set OPENAI_API_KEY or QWEN_API_KEY.",
                "visual_evidence_url": None
            }

        system_prompt = (
            "You are AKASHA, an expert Satellite Intelligence & Remote Sensing AI Assistant. "
            "Help the user analyze satellite imagery, geospatial features, land cover, and remote sensing queries."
        )

        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=800
        )

        reply_text = response.choices[0].message.content

        return {
            "model_used": model_to_use,
            "response": reply_text,
            "visual_evidence_url": None
        }

    except Exception as e:
        print(f"AI API Error: {e}")
        return {
            "model_used": model_name or "Qwen-VL",
            "response": f"⚠️ Error processing query with AI model: {str(e)}",
            "visual_evidence_url": None
        }
