import os
import base64
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


SYSTEM_PROMPT = (
    "You are AKASHA, an elite Earth Observation & Satellite Intelligence AI. "
    "You possess deep expertise in remote sensing, multispectral imagery interpretation, "
    "land use and land cover (LULC) classification, urban growth monitoring, environmental assessment, "
    "and bi-temporal satellite change detection. Provide precise, technical, and structured insights."
)


def query_qwen(query: str, image_data_list: list = None) -> dict:
    """
    Send a combined text + image query to the Qwen API (DashScope).
    - No images → text-only chat via qwen-max
    - 1 image  → single image analysis via qwen-vl-max
    - 2+ images → bi-temporal change detection via qwen-vl-max
    Falls back to OpenAI (gpt-4o / gpt-4o-mini) if only OPENAI_API_KEY is set.
    """
    qwen_api_key = os.getenv("QWEN_API_KEY", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")

    images = image_data_list or []
    has_images = len(images) > 0
    is_bitemporal = len(images) >= 2

    # ── Pick provider & model ────────────────────────────────────────────────
    if OpenAI is None:
        return {
            "model_used": "none",
            "response": "⚠️ openai library is not installed in the environment.",
            "image_count": len(images),
            "mode": "error"
        }

    if qwen_api_key and qwen_api_key not in ("", "your_qwen_api_key_here"):
        client = OpenAI(
            api_key=qwen_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        model = "qwen-vl-max" if has_images else "qwen-max"
    elif openai_api_key and openai_api_key not in ("", "your_openai_api_key_here"):
        client = OpenAI(api_key=openai_api_key)
        model = "gpt-4o" if has_images else "gpt-4o-mini"
    else:
        return {
            "model_used": "none",
            "response": (
                "⚠️ No API key configured. "
                "Add QWEN_API_KEY (preferred) or OPENAI_API_KEY to backend/.env"
            ),
            "image_count": len(images),
            "mode": "error"
        }

    # ── Build message payload ────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if has_images:
        user_content = []

        if is_bitemporal:
            # Multi-image: bi-temporal change detection
            prompt_text = (
                f"{query or 'Perform comprehensive change detection between these satellite images.'}\n\n"
                f"You are provided with {len(images)} sequential satellite images. "
                "Systematically analyze the temporal sequence:\n"
                "1. Urban & Infrastructure Changes: new buildings, roads, demolitions.\n"
                "2. Vegetation & Canopy: quantify gains or losses.\n"
                "3. Hydrological Changes: water bodies, drainage, moisture.\n"
                "4. Summary: concise environmental/urban trajectory assessment."
            )
        else:
            # Single image analysis
            prompt_text = (
                query or
                "Analyze this satellite image. Identify land cover classes, "
                "infrastructure, and key geographical features."
            )

        user_content.append({"type": "text", "text": prompt_text})

        for idx, img in enumerate(images, start=1):
            b64 = base64.b64encode(img["bytes"]).decode("utf-8")
            mime = img.get("content_type") or "image/png"
            if is_bitemporal:
                user_content.append({
                    "type": "text",
                    "text": f"--- Frame {idx} / {img.get('filename', f'image_{idx}')} ---"
                })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })

        messages.append({"role": "user", "content": user_content})
    else:
        # Text-only query
        messages.append({"role": "user", "content": query})

    # ── Call API ─────────────────────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4 if has_images else 0.7,
            max_tokens=1500
        )
        reply = response.choices[0].message.content
        mode = (
            "bitemporal_change_detection" if is_bitemporal
            else ("image_analysis" if has_images else "text_query")
        )
        return {
            "model_used": model,
            "response": reply,
            "image_count": len(images),
            "mode": mode
        }

    except Exception as e:
        print(f"[AKASHA] API Error: {e}")
        return {
            "model_used": model,
            "response": f"⚠️ Error querying {model}: {str(e)}",
            "image_count": len(images),
            "mode": "error"
        }
