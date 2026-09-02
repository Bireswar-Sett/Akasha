"""
Satellite image analysis via Qwen Hugging Face Gradio Space.
This module is the canonical backend entry point for all AI inference.
All calls go through QwenService -> AdityaSingh1531/qwen -> /ask_akasha.
"""
from services.qwen_service import QwenService, get_qwen_service

__all__ = ["QwenService", "get_qwen_service"]
