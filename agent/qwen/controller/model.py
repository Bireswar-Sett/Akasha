from __future__ import annotations

import json
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from qwen.controller.services import QWEN_MODEL_ID


class QwenEngine:
    """
    Persistent Qwen2.5-7B-Instruct inference engine.

    The engine exposes:
      - chat()
      - chat_with_tools()

    It deliberately contains no satellite workflow logic. That belongs to
    QwenController and ToolExecutor.
    """

    def __init__(
        self,
        model_id: str = QWEN_MODEL_ID,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen requires a CUDA-capable runtime. "
                "Use a ZeroGPU/GPU Hugging Face Space."
            )

        self.model_id = model_id

        print(f"[Qwen] Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[Qwen] Building 4-bit quantization config...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        print(f"[Qwen] Loading model: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.eos_token_id = self.tokenizer.eos_token_id

        print(f"[Qwen] Device: {self.device}")
        print("[Qwen] Model ready.")

    @staticmethod
    def _validate_messages(
        messages: list[dict[str, Any]],
    ) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list.")

        valid_roles = {
            "system",
            "user",
            "assistant",
            "tool",
        }

        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("Each message must be a dictionary.")

            role = message.get("role")
            if role not in valid_roles:
                raise ValueError(f"Unsupported message role: {role!r}")

            if role != "tool" and "content" not in message:
                raise ValueError(
                    "Every non-tool message must contain content."
                )

    @staticmethod
    def _validate_generation_parameters(
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> None:
        if not 1 <= max_new_tokens <= 2048:
            raise ValueError(
                "max_new_tokens must be between 1 and 2048."
            )

        if temperature < 0:
            raise ValueError("temperature must be >= 0.")

        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in the range (0, 1].")

        if not isinstance(do_sample, bool):
            raise TypeError("do_sample must be bool.")

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        self._validate_messages(messages)

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        if tools:
            kwargs["tools"] = tools

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                "Qwen chat-template processing failed."
            ) from exc

    def _tokenize(
        self,
        text: str,
    ) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        return {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

    @torch.inference_mode()
    def _generate(
        self,
        inputs: dict[str, torch.Tensor],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": do_sample,
            "pad_token_id": self.eos_token_id,
        }

        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        return self.model.generate(
            **inputs,
            **kwargs,
        )

    @torch.inference_mode()
    def chat(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        self._validate_generation_parameters(
            max_new_tokens,
            temperature,
            top_p,
            do_sample,
        )

        text = self._apply_chat_template(messages)
        inputs = self._tokenize(text)

        outputs = self._generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]

        return self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        ).strip()

    @torch.inference_mode()
    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(tools, list) or not tools:
            raise ValueError("tools must be a non-empty list.")

        self._validate_generation_parameters(
            max_new_tokens,
            temperature,
            top_p,
            do_sample,
        )

        text = self._apply_chat_template(
            messages,
            tools=tools,
        )
        inputs = self._tokenize(text)

        outputs = self._generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]

        raw_text = self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=False,
        ).strip()

        tool_calls = self._parse_tool_calls(raw_text)

        if tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "raw": raw_text,
            }

        clean_text = self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        ).strip()

        return {
            "type": "text",
            "content": clean_text,
            "raw": raw_text,
        }

    @staticmethod
    def _parse_tool_calls(
        text: str,
    ) -> list[dict[str, Any]]:
        import re

        candidates: list[str] = []
        stripped = text.strip()

        for match in re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            text,
            flags=re.DOTALL,
        ):
            if match.strip():
                candidates.append(match.strip())

        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)

        if stripped.startswith("[") and stripped.endswith("]"):
            candidates.append(stripped)

        for block in re.findall(
            r"```(?:json)?\s*(.*?)```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            if block.strip():
                candidates.append(block.strip())

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            calls = QwenEngine._normalize_tool_calls(parsed)
            if calls:
                return calls

        return []

    @staticmethod
    def _normalize_tool_calls(
        value: Any,
    ) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            if "tool_calls" in value:
                return QwenEngine._normalize_tool_calls(
                    value["tool_calls"]
                )

            if (
                isinstance(value.get("name"), str)
                and "arguments" in value
            ):
                arguments = value["arguments"]
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        return []

                if not isinstance(arguments, dict):
                    return []

                return [
                    {
                        "name": value["name"],
                        "arguments": arguments,
                    }
                ]

            if isinstance(value.get("function"), dict):
                function = value["function"]
                name = function.get("name")
                arguments = function.get("arguments", {})

                if not isinstance(name, str):
                    return []

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        return []

                if not isinstance(arguments, dict):
                    return []

                return [
                    {
                        "name": name,
                        "arguments": arguments,
                    }
                ]

            return []

        if isinstance(value, list):
            calls: list[dict[str, Any]] = []
            for item in value:
                calls.extend(
                    QwenEngine._normalize_tool_calls(item)
                )
            return calls

        return []

    def ask(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **generation_kwargs: Any,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        messages: list[dict[str, Any]] = []

        if system_prompt and system_prompt.strip():
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt.strip(),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt.strip(),
            }
        )

        return self.chat(
            messages,
            **generation_kwargs,
        )

    def info(self) -> dict[str, Any]:
        allocated = 0.0
        reserved = 0.0

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3

        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "allocated_gb": allocated,
            "reserved_gb": reserved,
        }
