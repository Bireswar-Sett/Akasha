from __future__ import annotations

import json
import logging
import re
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from qwen.controller.services import QWEN_MODEL_ID


logger = logging.getLogger("akasha.qwen.model")


class QwenEngine:
    """
    Persistent Qwen2.5-7B-Instruct inference engine.

    Responsibilities:
        - load tokenizer/model
        - format chat messages
        - tokenize input
        - generate text
        - parse structured tool-call output

    This class contains NO remote-sensing workflow logic.

    The controller decides:
        WHAT should happen

    The executor decides:
        HOW specialist tools are physically invoked
    """

    def __init__(
        self,
        model_id: str = QWEN_MODEL_ID,
    ) -> None:
        if not isinstance(
            model_id,
            str,
        ) or not model_id.strip():
            raise ValueError(
                "model_id must be a non-empty string."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen requires a CUDA-capable runtime."
            )

        self.model_id = model_id.strip()

        logger.info(
            "Loading Qwen tokenizer: %s",
            self.model_id,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        logger.info(
            "Building Qwen 4-bit quantization configuration."
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        logger.info(
            "Loading Qwen model: %s",
            self.model_id,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )

        self.model.eval()

        self.device = self._resolve_device()

        self.eos_token_id = (
            self.tokenizer.eos_token_id
        )

        self.pad_token_id = (
            self.tokenizer.pad_token_id
        )

        logger.info(
            "Qwen device: %s",
            self.device,
        )

        logger.info(
            "Qwen model ready."
        )

    # ------------------------------------------------------------------
    # Model/device helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        """
        Resolve the model's effective input device.

        With quantized/device-mapped models, parameters may not all live
        on the same device. The first available parameter is the safest
        input-placement reference for the current deployment.
        """
        try:
            return next(
                self.model.parameters()
            ).device
        except StopIteration as exc:
            raise RuntimeError(
                "Qwen model contains no parameters."
            ) from exc

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_messages(
        messages: list[dict[str, Any]],
    ) -> None:
        if (
            not isinstance(messages, list)
            or not messages
        ):
            raise ValueError(
                "messages must be a non-empty list."
            )

        valid_roles = {
            "system",
            "user",
            "assistant",
            "tool",
        }

        for index, message in enumerate(
            messages
        ):
            if not isinstance(
                message,
                dict,
            ):
                raise TypeError(
                    f"Message {index} must be a dictionary."
                )

            role = message.get(
                "role"
            )

            if role not in valid_roles:
                raise ValueError(
                    f"Unsupported message role: {role!r}"
                )

            if role != "tool":
                if "content" not in message:
                    raise ValueError(
                        f"Message {index} must contain content."
                    )

                content = message.get(
                    "content"
                )

                if not isinstance(
                    content,
                    str,
                ):
                    raise TypeError(
                        f"Message {index} content must be a string."
                    )

    @staticmethod
    def _validate_generation_parameters(
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> None:
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(
                max_new_tokens,
                int,
            )
        ):
            raise TypeError(
                "max_new_tokens must be an integer."
            )

        if not 1 <= max_new_tokens <= 2048:
            raise ValueError(
                "max_new_tokens must be between 1 and 2048."
            )

        if (
            isinstance(temperature, bool)
            or not isinstance(
                temperature,
                (int, float),
            )
        ):
            raise TypeError(
                "temperature must be numeric."
            )

        if temperature < 0:
            raise ValueError(
                "temperature must be >= 0."
            )

        if (
            isinstance(top_p, bool)
            or not isinstance(
                top_p,
                (int, float),
            )
        ):
            raise TypeError(
                "top_p must be numeric."
            )

        if not 0 < top_p <= 1:
            raise ValueError(
                "top_p must be in the range (0, 1]."
            )

        if not isinstance(
            do_sample,
            bool,
        ):
            raise TypeError(
                "do_sample must be bool."
            )

        if not do_sample and temperature != 0:
            # Temperature is ignored by generate() when sampling is off.
            # Keeping the caller's value is harmless, but we normalize it
            # internally to avoid implying sampling behavior.
            logger.debug(
                "do_sample=False; temperature will not affect generation."
            )

    # ------------------------------------------------------------------
    # Chat formatting
    # ------------------------------------------------------------------

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        self._validate_messages(
            messages
        )

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        if tools:
            if not isinstance(
                tools,
                list,
            ):
                raise TypeError(
                    "tools must be a list."
                )

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

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(
        self,
        text: str,
    ) -> dict[str, torch.Tensor]:
        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError(
                "text must be a non-empty string."
            )

        try:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Qwen tokenization failed."
            ) from exc

        return {
            key: value.to(
                self.device
            )
            for key, value in encoded.items()
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _generate(
        self,
        inputs: dict[str, torch.Tensor],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> torch.Tensor:
        self._validate_generation_parameters(
            max_new_tokens,
            temperature,
            top_p,
            do_sample,
        )

        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": do_sample,
            "pad_token_id": self.pad_token_id,
        }

        if self.eos_token_id is not None:
            kwargs["eos_token_id"] = (
                self.eos_token_id
            )

        if do_sample:
            kwargs["temperature"] = float(
                temperature
            )
            kwargs["top_p"] = float(
                top_p
            )

        try:
            return self.model.generate(
                **inputs,
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                "Qwen generation failed."
            ) from exc

    # ------------------------------------------------------------------
    # Normal text chat
    # ------------------------------------------------------------------

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

        text = self._apply_chat_template(
            messages
        )

        inputs = self._tokenize(
            text
        )

        outputs = self._generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        input_length = (
            inputs["input_ids"].shape[1]
        )

        generated_tokens = (
            outputs[:, input_length:]
        )

        result = self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        ).strip()

        return result

    # ------------------------------------------------------------------
    # Tool-aware chat
    # ------------------------------------------------------------------

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
        if (
            not isinstance(
                tools,
                list,
            )
            or not tools
        ):
            raise ValueError(
                "tools must be a non-empty list."
            )

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

        inputs = self._tokenize(
            text
        )

        outputs = self._generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        input_length = (
            inputs["input_ids"].shape[1]
        )

        generated_tokens = (
            outputs[:, input_length:]
        )

        raw_text = self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=False,
        ).strip()

        tool_calls = self._parse_tool_calls(
            raw_text
        )

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

    # ------------------------------------------------------------------
    # Tool-call parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(
        text: str,
    ) -> list[dict[str, Any]]:
        """
        Parse common Qwen/tool-call output forms.

        Accepted forms include:

            <tool_call>
            {"name": "...", "arguments": {...}}
            </tool_call>

        plain JSON objects/lists

        fenced JSON blocks

        OpenAI-style:
            {"function": {"name": "...", "arguments": {...}}}

        Parsing only normalizes syntax. It does NOT authorize a tool.
        The Python executor remains the authority.
        """

        if (
            not isinstance(
                text,
                str,
            )
            or not text.strip()
        ):
            return []

        candidates: list[str] = []

        stripped = text.strip()

        # Native Qwen-style tool tags.
        for match in re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            text,
            flags=re.DOTALL,
        ):
            value = match.strip()

            if value:
                candidates.append(
                    value
                )

        # Entire response as JSON.
        if (
            stripped.startswith("{")
            and stripped.endswith("}")
        ):
            candidates.append(
                stripped
            )

        if (
            stripped.startswith("[")
            and stripped.endswith("]")
        ):
            candidates.append(
                stripped
            )

        # Fenced JSON.
        for block in re.findall(
            r"```(?:json)?\s*(.*?)```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            value = block.strip()

            if value:
                candidates.append(
                    value
                )

        # Avoid repeatedly parsing identical candidates.
        seen: set[str] = set()

        for candidate in candidates:
            if candidate in seen:
                continue

            seen.add(candidate)

            try:
                parsed = json.loads(
                    candidate
                )
            except json.JSONDecodeError:
                continue

            normalized = (
                QwenEngine._normalize_tool_calls(
                    parsed
                )
            )

            if normalized:
                return normalized

        return []

    @staticmethod
    def _normalize_tool_calls(
        value: Any,
    ) -> list[dict[str, Any]]:
        """
        Normalize tool-call JSON into:

            [
                {
                    "name": "...",
                    "arguments": {...}
                }
            ]

        This does not check whether the tool exists. That belongs to the
        executor/controller validation layer.
        """

        if isinstance(
            value,
            dict,
        ):
            if "tool_calls" in value:
                return (
                    QwenEngine._normalize_tool_calls(
                        value["tool_calls"]
                    )
                )

            if (
                isinstance(
                    value.get("name"),
                    str,
                )
                and "arguments" in value
            ):
                name = value["name"].strip()

                if not name:
                    return []

                arguments = value["arguments"]

                if isinstance(
                    arguments,
                    str,
                ):
                    try:
                        arguments = json.loads(
                            arguments
                        )
                    except json.JSONDecodeError:
                        return []

                if not isinstance(
                    arguments,
                    dict,
                ):
                    return []

                return [
                    {
                        "name": name,
                        "arguments": arguments,
                    }
                ]

            function = value.get(
                "function"
            )

            if isinstance(
                function,
                dict,
            ):
                name = function.get(
                    "name"
                )

                arguments = function.get(
                    "arguments",
                    {},
                )

                if not isinstance(
                    name,
                    str,
                ):
                    return []

                name = name.strip()

                if not name:
                    return []

                if isinstance(
                    arguments,
                    str,
                ):
                    try:
                        arguments = json.loads(
                            arguments
                        )
                    except json.JSONDecodeError:
                        return []

                if not isinstance(
                    arguments,
                    dict,
                ):
                    return []

                return [
                    {
                        "name": name,
                        "arguments": arguments,
                    }
                ]

            return []

        if isinstance(
            value,
            list,
        ):
            calls: list[dict[str, Any]] = []

            for item in value:
                calls.extend(
                    QwenEngine._normalize_tool_calls(
                        item
                    )
                )

            return calls

        return []

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    def ask(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **generation_kwargs: Any,
    ) -> str:
        if (
            not isinstance(
                prompt,
                str,
            )
            or not prompt.strip()
        ):
            raise ValueError(
                "prompt must be a non-empty string."
            )

        messages: list[dict[str, Any]] = []

        if (
            system_prompt
            and system_prompt.strip()
        ):
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

    # ------------------------------------------------------------------
    # Runtime information
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        allocated = 0.0
        reserved = 0.0

        if torch.cuda.is_available():
            try:
                allocated = (
                    torch.cuda.memory_allocated()
                    / 1024**3
                )
                reserved = (
                    torch.cuda.memory_reserved()
                    / 1024**3
                )
            except RuntimeError:
                pass

        gpu_name = None

        if torch.cuda.is_available():
            try:
                gpu_name = (
                    torch.cuda.get_device_name(0)
                )
            except RuntimeError:
                gpu_name = None

        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "gpu": gpu_name,
            "allocated_gb": round(
                allocated,
                3,
            ),
            "reserved_gb": round(
                reserved,
                3,
            ),
        }