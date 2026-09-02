from __future__ import annotations

import json
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


class QwenEngine:
    """
    Persistent Qwen inference engine.

    Responsibilities:
        - Load Qwen once.
        - Provide ordinary chat generation.
        - Provide structured tool-call generation.
        - Keep model-specific Transformers logic out of the controller.

    Public interfaces:

        chat(messages)
        chat_with_tools(messages, tools)

    Messages use the standard format:

        {
            "role": "user",
            "content": "..."
        }

    Tool definitions use the OpenAI-style function schema expected by
    modern Transformers/Qwen chat templates.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
    ) -> None:

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen requires a CUDA-capable GPU."
            )

        self.model_id = model_id

        print(
            f"[Qwen] Loading {model_id}..."
        )

        # ==============================================================
        # TOKENIZER
        # ==============================================================

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
            )
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        # ==============================================================
        # 4-BIT QUANTIZATION
        # ==============================================================

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        )

        # ==============================================================
        # MODEL
        # ==============================================================

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )

        self.model.eval()

        self.device = next(
            self.model.parameters()
        ).device

        self.eos_token_id = (
            self.tokenizer.eos_token_id
        )

        print(
            f"[Qwen] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "[Qwen] Model ready."
        )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_messages(
        messages: list[dict[str, Any]],
    ) -> None:

        if not isinstance(
            messages,
            list,
        ):
            raise TypeError(
                "messages must be a list."
            )

        if not messages:
            raise ValueError(
                "messages must not be empty."
            )

        valid_roles = {
            "system",
            "user",
            "assistant",
            "tool",
        }

        for message in messages:

            if not isinstance(
                message,
                dict,
            ):
                raise TypeError(
                    "Every message must be a dictionary."
                )

            role = message.get(
                "role"
            )

            if role not in valid_roles:
                raise ValueError(
                    f"Unsupported message role: "
                    f"{role!r}"
                )

            if (
                role != "tool"
                and "content" not in message
            ):
                raise ValueError(
                    "Non-tool messages must contain "
                    "'content'."
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
                "max_new_tokens must be between "
                "1 and 2048."
            )

        if temperature < 0:
            raise ValueError(
                "temperature must be >= 0."
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
                "do_sample must be a bool."
            )

    # ==================================================================
    # TOKENIZATION
    # ==================================================================

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Convert structured messages into Qwen's native chat format.

        When tools are provided, they are passed into the tokenizer's
        chat template so Qwen receives the tool definitions in the
        format expected by the model.
        """

        self._validate_messages(
            messages
        )

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        if tools is not None:
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

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        return {
            key: value.to(
                self.device
            )
            for key, value in inputs.items()
        }

    # ==================================================================
    # GENERATION
    # ==================================================================

    @torch.inference_mode()
    def _generate(
        self,
        inputs: dict[str, torch.Tensor],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> torch.Tensor:

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }

        if do_sample:

            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

        else:

            generation_kwargs.update(
                {
                    "do_sample": False,
                }
            )

        return self.model.generate(
            **inputs,
            **generation_kwargs,
        )

    # ==================================================================
    # NORMAL CHAT
    # ==================================================================

    @torch.inference_mode()
    def chat(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Normal Qwen conversation without tools.
        """

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

        generated_tokens = outputs[
            :,
            input_length:,
        ]

        return self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        ).strip()

    # ==================================================================
    # TOOL-CALLING CHAT
    # ==================================================================

    @torch.inference_mode()
    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> dict[str, Any]:
        """
        Ask Qwen to either:

            1. produce a normal assistant response, or
            2. emit one or more structured tool calls.

        Returns:

            {
                "type": "text",
                "content": "..."
            }

        or:

            {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "name": "...",
                        "arguments": {...}
                    }
                ],
                "raw": "..."
            }
        """

        if not isinstance(
            tools,
            list,
        ):
            raise TypeError(
                "tools must be a list."
            )

        if not tools:
            raise ValueError(
                "tools must not be empty."
            )

        self._validate_generation_parameters(
            max_new_tokens,
            temperature,
            top_p,
            do_sample,
        )

        # --------------------------------------------------------------
        # Build Qwen prompt with tools.
        # --------------------------------------------------------------

        text = self._apply_chat_template(
            messages,
            tools=tools,
        )

        # --------------------------------------------------------------
        # Tokenize.
        # --------------------------------------------------------------

        inputs = self._tokenize(
            text
        )

        # --------------------------------------------------------------
        # Generate.
        # --------------------------------------------------------------

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

        generated_tokens = outputs[
            :,
            input_length:,
        ]

        raw_text = self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=False,
        ).strip()

        # --------------------------------------------------------------
        # Try to extract structured tool calls.
        # --------------------------------------------------------------

        tool_calls = self._parse_tool_calls(
            raw_text
        )

        if tool_calls:

            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "raw": raw_text,
            }

        # --------------------------------------------------------------
        # No tool call.
        # --------------------------------------------------------------

        clean_text = self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        ).strip()

        return {
            "type": "text",
            "content": clean_text,
            "raw": raw_text,
        }

    # ==================================================================
    # TOOL-CALL PARSER
    # ==================================================================

    @staticmethod
    def _parse_tool_calls(
        text: str,
    ) -> list[dict[str, Any]]:
        """
        Parse Qwen tool-call output.

        Supported format:

            <tool_call>
            {"name": "geochat", "arguments": {...}}
            </tool_call>

        Also supports:
            - raw JSON objects
            - raw JSON arrays
            - fenced JSON blocks
        """

        import re

        candidates: list[str] = []

        stripped = text.strip()

        # ==============================================================
        # Qwen XML-style tool calls
        # ==============================================================

        matches = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            text,
            flags=re.DOTALL,
        )

        for match in matches:
            candidate = match.strip()

            if candidate:
                candidates.append(candidate)

        # ==============================================================
        # Entire response is a JSON object
        # ==============================================================

        if (
            stripped.startswith("{")
            and stripped.endswith("}")
        ):
            candidates.append(stripped)

        # ==============================================================
        # Entire response is a JSON array
        # ==============================================================

        if (
            stripped.startswith("[")
            and stripped.endswith("]")
        ):
            candidates.append(stripped)

        # ==============================================================
        # Fenced JSON
        # ==============================================================

        if "```json" in text:

            blocks = text.split("```json")[1:]

            for block in blocks:

                candidate = block.split(
                    "```",
                    1,
                )[0].strip()

                if candidate:
                    candidates.append(candidate)

        # ==============================================================
        # Parse candidates
        # ==============================================================

        for candidate in candidates:

            try:
                parsed = json.loads(candidate)

            except json.JSONDecodeError:
                continue

            calls = QwenEngine._normalize_tool_calls(
                parsed
            )

            if calls:
                return calls

        return []

    @staticmethod
    def _normalize_tool_calls(
        value: Any,
    ) -> list[dict[str, Any]]:
        """
        Normalize several JSON tool-call layouts.
        """

        if isinstance(
            value,
            dict,
        ):

            # ------------------------------------------
            # {"tool_calls": [...]}
            # ------------------------------------------

            if "tool_calls" in value:

                return QwenEngine._normalize_tool_calls(
                    value["tool_calls"]
                )

            # ------------------------------------------
            # {"name": "...", "arguments": {...}}
            # ------------------------------------------

            if (
                isinstance(
                    value.get("name"),
                    str,
                )
                and "arguments" in value
            ):

                arguments = value[
                    "arguments"
                ]

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
                        "name": value["name"],
                        "arguments": arguments,
                    }
                ]

            # ------------------------------------------
            # {"function": {...}}
            # ------------------------------------------

            if isinstance(
                value.get("function"),
                dict,
            ):

                function = value[
                    "function"
                ]

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

        # ------------------------------------------------------------------
        # List of calls.
        # ------------------------------------------------------------------

        if isinstance(
            value,
            list,
        ):

            results: list[
                dict[str, Any]
            ] = []

            for item in value:

                normalized = (
                    QwenEngine._normalize_tool_calls(
                        item
                    )
                )

                results.extend(
                    normalized
                )

            return results

        return []

    # ==================================================================
    # CONVENIENCE HELPER
    # ==================================================================

    def ask(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **generation_kwargs: Any,
    ) -> str:
        """
        Convenience wrapper for a single user prompt.
        """

        if not isinstance(
            prompt,
            str,
        ):
            raise TypeError(
                "prompt must be a string."
            )

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "prompt must not be empty."
            )

        messages: list[
            dict[str, Any]
        ] = []

        if system_prompt is not None:

            if not isinstance(
                system_prompt,
                str,
            ):
                raise TypeError(
                    "system_prompt must be a string."
                )

            system_prompt = (
                system_prompt.strip()
            )

            if system_prompt:

                messages.append(
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return self.chat(
            messages,
            **generation_kwargs,
        )

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================

    def info(self) -> dict[str, Any]:
        """
        Return basic runtime information.
        """

        allocated = 0.0
        reserved = 0.0

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            reserved = (
                torch.cuda.memory_reserved()
                / 1024**3
            )

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