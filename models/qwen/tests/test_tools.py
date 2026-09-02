from __future__ import annotations

from models.qwen.controller.model import QwenEngine
from models.qwen.controller.prompts import SYSTEM_PROMPT
from models.qwen.controller.tools import get_tool


def main() -> None:
    print("=" * 80)
    print("QWEN TOOL-CALLING TEST")
    print("=" * 80)

    qwen = QwenEngine()

    geochat_tool = get_tool(
        "geochat"
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "I have one satellite image at "
                "/data/test.png. "
                "Describe what is visible in it."
            ),
        },
    ]

    print("\nAsking Qwen to choose a tool...")

    result = qwen.chat_with_tools(
        messages=messages,
        tools=[
            geochat_tool,
        ],
        max_new_tokens=128,
        do_sample=False,
    )

    print("\nResult type:")
    print(
        result["type"]
    )

    print("\nResult:")
    print(
        result
    )

    if result["type"] == "tool_calls":

        print("\n✓ Qwen produced a tool call.")

        for call in result["tool_calls"]:

            print(
                f"\nTool: {call['name']}"
            )

            print(
                f"Arguments: {call['arguments']}"
            )

    elif result["type"] == "text":

        print(
            "\nQwen returned text instead of a tool call:"
        )

        print(
            result["content"]
        )

    else:

        raise RuntimeError(
            "Unexpected Qwen result type."
        )

    print("\n" + "=" * 80)
    print("✓ QWEN TOOL INTERFACE TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()