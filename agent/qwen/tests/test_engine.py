from __future__ import annotations

from models.qwen.controller.model import QwenEngine


def main() -> None:
    print("=" * 80)
    print("QWEN ENGINE TEST")
    print("=" * 80)

    qwen = QwenEngine()

    print("\nEngine info:")
    print(
        qwen.info()
    )

    response = qwen.ask(
        "What is 2 + 2?",
        system_prompt=(
            "You are a helpful AI assistant."
        ),
        max_new_tokens=16,
        do_sample=False,
    )

    print("\nQwen response:")
    print(
        response
    )

    if not response:
        raise RuntimeError(
            "Qwen returned an empty response."
        )

    print("\n" + "=" * 80)
    print("✓ QWEN ENGINE TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()