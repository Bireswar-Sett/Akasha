from huggingface_hub import hf_hub_download, list_repo_files
import json


MODEL_ID = "MBZUAI/geochat-7B"


def print_repo_files():
    print("=" * 80)
    print("HUGGING FACE REPOSITORY FILES")
    print("=" * 80)

    for filename in list_repo_files(MODEL_ID):
        print(filename)


def load_raw_config():
    config_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="config.json",
    )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_config(config):
    print("\n" + "=" * 80)
    print("RAW GEOCHAT CONFIGURATION")
    print("=" * 80)

    print(json.dumps(config, indent=4))


def main():
    print(f"Inspecting: {MODEL_ID}\n")

    print_repo_files()

    config = load_raw_config()
    print_config(config)


if __name__ == "__main__":
    main()