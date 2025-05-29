import os


def normalize_text(text: str) -> str:
    return (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("–", "-")
            .replace("—", "--")
            .replace("…", "...")
    )

def normalize_and_overwrite_all_txt_files(folder: str):
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            normalized = normalize_text(content)
            with open(path, "w", encoding="utf-8") as f:
                f.write(normalized)
            print(f"Normalized and saved: {filename}")

# Example usage
normalize_and_overwrite_all_txt_files("data/train")
normalize_and_overwrite_all_txt_files("data/proxy_test")
