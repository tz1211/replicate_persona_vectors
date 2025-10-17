import fire
from config import setup_credentials
from huggingface_hub import snapshot_download

config = setup_credentials()

# Download model from Hugging Face
def main(model_name: str="Qwen/Qwen3-1.7B", local_dir: str="models/Qwen/Qwen3-1.7B"):
    model_path = snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        token=config.hf_token
    )

if __name__ == "__main__":
    fire.Fire(main)