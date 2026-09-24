"""Fetch only the pinned model snapshot during image build."""

from huggingface_hub import snapshot_download

from token_trail_backend.config import MODEL_REPO, MODEL_REVISION

snapshot_download(
    repo_id=MODEL_REPO,
    revision=MODEL_REVISION,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
)
