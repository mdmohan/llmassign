"""Single source of truth for the Hugging Face cache directory."""

import os
from pathlib import Path


DEFAULT_CACHE_DIR = "/home/jovyan/llmgenai/hf_cache"
CACHE_DIR = Path(
    os.environ.get("CACHE_DIR", DEFAULT_CACHE_DIR)
).expanduser().resolve()

# Keep Hugging Face components that consult HF_HOME aligned with the explicit
# cache_dir passed by this project.
os.environ["HF_HOME"] = str(CACHE_DIR)
