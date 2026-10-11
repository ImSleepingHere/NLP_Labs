"""Download the models/data the labs load at runtime.

Run after `pip install -r requirements.txt`:
    python setup_resources.py
"""
import os
import subprocess
import sys

import nltk

# NLTK data (Labs 2-5): tokenizers, stopwords, WordNet lemmatizer
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)
    print(f"nltk: {pkg} ok")

# spaCy English model (Lab 2)
try:
    import en_core_web_sm  # noqa: F401
    print("spacy: en_core_web_sm already installed")
except ImportError:
    # `spacy download` fetches from github.com; if that host is blocked, download
    # en_core_web_sm-3.8.0-py3-none-any.whl via api.github.com and pip install it.
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Hugging Face models (Lab 7) - cached under ~/.cache/huggingface
# Xet transfers 404 on legacy repo names like "distilgpt2", so use plain HTTP.
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download

for repo in [
    "distilgpt2",
    "google/flan-t5-small",
    "distilbert-base-uncased-finetuned-sst-2-english",
]:
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
    print(f"hf: {repo} ok")
