"""
Configuration module for Machine Translation Orchestrator.

All service URLs are loaded from environment variables with sensible defaults.
"""

import os

# Service URLs
LID_SERVICE_URL = os.getenv("LID_SERVICE_URL", "http://lid-fasttext:8080")
NLLB_SERVICE_URL = os.getenv("NLLB_SERVICE_URL", "http://nllb-200-3b:8080")
QWEN_MT_SERVICE_URL = os.getenv("QWEN_MT_SERVICE_URL", "http://qwen-mt-7b:8080")
LLAMA_SERVICE_URL = os.getenv("LLAMA_SERVICE_URL", "http://llama-3-8b-instruct:8080")
MISTRAL_SERVICE_URL = os.getenv("MISTRAL_SERVICE_URL", "http://mistral-nemo-12b:8080")
GEMMA_SERVICE_URL = os.getenv("GEMMA_SERVICE_URL", "http://gemma-x2-28-9b:8080")
QWEN_FAST_SERVICE_URL = os.getenv("QWEN_FAST_SERVICE_URL", "http://qwen2-1-5b-instruct:8080")
NLLB_DISTILLED_SERVICE_URL = os.getenv("NLLB_DISTILLED_SERVICE_URL", "http://nllb-200-distilled-1b:8080")

# Default target language (English)
DEFAULT_TARGET_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")
