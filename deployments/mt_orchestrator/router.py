"""
Model Routing module.

Routes translation requests to the optimal model based on language and text characteristics.
"""

from typing import Dict, Tuple

from .config import (
    GEMMA_SERVICE_URL,
    LLAMA_SERVICE_URL,
    MISTRAL_SERVICE_URL,
    NLLB_DISTILLED_SERVICE_URL,
    NLLB_SERVICE_URL,
    QWEN_FAST_SERVICE_URL,
    QWEN_MT_SERVICE_URL,
)
from .models import TextCategory


# Language to model mapping - module-level dicts
LANGUAGE_MODEL_MAP: Dict[str, str] = {
    "zh": "qwen_mt",
    "zh-cn": "qwen_mt",
    "zh-tw": "qwen_mt",
    "ja": "qwen_mt",  # Japanese also benefits from Qwen
    "ko": "qwen_mt",  # Korean also benefits from Qwen
    "ru": "nllb",
    "uk": "nllb",
    "bg": "nllb",
    "es": "llama",
    "fr": "mistral",
    "ar": "gemma",
    "hi": "nllb",
    "bn": "nllb",
    "ta": "nllb",
    "te": "nllb",
}

# Category to model mapping (when language doesn't dictate)
CATEGORY_MODEL_MAP: Dict[TextCategory, str] = {
    TextCategory.SHORT_INFORMAL: "qwen_fast",
    TextCategory.LONG_FORMAL: "mistral",
    TextCategory.LITERARY_NUANCED: "llama",
    TextCategory.TECHNICAL: "nllb_distilled",
}

# Model to URL mapping
MODEL_URL_MAP: Dict[str, str] = {
    "qwen_mt": QWEN_MT_SERVICE_URL,
    "nllb": NLLB_SERVICE_URL,
    "llama": LLAMA_SERVICE_URL,
    "mistral": MISTRAL_SERVICE_URL,
    "gemma": GEMMA_SERVICE_URL,
    "qwen_fast": QWEN_FAST_SERVICE_URL,
    "nllb_distilled": NLLB_DISTILLED_SERVICE_URL,
}

# Model to served name mapping
MODEL_NAME_MAP: Dict[str, str] = {
    "qwen_mt": "qwen-mt-7b",
    "nllb": "nllb-200-3b",
    "llama": "llama-3-8b-instruct",
    "mistral": "mistral-nemo-12b",
    "gemma": "gemma-x2-28-9b",
    "qwen_fast": "qwen2-1-5b-instruct",
    "nllb_distilled": "nllb-200-distilled-1b",
}


def select_model(
    source_lang: str,
    target_lang: str,
    text_category: TextCategory,
    word_count: int
) -> Tuple[str, str, str]:
    """
    Select the optimal model for translation.
    
    Returns: (model_key, model_url, model_name)
    
    Model Selection Matrix:
    
    | Source Lang  | Category        | Model                          |
    |--------------|-----------------|--------------------------------|
    | Chinese      | Any             | Qwen/Qwen-MT-7B               |
    | Russian      | Any             | NLLB-200-3.3B                 |
    | Spanish      | Any             | Llama-3.1-8B-Instruct         |
    | French       | Any             | Mistral-Nemo-12B              |
    | Arabic       | Any             | Gemma-X2-28-9B                |
    | Hindi/Bengali| Any             | NLLB-200-3.3B                 |
    | Any          | short_informal  | Qwen2-1.5B-Instruct (fast)    |
    | Any          | technical       | NLLB-200-distilled-1.3B       |
    | Any          | long_formal     | Mistral-Nemo-12B/Llama-3.1-8B |
    | Any          | literary_nuanced| Llama-3.1-8B-Instruct         |
    | Default      | Default         | NLLB-200-3.3B (200+ langs)    |
    """
    # Priority 1: Language-specific routing
    if source_lang.lower() in LANGUAGE_MODEL_MAP:
        model_key = LANGUAGE_MODEL_MAP[source_lang.lower()]
        return (
            model_key,
            MODEL_URL_MAP[model_key],
            MODEL_NAME_MAP[model_key]
        )
    
    # Priority 2: For very short texts, use fast model regardless
    if word_count < 50 and text_category == TextCategory.SHORT_INFORMAL:
        return (
            "qwen_fast",
            MODEL_URL_MAP["qwen_fast"],
            MODEL_NAME_MAP["qwen_fast"]
        )
    
    # Priority 3: Category-based routing
    if text_category in CATEGORY_MODEL_MAP:
        model_key = CATEGORY_MODEL_MAP[text_category]
        return (
            model_key,
            MODEL_URL_MAP[model_key],
            MODEL_NAME_MAP[model_key]
        )
    
    # Default: NLLB for broad language support
    return (
        "nllb",
        MODEL_URL_MAP["nllb"],
        MODEL_NAME_MAP["nllb"]
    )
