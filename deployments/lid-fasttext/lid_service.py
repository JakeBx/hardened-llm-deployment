"""
Language Identification Service using FastText

This service provides fast, accurate language identification using:
- Primary: facebook/fasttext-language-identification (200+ languages)
- Fallback: cis-lmu/glotlid (better for low-resource languages)

Endpoints:
- POST /v1/identify: Identify language of input text
- POST /v1/identify/batch: Batch language identification
- GET /v1/languages: List supported languages

Environment Variables:
- MODEL_PATH: Path to the FastText model file
- USE_GLOTLID: Set to "true" to use GlotLID instead of FastText
- CACHE_DIR: Directory for model caching
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import fasttext
import uvicorn
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

# Suppress FastText warnings about model loading
fasttext.FastText.eprint = lambda x: None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    """Service configuration."""
    MODEL_PATH = os.getenv("MODEL_PATH", "")
    USE_GLOTLID = os.getenv("USE_GLOTLID", "false").lower() == "true"
    CACHE_DIR = os.getenv("CACHE_DIR", "/app/models")
    
    # Model identifiers
    FASTTEXT_REPO = "facebook/fasttext-language-identification"
    FASTTEXT_FILE = "model.bin"
    
    GLOTLID_REPO = "cis-lmu/glotlid"
    GLOTLID_FILE = "model.bin"


# ==============================================================================
# Data Models
# ==============================================================================

class IdentifyRequest(BaseModel):
    """Language identification request."""
    text: str = Field(..., description="Text to identify language for", min_length=1)
    top_k: int = Field(1, description="Number of top predictions to return", ge=1, le=10)


class IdentifyResponse(BaseModel):
    """Language identification response."""
    language: str = Field(..., description="ISO language code")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    script: Optional[str] = Field(None, description="Detected script")
    alternatives: Optional[List[Dict[str, float]]] = Field(
        None, 
        description="Alternative language predictions with scores"
    )


class BatchIdentifyRequest(BaseModel):
    """Batch language identification request."""
    texts: List[str] = Field(..., description="List of texts to identify", min_items=1)
    top_k: int = Field(1, description="Number of top predictions per text", ge=1, le=10)


class BatchIdentifyResponse(BaseModel):
    """Batch language identification response."""
    results: List[IdentifyResponse]


# ==============================================================================
# FastText Language Identifier
# ==============================================================================

class FastTextLID:
    """
    FastText-based Language Identification.
    
    Uses facebook/fasttext-language-identification model which supports
    200+ languages with high accuracy and very fast inference.
    """
    
    # Language code normalization (ISO 639-3 to ISO 639-1 where applicable)
    LANGUAGE_MAPPING = {
        "__label__eng_Latn": "en",
        "__label__zho_Hans": "zh",
        "__label__zho_Hant": "zh-tw",
        "__label__rus_Cyrl": "ru",
        "__label__spa_Latn": "es",
        "__label__fra_Latn": "fr",
        "__label__deu_Latn": "de",
        "__label__ara_Arab": "ar",
        "__label__hin_Deva": "hi",
        "__label__ben_Beng": "bn",
        "__label__por_Latn": "pt",
        "__label__jpn_Jpan": "ja",
        "__label__kor_Hang": "ko",
        "__label__ita_Latn": "it",
        "__label__nld_Latn": "nl",
        "__label__pol_Latn": "pl",
        "__label__tur_Latn": "tr",
        "__label__vie_Latn": "vi",
        "__label__tha_Thai": "th",
        "__label__ukr_Cyrl": "uk",
        "__label__ces_Latn": "cs",
        "__label__ell_Grek": "el",
        "__label__heb_Hebr": "he",
        "__label__ron_Latn": "ro",
        "__label__hun_Latn": "hu",
        "__label__swe_Latn": "sv",
        "__label__dan_Latn": "da",
        "__label__fin_Latn": "fi",
        "__label__nor_Latn": "no",
        "__label__ind_Latn": "id",
        "__label__msa_Latn": "ms",
        "__label__fil_Latn": "tl",
        "__label__tam_Taml": "ta",
        "__label__tel_Telu": "te",
        "__label__mar_Deva": "mr",
        "__label__guj_Gujr": "gu",
        "__label__kan_Knda": "kn",
        "__label__mal_Mlym": "ml",
        "__label__pan_Guru": "pa",
        "__label__urd_Arab": "ur",
        "__label__fas_Arab": "fa",
        "__label__bul_Cyrl": "bg",
        "__label__hrv_Latn": "hr",
        "__label__slk_Latn": "sk",
        "__label__slv_Latn": "sl",
        "__label__srp_Cyrl": "sr",
        "__label__cat_Latn": "ca",
        "__label__eus_Latn": "eu",
        "__label__glg_Latn": "gl",
    }
    
    # Script extraction patterns
    SCRIPT_SUFFIXES = {
        "Latn": "latin",
        "Cyrl": "cyrillic",
        "Arab": "arabic",
        "Hans": "simplified_chinese",
        "Hant": "traditional_chinese",
        "Deva": "devanagari",
        "Beng": "bengali",
        "Jpan": "japanese",
        "Hang": "hangul",
        "Thai": "thai",
        "Grek": "greek",
        "Hebr": "hebrew",
        "Taml": "tamil",
        "Telu": "telugu",
        "Gujr": "gujarati",
        "Knda": "kannada",
        "Mlym": "malayalam",
        "Guru": "gurmukhi",
    }

    def __init__(self, model_path: str):
        """Initialize the FastText LID model."""
        self.model = fasttext.load_model(model_path)
        logger.info(f"Loaded FastText model from {model_path}")

    def identify(self, text: str, top_k: int = 1) -> IdentifyResponse:
        """
        Identify the language of the input text.
        
        Args:
            text: Input text to identify
            top_k: Number of top predictions to return
            
        Returns:
            IdentifyResponse with language code, confidence, and alternatives
        """
        # Clean text for prediction (single line, remove excessive whitespace)
        cleaned_text = self._clean_text(text)
        
        # Get predictions
        labels, scores = self.model.predict(cleaned_text, k=top_k)
        
        # Parse primary prediction
        primary_label = labels[0]
        primary_score = float(scores[0])
        
        # Normalize language code
        language = self._normalize_language(primary_label)
        script = self._extract_script(primary_label)
        
        # Build alternatives list
        alternatives = None
        if top_k > 1 and len(labels) > 1:
            alternatives = [
                {
                    "language": self._normalize_language(label),
                    "confidence": float(score)
                }
                for label, score in zip(labels[1:], scores[1:])
            ]
        
        return IdentifyResponse(
            language=language,
            confidence=primary_score,
            script=script,
            alternatives=alternatives
        )

    def identify_batch(
        self, 
        texts: List[str], 
        top_k: int = 1
    ) -> List[IdentifyResponse]:
        """Identify languages for a batch of texts."""
        return [self.identify(text, top_k) for text in texts]

    def _clean_text(self, text: str) -> str:
        """Clean text for prediction."""
        # Replace newlines with spaces
        text = text.replace("\n", " ").replace("\r", " ")
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        # Trim
        text = text.strip()
        # Truncate if too long (FastText works well with shorter text)
        if len(text) > 1000:
            text = text[:1000]
        return text

    def _normalize_language(self, label: str) -> str:
        """Normalize FastText label to ISO language code."""
        # Check direct mapping
        if label in self.LANGUAGE_MAPPING:
            return self.LANGUAGE_MAPPING[label]
        
        # Extract language code from label format: __label__xxx_Yyyy
        match = re.match(r"__label__([a-z]{3})_", label)
        if match:
            # Return ISO 639-3 code
            return match.group(1)
        
        # Fallback: strip label prefix
        return label.replace("__label__", "")[:3]

    def _extract_script(self, label: str) -> Optional[str]:
        """Extract script name from FastText label."""
        for suffix, script_name in self.SCRIPT_SUFFIXES.items():
            if label.endswith(suffix):
                return script_name
        return None

    @staticmethod
    def get_supported_languages() -> List[Dict[str, str]]:
        """Return list of supported languages."""
        # This is a subset of the 200+ supported languages
        return [
            {"code": "en", "name": "English", "script": "latin"},
            {"code": "zh", "name": "Chinese (Simplified)", "script": "simplified_chinese"},
            {"code": "zh-tw", "name": "Chinese (Traditional)", "script": "traditional_chinese"},
            {"code": "es", "name": "Spanish", "script": "latin"},
            {"code": "fr", "name": "French", "script": "latin"},
            {"code": "de", "name": "German", "script": "latin"},
            {"code": "ru", "name": "Russian", "script": "cyrillic"},
            {"code": "ar", "name": "Arabic", "script": "arabic"},
            {"code": "hi", "name": "Hindi", "script": "devanagari"},
            {"code": "bn", "name": "Bengali", "script": "bengali"},
            {"code": "pt", "name": "Portuguese", "script": "latin"},
            {"code": "ja", "name": "Japanese", "script": "japanese"},
            {"code": "ko", "name": "Korean", "script": "hangul"},
            {"code": "it", "name": "Italian", "script": "latin"},
            {"code": "nl", "name": "Dutch", "script": "latin"},
            {"code": "pl", "name": "Polish", "script": "latin"},
            {"code": "tr", "name": "Turkish", "script": "latin"},
            {"code": "vi", "name": "Vietnamese", "script": "latin"},
            {"code": "th", "name": "Thai", "script": "thai"},
            {"code": "uk", "name": "Ukrainian", "script": "cyrillic"},
            {"code": "el", "name": "Greek", "script": "greek"},
            {"code": "he", "name": "Hebrew", "script": "hebrew"},
            {"code": "ta", "name": "Tamil", "script": "tamil"},
            {"code": "te", "name": "Telugu", "script": "telugu"},
            {"code": "ur", "name": "Urdu", "script": "arabic"},
            {"code": "fa", "name": "Persian", "script": "arabic"},
        ]


# ==============================================================================
# FastAPI Application
# ==============================================================================

app = FastAPI(
    title="Language Identification Service",
    description="Fast language identification using FastText (200+ languages)",
    version="1.0.0"
)

# Global model instance
lid_model: Optional[FastTextLID] = None


def download_model() -> str:
    """Download the FastText model from HuggingFace Hub."""
    cache_dir = Path(Config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if Config.USE_GLOTLID:
        repo = Config.GLOTLID_REPO
        filename = Config.GLOTLID_FILE
        logger.info(f"Using GlotLID model from {repo}")
    else:
        repo = Config.FASTTEXT_REPO
        filename = Config.FASTTEXT_FILE
        logger.info(f"Using FastText model from {repo}")
    
    model_path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        cache_dir=str(cache_dir),
        local_dir=str(cache_dir),
    )
    
    return model_path


@app.on_event("startup")
async def startup():
    """Initialize the LID model on startup."""
    global lid_model
    
    try:
        # Use provided model path or download from HuggingFace
        if Config.MODEL_PATH and Path(Config.MODEL_PATH).exists():
            model_path = Config.MODEL_PATH
        else:
            model_path = download_model()
        
        lid_model = FastTextLID(model_path)
        logger.info("Language Identification Service started successfully")
    except Exception as e:
        logger.error(f"Failed to load LID model: {e}")
        raise


@app.get("/health")
async def health():
    """Health check endpoint."""
    if lid_model is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    if lid_model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}


@app.post("/v1/identify", response_model=IdentifyResponse)
async def identify(request: IdentifyRequest):
    """
    Identify the language of the input text.
    
    Returns the most likely language with confidence score,
    and optionally alternative predictions.
    """
    if lid_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    return lid_model.identify(request.text, request.top_k)


@app.post("/v1/identify/batch", response_model=BatchIdentifyResponse)
async def identify_batch(request: BatchIdentifyRequest):
    """
    Identify languages for a batch of texts.
    
    More efficient than making multiple single requests.
    """
    if lid_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = lid_model.identify_batch(request.texts, request.top_k)
    return BatchIdentifyResponse(results=results)


@app.get("/v1/languages")
async def list_languages():
    """List supported languages."""
    return {
        "languages": FastTextLID.get_supported_languages(),
        "total_supported": "200+",
        "model": "GlotLID" if Config.USE_GLOTLID else "FastText"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
