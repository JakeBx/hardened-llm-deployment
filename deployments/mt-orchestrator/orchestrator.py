"""
Machine Translation Orchestrator for KServe InferenceGraph

This orchestrator implements a four-stage pipeline:
1. Language Identification (LID) - Detects source language
2. Text Characterization - Analyzes text type (short/informal, long/formal, technical)
3. Pre-processing - De-noising, sentence splitting, glossary injection
4. Model Selection - Routes to optimal translation model

Architecture:
Detection → Characterization → Pre-processing → Model Selection → Translation

Environment Variables:
- LID_SERVICE_URL: URL for language identification service
- NLLB_SERVICE_URL: URL for NLLB-200-3.3B model
- QWEN_MT_SERVICE_URL: URL for Qwen-MT-7B model (Chinese)
- LLAMA_SERVICE_URL: URL for Llama-3.1-8B-Instruct (Spanish/French)
- MISTRAL_SERVICE_URL: URL for Mistral-Nemo-12B (long documents)
- GEMMA_SERVICE_URL: URL for Gemma-X2-28-9B (Arabic)
- QWEN_FAST_SERVICE_URL: URL for Qwen2-1.5B-Instruct (fast/informal)
- NLLB_DISTILLED_SERVICE_URL: URL for NLLB-200-distilled-1.3B (technical)
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
import pysbd
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

class ServiceConfig:
    """Central configuration for all service URLs."""
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


# ==============================================================================
# Data Models
# ==============================================================================

class TextCategory(str, Enum):
    """Text characterization categories."""
    SHORT_INFORMAL = "short_informal"    # Chat, tweets, social media
    LONG_FORMAL = "long_formal"          # Legal, medical, technical manuals
    LITERARY_NUANCED = "literary_nuanced" # Marketing, creative writing
    TECHNICAL = "technical"              # Code, API docs, specifications


class LanguageFamily(str, Enum):
    """Language family groupings for model routing."""
    CHINESE = "chinese"        # zh, zh-cn, zh-tw
    SLAVIC = "slavic"          # ru, uk, bg, etc.
    ROMANCE = "romance"        # es, fr, pt, it, ro
    ARABIC = "arabic"          # ar, ar-eg, etc.
    INDIC = "indic"           # hi, bn, ta, te, etc.
    GERMANIC = "germanic"     # de, nl, sv, etc.
    OTHER = "other"           # Everything else


@dataclass
class LIDResult:
    """Language identification result."""
    language_code: str  # ISO 639-1 or ISO 639-3
    confidence: float   # 0.0 to 1.0
    script: Optional[str] = None  # Detected script (Latin, Cyrillic, etc.)


@dataclass
class CharacterizationResult:
    """Text characterization result."""
    category: TextCategory
    word_count: int
    avg_sentence_length: float
    formality_score: float  # 0.0 (informal) to 1.0 (formal)
    technical_density: float  # Ratio of technical terms


@dataclass
class PreprocessedText:
    """Pre-processed text ready for translation."""
    original_text: str
    cleaned_text: str
    sentences: List[str]
    detected_entities: List[str]  # Named entities to preserve
    glossary_terms: Dict[str, str]  # Term -> Translation mapping


class TranslationRequest(BaseModel):
    """Incoming translation request."""
    text: str = Field(..., description="Text to translate")
    source_lang: Optional[str] = Field(None, description="Source language code (auto-detected if not provided)")
    target_lang: str = Field("en", description="Target language code")
    domain: Optional[str] = Field(None, description="Domain hint (legal, medical, technical, etc.)")
    glossary: Optional[Dict[str, str]] = Field(None, description="Custom glossary terms")
    stream: bool = Field(False, description="Whether to stream the response")


class TranslationResponse(BaseModel):
    """Translation response."""
    translated_text: str
    source_lang: str
    source_lang_confidence: float
    target_lang: str
    model_used: str
    text_category: str
    preprocessing_applied: List[str]
    latency_ms: float


# ==============================================================================
# Language Detection
# ==============================================================================

class LanguageIdentifier:
    """
    Language identification using FastText or GlotLID.
    
    Primary: facebook/fasttext-language-identification (200+ languages)
    Fallback: cis-lmu/glotlid (better for low-resource languages)
    """
    
    # Language code normalization
    LANGUAGE_ALIASES = {
        "zho": "zh", "cmn": "zh", "yue": "zh",
        "rus": "ru",
        "spa": "es",
        "fra": "fr",
        "ara": "ar",
        "hin": "hi",
        "ben": "bn",
        "deu": "de",
        "jpn": "ja",
        "kor": "ko",
        "por": "pt",
        "ita": "it",
    }
    
    # Script detection patterns
    SCRIPT_PATTERNS = {
        "arabic": re.compile(r'[\u0600-\u06FF\u0750-\u077F]'),
        "cyrillic": re.compile(r'[\u0400-\u04FF]'),
        "cjk": re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF]'),
        "devanagari": re.compile(r'[\u0900-\u097F]'),
        "bengali": re.compile(r'[\u0980-\u09FF]'),
        "thai": re.compile(r'[\u0E00-\u0E7F]'),
        "korean": re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF]'),
        "japanese": re.compile(r'[\u3040-\u309F\u30A0-\u30FF]'),
    }

    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.lid_url = ServiceConfig.LID_SERVICE_URL

    async def identify(self, text: str) -> LIDResult:
        """
        Identify the source language of the text.
        
        Falls back to script-based detection if LID service is unavailable.
        """
        try:
            response = await self.http_client.post(
                f"{self.lid_url}/v1/identify",
                json={"text": text[:1000]},  # Truncate for efficiency
                timeout=5.0
            )
            response.raise_for_status()
            result = response.json()
            
            lang_code = result.get("language", "und")
            lang_code = self.LANGUAGE_ALIASES.get(lang_code, lang_code)
            
            return LIDResult(
                language_code=lang_code,
                confidence=result.get("confidence", 0.0),
                script=result.get("script")
            )
        except Exception as e:
            logger.warning(f"LID service call failed: {e}, using script-based fallback")
            return self._script_based_detection(text)
    
    def _script_based_detection(self, text: str) -> LIDResult:
        """Fallback script-based language detection."""
        for script_name, pattern in self.SCRIPT_PATTERNS.items():
            if pattern.search(text):
                # Map script to most common language
                script_lang_map = {
                    "arabic": "ar",
                    "cyrillic": "ru",
                    "cjk": "zh",
                    "devanagari": "hi",
                    "bengali": "bn",
                    "thai": "th",
                    "korean": "ko",
                    "japanese": "ja",
                }
                return LIDResult(
                    language_code=script_lang_map.get(script_name, "und"),
                    confidence=0.6,  # Lower confidence for fallback
                    script=script_name
                )
        
        # Default to undetermined with Latin script assumption
        return LIDResult(
            language_code="und",
            confidence=0.0,
            script="latin"
        )

    @staticmethod
    def get_language_family(lang_code: str) -> LanguageFamily:
        """Determine the language family for routing decisions."""
        family_map = {
            "zh": LanguageFamily.CHINESE,
            "zh-cn": LanguageFamily.CHINESE,
            "zh-tw": LanguageFamily.CHINESE,
            "ru": LanguageFamily.SLAVIC,
            "uk": LanguageFamily.SLAVIC,
            "bg": LanguageFamily.SLAVIC,
            "es": LanguageFamily.ROMANCE,
            "fr": LanguageFamily.ROMANCE,
            "pt": LanguageFamily.ROMANCE,
            "it": LanguageFamily.ROMANCE,
            "ro": LanguageFamily.ROMANCE,
            "ar": LanguageFamily.ARABIC,
            "hi": LanguageFamily.INDIC,
            "bn": LanguageFamily.INDIC,
            "ta": LanguageFamily.INDIC,
            "te": LanguageFamily.INDIC,
            "de": LanguageFamily.GERMANIC,
            "nl": LanguageFamily.GERMANIC,
            "sv": LanguageFamily.GERMANIC,
        }
        return family_map.get(lang_code.lower(), LanguageFamily.OTHER)


# ==============================================================================
# Text Characterization
# ==============================================================================

class TextCharacterizer:
    """
    Analyzes text to determine its category for optimal model selection.
    
    Categories:
    - short_informal: Real-time chat, tweets, social media (< 100 words, informal)
    - long_formal: Legal documents, medical papers (> 500 words, formal)
    - literary_nuanced: Marketing copy, creative writing (variable, nuanced)
    - technical: Code, API docs, specifications (high technical term density)
    """
    
    # Informal indicators
    INFORMAL_PATTERNS = [
        re.compile(r'\b(lol|omg|btw|idk|tbh|ngl|brb|afk)\b', re.IGNORECASE),
        re.compile(r'[!?]{2,}'),  # Multiple punctuation
        re.compile(r'[\U0001F600-\U0001F64F]'),  # Emojis
        re.compile(r'@\w+'),  # Mentions
        re.compile(r'#\w+'),  # Hashtags
    ]
    
    # Technical indicators
    TECHNICAL_PATTERNS = [
        re.compile(r'\b(API|SDK|HTTP|JSON|XML|REST|SQL|HTML|CSS)\b'),
        re.compile(r'\b(function|class|method|variable|parameter|return)\b', re.IGNORECASE),
        re.compile(r'[\w]+\([\w,\s]*\)'),  # Function calls
        re.compile(r'`[^`]+`'),  # Code snippets
        re.compile(r'\b\d+\.\d+\.\d+\b'),  # Version numbers
    ]
    
    # Formal indicators
    FORMAL_PATTERNS = [
        re.compile(r'\b(hereby|whereas|pursuant|notwithstanding|heretofore)\b', re.IGNORECASE),
        re.compile(r'\b(shall|must|may not|is required|in accordance with)\b', re.IGNORECASE),
        re.compile(r'\b(diagnosis|prognosis|treatment|contraindicated)\b', re.IGNORECASE),
    ]

    def __init__(self):
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

    def characterize(self, text: str) -> CharacterizationResult:
        """Analyze text and return characterization result."""
        sentences = self.segmenter.segment(text)
        words = text.split()
        word_count = len(words)
        avg_sentence_length = word_count / len(sentences) if sentences else 0
        
        # Calculate scores
        informal_score = self._calculate_informal_score(text)
        formal_score = self._calculate_formal_score(text)
        technical_score = self._calculate_technical_score(text)
        
        # Determine category
        category = self._determine_category(
            word_count=word_count,
            informal_score=informal_score,
            formal_score=formal_score,
            technical_score=technical_score,
            avg_sentence_length=avg_sentence_length
        )
        
        # Calculate formality as formal_score - informal_score, normalized to 0-1
        formality = min(1.0, max(0.0, 0.5 + (formal_score - informal_score)))
        
        return CharacterizationResult(
            category=category,
            word_count=word_count,
            avg_sentence_length=avg_sentence_length,
            formality_score=formality,
            technical_density=technical_score
        )
    
    def _calculate_informal_score(self, text: str) -> float:
        """Calculate informal language score (0.0 to 1.0)."""
        matches = sum(
            len(pattern.findall(text)) 
            for pattern in self.INFORMAL_PATTERNS
        )
        # Normalize by text length
        return min(1.0, matches / max(1, len(text.split()) / 10))
    
    def _calculate_formal_score(self, text: str) -> float:
        """Calculate formal language score (0.0 to 1.0)."""
        matches = sum(
            len(pattern.findall(text)) 
            for pattern in self.FORMAL_PATTERNS
        )
        return min(1.0, matches / max(1, len(text.split()) / 20))
    
    def _calculate_technical_score(self, text: str) -> float:
        """Calculate technical language score (0.0 to 1.0)."""
        matches = sum(
            len(pattern.findall(text)) 
            for pattern in self.TECHNICAL_PATTERNS
        )
        return min(1.0, matches / max(1, len(text.split()) / 10))
    
    def _determine_category(
        self,
        word_count: int,
        informal_score: float,
        formal_score: float,
        technical_score: float,
        avg_sentence_length: float
    ) -> TextCategory:
        """Determine the text category based on all factors."""
        # Technical takes priority if score is high
        if technical_score > 0.3:
            return TextCategory.TECHNICAL
        
        # Short informal: < 100 words, high informal score
        if word_count < 100 and informal_score > 0.2:
            return TextCategory.SHORT_INFORMAL
        
        # Long formal: > 500 words, high formal score, long sentences
        if word_count > 500 and formal_score > 0.2 and avg_sentence_length > 15:
            return TextCategory.LONG_FORMAL
        
        # Literary/nuanced: moderate length, balanced scores
        if 100 <= word_count <= 2000 and formal_score < 0.3 and informal_score < 0.3:
            return TextCategory.LITERARY_NUANCED
        
        # Default to short_informal for short text, long_formal for long text
        if word_count < 200:
            return TextCategory.SHORT_INFORMAL
        return TextCategory.LONG_FORMAL


# ==============================================================================
# Text Preprocessing
# ==============================================================================

class TextPreprocessor:
    """
    Pre-processes text for optimal translation quality.
    
    Steps:
    1. De-noising: Remove HTML, excessive whitespace, normalize Unicode
    2. Sentence splitting: Use PySBD for proper segmentation
    3. Entity preservation: Identify named entities to preserve
    4. Glossary injection: Apply custom terminology mappings
    """
    
    # HTML/XML pattern
    HTML_PATTERN = re.compile(r'<[^>]+>')
    
    # Whitespace normalization
    WHITESPACE_PATTERN = re.compile(r'\s+')
    
    # URL pattern
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        r'(?:/[-\w%!$&\'()*+,.:;=@~]+)*/?(?:\?[-\w%!$&\'()*+,.:;=@~]*)?'
    )
    
    # Email pattern
    EMAIL_PATTERN = re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b')

    def __init__(self, source_lang: str = "en"):
        try:
            self.segmenter = pysbd.Segmenter(language=source_lang, clean=False)
        except Exception:
            # Fallback to English segmenter
            self.segmenter = pysbd.Segmenter(language="en", clean=False)

    def preprocess(
        self, 
        text: str, 
        glossary: Optional[Dict[str, str]] = None
    ) -> PreprocessedText:
        """
        Pre-process text for translation.
        
        Returns structured result with all preprocessing artifacts.
        """
        # Step 1: Extract entities to preserve
        entities = self._extract_entities(text)
        
        # Step 2: De-noise
        cleaned = self._denoise(text)
        
        # Step 3: Split into sentences
        sentences = self.segmenter.segment(cleaned)
        
        # Step 4: Apply glossary
        glossary_terms = glossary or {}
        
        return PreprocessedText(
            original_text=text,
            cleaned_text=cleaned,
            sentences=sentences,
            detected_entities=entities,
            glossary_terms=glossary_terms
        )
    
    def _denoise(self, text: str) -> str:
        """Remove noise from text while preserving meaning."""
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Preserve URLs by replacing with placeholders
        urls = self.URL_PATTERN.findall(text)
        for i, url in enumerate(urls):
            text = text.replace(url, f"__URL_{i}__")
        
        # Preserve emails
        emails = self.EMAIL_PATTERN.findall(text)
        for i, email in enumerate(emails):
            text = text.replace(email, f"__EMAIL_{i}__")
        
        # Normalize whitespace
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()
        
        # Restore URLs
        for i, url in enumerate(urls):
            text = text.replace(f"__URL_{i}__", url)
        
        # Restore emails
        for i, email in enumerate(emails):
            text = text.replace(f"__EMAIL_{i}__", email)
        
        return text
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities that should be preserved during translation."""
        entities = []
        
        # Extract URLs
        entities.extend(self.URL_PATTERN.findall(text))
        
        # Extract emails
        entities.extend(self.EMAIL_PATTERN.findall(text))
        
        # Extract quoted terms (often product names, proper nouns)
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        
        # Extract terms in backticks (code/technical terms)
        code_terms = re.findall(r'`([^`]+)`', text)
        entities.extend(code_terms)
        
        return entities


# ==============================================================================
# Model Router
# ==============================================================================

class ModelRouter:
    """
    Routes translation requests to the optimal model based on:
    - Source language
    - Text category
    - Specific requirements
    
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
    
    # Language to model mapping
    LANGUAGE_MODEL_MAP = {
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
    CATEGORY_MODEL_MAP = {
        TextCategory.SHORT_INFORMAL: "qwen_fast",
        TextCategory.LONG_FORMAL: "mistral",
        TextCategory.LITERARY_NUANCED: "llama",
        TextCategory.TECHNICAL: "nllb_distilled",
    }
    
    # Model to URL mapping
    MODEL_URL_MAP = {
        "qwen_mt": ServiceConfig.QWEN_MT_SERVICE_URL,
        "nllb": ServiceConfig.NLLB_SERVICE_URL,
        "llama": ServiceConfig.LLAMA_SERVICE_URL,
        "mistral": ServiceConfig.MISTRAL_SERVICE_URL,
        "gemma": ServiceConfig.GEMMA_SERVICE_URL,
        "qwen_fast": ServiceConfig.QWEN_FAST_SERVICE_URL,
        "nllb_distilled": ServiceConfig.NLLB_DISTILLED_SERVICE_URL,
    }
    
    # Model to served name mapping
    MODEL_NAME_MAP = {
        "qwen_mt": "qwen-mt-7b",
        "nllb": "nllb-200-3b",
        "llama": "llama-3-8b-instruct",
        "mistral": "mistral-nemo-12b",
        "gemma": "gemma-x2-28-9b",
        "qwen_fast": "qwen2-1-5b-instruct",
        "nllb_distilled": "nllb-200-distilled-1b",
    }

    def select_model(
        self,
        source_lang: str,
        target_lang: str,
        text_category: TextCategory,
        word_count: int
    ) -> Tuple[str, str, str]:
        """
        Select the optimal model for translation.
        
        Returns: (model_key, model_url, model_name)
        """
        # Priority 1: Language-specific routing
        if source_lang.lower() in self.LANGUAGE_MODEL_MAP:
            model_key = self.LANGUAGE_MODEL_MAP[source_lang.lower()]
            return (
                model_key,
                self.MODEL_URL_MAP[model_key],
                self.MODEL_NAME_MAP[model_key]
            )
        
        # Priority 2: For very short texts, use fast model regardless
        if word_count < 50 and text_category == TextCategory.SHORT_INFORMAL:
            return (
                "qwen_fast",
                self.MODEL_URL_MAP["qwen_fast"],
                self.MODEL_NAME_MAP["qwen_fast"]
            )
        
        # Priority 3: Category-based routing
        if text_category in self.CATEGORY_MODEL_MAP:
            model_key = self.CATEGORY_MODEL_MAP[text_category]
            return (
                model_key,
                self.MODEL_URL_MAP[model_key],
                self.MODEL_NAME_MAP[model_key]
            )
        
        # Default: NLLB for broad language support
        return (
            "nllb",
            self.MODEL_URL_MAP["nllb"],
            self.MODEL_NAME_MAP["nllb"]
        )


# ==============================================================================
# Translation Engine
# ==============================================================================

class TranslationEngine:
    """
    Core translation engine that orchestrates the full pipeline.
    
    Pipeline:
    1. Language Identification
    2. Text Characterization
    3. Pre-processing
    4. Model Selection
    5. Translation API Call
    6. Post-processing
    """

    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.lid = LanguageIdentifier(http_client)
        self.characterizer = TextCharacterizer()
        self.router = ModelRouter()

    async def translate(
        self,
        request: TranslationRequest
    ) -> TranslationResponse:
        """Execute the full translation pipeline."""
        start_time = time.time()
        preprocessing_steps = []
        
        # Stage 1: Language Identification
        if request.source_lang:
            lid_result = LIDResult(
                language_code=request.source_lang,
                confidence=1.0,
                script=None
            )
            preprocessing_steps.append("source_lang_provided")
        else:
            lid_result = await self.lid.identify(request.text)
            preprocessing_steps.append("lid_detection")
        
        logger.info(
            f"LID Result: lang={lid_result.language_code}, "
            f"confidence={lid_result.confidence:.2f}"
        )
        
        # Stage 2: Text Characterization
        char_result = self.characterizer.characterize(request.text)
        preprocessing_steps.append(f"characterized_as_{char_result.category.value}")
        
        logger.info(
            f"Characterization: category={char_result.category.value}, "
            f"words={char_result.word_count}, formality={char_result.formality_score:.2f}"
        )
        
        # Stage 3: Pre-processing
        preprocessor = TextPreprocessor(lid_result.language_code)
        preprocessed = preprocessor.preprocess(
            request.text,
            glossary=request.glossary
        )
        preprocessing_steps.append("text_cleaned")
        if preprocessed.detected_entities:
            preprocessing_steps.append(f"preserved_{len(preprocessed.detected_entities)}_entities")
        
        # Stage 4: Model Selection
        model_key, model_url, model_name = self.router.select_model(
            source_lang=lid_result.language_code,
            target_lang=request.target_lang,
            text_category=char_result.category,
            word_count=char_result.word_count
        )
        preprocessing_steps.append(f"routed_to_{model_key}")
        
        logger.info(f"Selected model: {model_name} at {model_url}")
        
        # Stage 5: Execute Translation
        translated_text = await self._call_translation_model(
            model_url=model_url,
            model_name=model_name,
            text=preprocessed.cleaned_text,
            source_lang=lid_result.language_code,
            target_lang=request.target_lang,
            category=char_result.category,
            glossary=preprocessed.glossary_terms
        )
        
        # Stage 6: Post-processing (restore entities if needed)
        translated_text = self._postprocess(
            translated_text,
            preprocessed.detected_entities
        )
        preprocessing_steps.append("postprocessed")
        
        latency_ms = (time.time() - start_time) * 1000
        
        return TranslationResponse(
            translated_text=translated_text,
            source_lang=lid_result.language_code,
            source_lang_confidence=lid_result.confidence,
            target_lang=request.target_lang,
            model_used=model_name,
            text_category=char_result.category.value,
            preprocessing_applied=preprocessing_steps,
            latency_ms=latency_ms
        )

    async def translate_stream(
        self,
        request: TranslationRequest
    ) -> AsyncGenerator[str, None]:
        """Execute streaming translation pipeline."""
        preprocessing_steps = []
        
        # Stage 1: Language Identification
        if request.source_lang:
            lid_result = LIDResult(
                language_code=request.source_lang,
                confidence=1.0,
                script=None
            )
        else:
            lid_result = await self.lid.identify(request.text)
        
        # Stage 2: Text Characterization
        char_result = self.characterizer.characterize(request.text)
        
        # Stage 3: Pre-processing
        preprocessor = TextPreprocessor(lid_result.language_code)
        preprocessed = preprocessor.preprocess(
            request.text,
            glossary=request.glossary
        )
        
        # Stage 4: Model Selection
        model_key, model_url, model_name = self.router.select_model(
            source_lang=lid_result.language_code,
            target_lang=request.target_lang,
            text_category=char_result.category,
            word_count=char_result.word_count
        )
        
        # Yield metadata first
        metadata = {
            "type": "metadata",
            "source_lang": lid_result.language_code,
            "source_lang_confidence": lid_result.confidence,
            "model_used": model_name,
            "text_category": char_result.category.value,
        }
        yield f"data: {json.dumps(metadata)}\n\n"
        
        # Stage 5: Stream Translation
        async for chunk in self._stream_translation_model(
            model_url=model_url,
            model_name=model_name,
            text=preprocessed.cleaned_text,
            source_lang=lid_result.language_code,
            target_lang=request.target_lang,
            category=char_result.category,
            glossary=preprocessed.glossary_terms
        ):
            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
        
        yield "data: [DONE]\n\n"

    async def _call_translation_model(
        self,
        model_url: str,
        model_name: str,
        text: str,
        source_lang: str,
        target_lang: str,
        category: TextCategory,
        glossary: Dict[str, str]
    ) -> str:
        """Call the selected translation model."""
        # Build the translation prompt based on model type
        prompt = self._build_translation_prompt(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            category=category,
            glossary=glossary
        )
        
        try:
            response = await self.http_client.post(
                f"{model_url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._get_system_prompt(source_lang, target_lang, category)
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max(len(text) * 2, 1024),  # Allow expansion
                    "temperature": 0.3,  # Low for translation accuracy
                    "stream": False
                },
                timeout=120.0
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Translation model call failed: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Translation service unavailable: {str(e)}"
            )

    async def _stream_translation_model(
        self,
        model_url: str,
        model_name: str,
        text: str,
        source_lang: str,
        target_lang: str,
        category: TextCategory,
        glossary: Dict[str, str]
    ) -> AsyncGenerator[str, None]:
        """Stream translation from the selected model."""
        prompt = self._build_translation_prompt(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            category=category,
            glossary=glossary
        )
        
        try:
            async with self.http_client.stream(
                "POST",
                f"{model_url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._get_system_prompt(source_lang, target_lang, category)
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max(len(text) * 2, 1024),
                    "temperature": 0.3,
                    "stream": True
                },
                timeout=120.0
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Streaming translation failed: {e}")
            yield f"[Error: {str(e)}]"

    def _build_translation_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        category: TextCategory,
        glossary: Dict[str, str]
    ) -> str:
        """Build the translation prompt with optional glossary constraints."""
        prompt_parts = [f"Translate the following text from {source_lang} to {target_lang}:"]
        
        # Add glossary constraints if present
        if glossary:
            glossary_str = "\n".join(f"- '{k}' → '{v}'" for k, v in glossary.items())
            prompt_parts.append(f"\nUse these specific translations:\n{glossary_str}")
        
        # Add category-specific instructions
        if category == TextCategory.TECHNICAL:
            prompt_parts.append("\nMaintain technical accuracy and preserve code/technical terms.")
        elif category == TextCategory.LONG_FORMAL:
            prompt_parts.append("\nMaintain formal register and legal/professional terminology.")
        elif category == TextCategory.LITERARY_NUANCED:
            prompt_parts.append("\nPreserve stylistic nuances and creative expression.")
        elif category == TextCategory.SHORT_INFORMAL:
            prompt_parts.append("\nKeep the informal, conversational tone.")
        
        prompt_parts.append(f"\n\nText to translate:\n{text}")
        
        return "\n".join(prompt_parts)

    def _get_system_prompt(
        self,
        source_lang: str,
        target_lang: str,
        category: TextCategory
    ) -> str:
        """Get the system prompt for the translation model."""
        base_prompt = (
            f"You are a professional translator specializing in {source_lang} to {target_lang} translation. "
            "Provide accurate, natural translations that preserve the original meaning and tone."
        )
        
        if category == TextCategory.TECHNICAL:
            base_prompt += " You specialize in technical documentation and preserve all technical terminology exactly."
        elif category == TextCategory.LONG_FORMAL:
            base_prompt += " You specialize in legal and formal documents, maintaining precise legal terminology."
        elif category == TextCategory.LITERARY_NUANCED:
            base_prompt += " You specialize in creative and marketing translation, capturing subtle nuances."
        
        return base_prompt

    def _postprocess(self, text: str, entities: List[str]) -> str:
        """Post-process translated text."""
        # For now, just clean up any double spaces or trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# ==============================================================================
# FastAPI Application
# ==============================================================================

app = FastAPI(
    title="Machine Translation Orchestrator",
    description="KServe InferenceGraph orchestrator for multi-model machine translation",
    version="1.0.0"
)

# Global HTTP client and translation engine
http_client: Optional[httpx.AsyncClient] = None
translation_engine: Optional[TranslationEngine] = None


@app.on_event("startup")
async def startup():
    """Initialize HTTP client and translation engine on startup."""
    global http_client, translation_engine
    http_client = httpx.AsyncClient(timeout=60.0)
    translation_engine = TranslationEngine(http_client)
    logger.info("Machine Translation Orchestrator started")


@app.on_event("shutdown")
async def shutdown():
    """Clean up resources on shutdown."""
    global http_client
    if http_client:
        await http_client.aclose()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    return {"status": "ready"}


@app.post("/v1/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate text using the optimal model based on language and text characteristics.
    
    The orchestrator will:
    1. Detect the source language (if not provided)
    2. Characterize the text (informal, formal, technical, etc.)
    3. Pre-process the text (de-noise, segment sentences)
    4. Route to the optimal translation model
    5. Post-process the result
    """
    if not translation_engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if request.stream:
        # Return streaming response
        return StreamingResponse(
            translation_engine.translate_stream(request),
            media_type="text/event-stream"
        )
    
    return await translation_engine.translate(request)


@app.post("/v1/translate/batch")
async def translate_batch(requests: List[TranslationRequest]):
    """
    Batch translate multiple texts concurrently.
    
    Optimal for processing multiple documents or segments.
    """
    if not translation_engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Execute all translations concurrently
    tasks = [translation_engine.translate(req) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to error responses
    responses = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            responses.append({
                "index": i,
                "error": str(result),
                "success": False
            })
        else:
            responses.append({
                "index": i,
                "result": result.model_dump(),
                "success": True
            })
    
    return {"results": responses}


@app.post("/v1/identify")
async def identify_language(text: str):
    """
    Identify the language of the given text.
    
    Uses FastText for accurate detection across 200+ languages.
    """
    if not translation_engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    result = await translation_engine.lid.identify(text)
    return {
        "language": result.language_code,
        "confidence": result.confidence,
        "script": result.script,
        "language_family": LanguageIdentifier.get_language_family(result.language_code).value
    }


@app.post("/v1/characterize")
async def characterize_text(text: str):
    """
    Characterize text to determine its category and properties.
    
    Returns:
    - category: short_informal, long_formal, literary_nuanced, or technical
    - word_count: Number of words
    - formality_score: 0.0 (informal) to 1.0 (formal)
    - technical_density: Ratio of technical terms
    """
    if not translation_engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    result = translation_engine.characterizer.characterize(text)
    return {
        "category": result.category.value,
        "word_count": result.word_count,
        "avg_sentence_length": result.avg_sentence_length,
        "formality_score": result.formality_score,
        "technical_density": result.technical_density
    }


@app.get("/v1/models")
async def list_models():
    """List available translation models and their specializations."""
    return {
        "models": [
            {
                "id": "qwen-mt-7b",
                "name": "Qwen/Qwen-MT-7B",
                "specialization": "Chinese, Japanese, Korean",
                "category": "asian_specialist"
            },
            {
                "id": "nllb-200-3b",
                "name": "facebook/nllb-200-3.3B",
                "specialization": "200+ languages, Russian, Hindi, Bengali",
                "category": "generalist"
            },
            {
                "id": "llama-3-8b-instruct",
                "name": "meta-llama/Llama-3.1-8B-Instruct",
                "specialization": "Spanish, literary/nuanced content",
                "category": "high_quality_llm"
            },
            {
                "id": "mistral-nemo-12b",
                "name": "mistralai/Mistral-Nemo-12B",
                "specialization": "French, long documents (128k context)",
                "category": "high_quality_llm"
            },
            {
                "id": "gemma-x2-28-9b",
                "name": "Gemma-X2-28-9B",
                "specialization": "Arabic, 28 most spoken languages",
                "category": "multilingual"
            },
            {
                "id": "qwen2-1-5b-instruct",
                "name": "Qwen2-1.5B-Instruct",
                "specialization": "Fast/real-time, informal content",
                "category": "fast"
            },
            {
                "id": "nllb-200-distilled-1b",
                "name": "facebook/nllb-200-distilled-1.3B",
                "specialization": "Technical documentation",
                "category": "technical"
            }
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
