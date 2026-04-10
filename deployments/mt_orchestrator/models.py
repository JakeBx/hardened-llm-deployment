"""
Data models for Machine Translation Orchestrator.

Contains all Pydantic models, dataclasses, and enums used across modules.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


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
