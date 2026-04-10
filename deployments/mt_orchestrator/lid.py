"""
Language Identification (LID) module.

Uses FastText or GlotLID for language detection with script-based fallback.
"""

import re
from typing import Optional

import httpx

from .config import LID_SERVICE_URL
from .models import LIDResult, LanguageFamily


# Script detection patterns - module-level for reuse
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

# Language code normalization aliases
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


class LanguageIdentifier:
    """
    Language identification using FastText or GlotLID.
    
    Primary: facebook/fasttext-language-identification (200+ languages)
    Fallback: cis-lmu/glotlid (better for low-resource languages)
    Falls back to script-based detection if LID service is unavailable.
    """
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.lid_url = LID_SERVICE_URL

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
            lang_code = LANGUAGE_ALIASES.get(lang_code, lang_code)
            
            return LIDResult(
                language_code=lang_code,
                confidence=result.get("confidence", 0.0),
                script=result.get("script")
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"LID service call failed: {e}, using script-based fallback")
            return self._script_based_detection(text)
    
    def _script_based_detection(self, text: str) -> LIDResult:
        """Fallback script-based language detection."""
        for script_name, pattern in SCRIPT_PATTERNS.items():
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
