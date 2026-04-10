"""
Text Preprocessing module.

Pre-processes text for optimal translation quality.
"""

import re
from typing import Dict, List, Optional

import pysbd
from bs4 import BeautifulSoup

from .models import PreprocessedText


# Module-level patterns for reuse
HTML_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')
URL_PATTERN = re.compile(
    r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    r'(?:/[-\w%!$&\'()*+,.:;=@~]+)*/?(?:\?[-\w%!$&\'()*+,.:;=@~]*)?'
)
EMAIL_PATTERN = re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b')


class TextPreprocessor:
    """
    Pre-processes text for optimal translation quality.
    
    Steps:
    1. De-noising: Remove HTML, excessive whitespace, normalize Unicode
    2. Sentence splitting: Use PySBD for proper segmentation
    3. Entity preservation: Identify named entities to preserve
    4. Glossary injection: Apply custom terminology mappings
    """
    
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
        urls = URL_PATTERN.findall(text)
        for i, url in enumerate(urls):
            text = text.replace(url, f"__URL_{i}__")
        
        # Preserve emails
        emails = EMAIL_PATTERN.findall(text)
        for i, email in enumerate(emails):
            text = text.replace(email, f"__EMAIL_{i}__")
        
        # Normalize whitespace
        text = WHITESPACE_PATTERN.sub(" ", text).strip()
        
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
        entities.extend(URL_PATTERN.findall(text))
        
        # Extract emails
        entities.extend(EMAIL_PATTERN.findall(text))
        
        # Extract quoted terms (often product names, proper nouns)
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        
        # Extract terms in backticks (code/technical terms)
        code_terms = re.findall(r'`([^`]+)`', text)
        entities.extend(code_terms)
        
        return entities
