"""
Text Characterisation module.

Analyzes text to determine its category for optimal model selection.
"""

import re
from typing import List

import pysbd

from .models import CharacterizationResult, TextCategory


# Informal indicators - module-level for reuse
INFORMAL_PATTERNS = [
    re.compile(r'\b(lol|omg|btw|idk|tbh|ngl|brb|afk)\b', re.IGNORECASE),
    re.compile(r'[!?]{2,}'),  # Multiple punctuation
    re.compile(r'[\U0001F600-\U0001F64F]'),  # Emojis
    re.compile(r'@\w+'),  # Mentions
    re.compile(r'#\w+'),  # Hashtags
]

# Technical indicators - module-level for reuse
TECHNICAL_PATTERNS = [
    re.compile(r'\b(API|SDK|HTTP|JSON|XML|REST|SQL|HTML|CSS)\b'),
    re.compile(r'\b(function|class|method|variable|parameter|return)\b', re.IGNORECASE),
    re.compile(r'[\w]+\([\w,\s]*\)'),  # Function calls
    re.compile(r'`[^`]+`'),  # Code snippets
    re.compile(r'\b\d+\.\d+\.\d+\b'),  # Version numbers
]

# Formal indicators - module-level for reuse
FORMAL_PATTERNS = [
    re.compile(r'\b(hereby|whereas|pursuant|notwithstanding|heretofore)\b', re.IGNORECASE),
    re.compile(r'\b(shall|must|may not|is required|in accordance with)\b', re.IGNORECASE),
    re.compile(r'\b(diagnosis|prognosis|treatment|contraindicated)\b', re.IGNORECASE),
]


class TextCharacterizer:
    """
    Analyzes text to determine its category for optimal model selection.
    
    Categories:
    - short_informal: Real-time chat, tweets, social media (< 100 words, informal)
    - long_formal: Legal documents, medical papers (> 500 words, formal)
    - literary_nuanced: Marketing copy, creative writing (variable, nuanced)
    - technical: Code, API docs, specifications (high technical term density)
    """
    
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
            for pattern in INFORMAL_PATTERNS
        )
        # Normalize by text length
        return min(1.0, matches / max(1, len(text.split()) / 10))
    
    def _calculate_formal_score(self, text: str) -> float:
        """Calculate formal language score (0.0 to 1.0)."""
        matches = sum(
            len(pattern.findall(text)) 
            for pattern in FORMAL_PATTERNS
        )
        return min(1.0, matches / max(1, len(text.split()) / 20))
    
    def _calculate_technical_score(self, text: str) -> float:
        """Calculate technical language score (0.0 to 1.0)."""
        matches = sum(
            len(pattern.findall(text)) 
            for pattern in TECHNICAL_PATTERNS
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
