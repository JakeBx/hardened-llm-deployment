"""
Translation Engine module.

Core translation engine that orchestrates the full pipeline.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Tuple

import httpx
from fastapi import HTTPException

from .characteriser import TextCharacterizer
from .lid import LanguageIdentifier
from .models import (
    LIDResult,
    PreprocessedText,
    TextCategory,
    TranslationRequest,
    TranslationResponse,
)
from .preprocessor import TextPreprocessor
from .router import select_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        model_key, model_url, model_name = select_model(
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
        model_key, model_url, model_name = select_model(
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
