"""
Machine Translation Orchestrator

A multi-stage translation pipeline that:
1. Detects source language (LID)
2. Characterizes text type
3. Pre-processes for translation
4. Routes to optimal model
"""

from .app import app

__all__ = ["app"]
