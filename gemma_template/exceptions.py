from __future__ import annotations


class GemmaTemplateError(Exception):
    """Gemma Template Exception"""

    def __init__(self, message):
        self.message = message


class LanguageError(GemmaTemplateError):
    """Language Error"""


class DatasetError(GemmaTemplateError):
    """Dataset Error"""


class MaxHiddenRatioError(DatasetError):
    """Max Hidden Ratio Error"""
