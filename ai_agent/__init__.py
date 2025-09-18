"""
AI Agent Package

A Python package for creating AI agents that read local text files
and use them to prompt remote Large Language Models.
"""

__version__ = "0.1.0"
__author__ = "Andrew Dorton"

from .agent import AIAgent
from .file_reader import FileReader
from .llm_client import LLMClient
from .config import Config

__all__ = ["AIAgent", "FileReader", "LLMClient", "Config"]
