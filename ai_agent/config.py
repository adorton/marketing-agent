"""
Configuration management for the AI Agent.

Handles loading and validation of configuration settings from environment
variables and configuration files.
"""

import os
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


class LLMConfig(BaseModel):
    """Configuration for LLM provider settings."""
    
    provider: str = Field(default="openai", description="LLM provider (openai, anthropic, custom)")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    model: str = Field(default="gpt-3.5-turbo", description="Model name to use")
    base_url: Optional[str] = Field(default=None, description="Custom base URL for API")
    max_tokens: int = Field(default=1000, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, description="Temperature for response generation")
    
    @validator('provider')
    def validate_provider(cls, v):
        """Validate that the provider is supported."""
        supported_providers = ['openai', 'anthropic', 'custom']
        if v not in supported_providers:
            raise ValueError(f"Provider must be one of: {supported_providers}")
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """Validate temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v


class FileConfig(BaseModel):
    """Configuration for file reading settings."""
    
    input_directory: str = Field(default="./input", description="Directory containing text files")
    file_extensions: List[str] = Field(default=[".txt", ".md"], description="File extensions to process")
    recursive: bool = Field(default=True, description="Whether to search subdirectories recursively")
    max_file_size: int = Field(default=1024 * 1024, description="Maximum file size in bytes (1MB default)")
    encoding: str = Field(default="utf-8", description="File encoding to use")


class Config(BaseModel):
    """Main configuration class."""
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    files: FileConfig = Field(default_factory=FileConfig)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        load_dotenv()
        
        llm_config = LLMConfig(
            provider=os.getenv('LLM_PROVIDER', 'openai'),
            api_key=os.getenv('LLM_API_KEY'),
            model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
            base_url=os.getenv('LLM_BASE_URL'),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '1000')),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.7'))
        )
        
        files_config = FileConfig(
            input_directory=os.getenv('INPUT_DIRECTORY', './input'),
            file_extensions=os.getenv('FILE_EXTENSIONS', '.txt,.md').split(','),
            recursive=os.getenv('RECURSIVE', 'true').lower() == 'true',
            max_file_size=int(os.getenv('MAX_FILE_SIZE', str(1024 * 1024))),
            encoding=os.getenv('FILE_ENCODING', 'utf-8')
        )
        
        return cls(llm=llm_config, files=files_config)
    
    def validate_api_key(self) -> bool:
        """Validate that API key is provided."""
        if not self.llm.api_key:
            raise ValueError("LLM API key is required. Set LLM_API_KEY environment variable.")
        return True
