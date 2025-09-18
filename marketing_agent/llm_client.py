"""
LLM client module for the AI Agent.

Handles communication with various Large Language Model providers
including OpenAI, Anthropic, and custom endpoints.
"""

import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass

import requests
from pydantic import BaseModel


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""
    
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    provider: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "provider": self.provider
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        """Initialize the provider."""
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.kwargs = kwargs
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def stream_response(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Stream a response from the LLM."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: Optional[str] = None, **kwargs):
        """Initialize OpenAI provider."""
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url or "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response using OpenAI API."""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            choice = data["choices"][0]
            
            return LLMResponse(
                content=choice["message"]["content"],
                model=self.model,
                usage=data.get("usage"),
                finish_reason=choice.get("finish_reason"),
                provider="openai"
            )
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
        except KeyError as e:
            self.logger.error(f"Unexpected response format from OpenAI: {e}")
            raise
    
    def stream_response(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Stream a response using OpenAI API."""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data.strip() == '[DONE]':
                            break
                        try:
                            json_data = json.loads(data)
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                delta = json_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
                            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenAI streaming API error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", **kwargs):
        """Initialize Anthropic provider."""
        super().__init__(api_key, model, **kwargs)
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response using Anthropic API."""
        url = "https://api.anthropic.com/v1/messages"
        
        # Convert OpenAI format to Anthropic format
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                user_messages.append(msg["content"])
            elif msg["role"] == "assistant":
                user_messages.append(msg["content"])
        
        # Combine user messages
        user_content = "\n\n".join(user_messages)
        
        payload = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "messages": [{"role": "user", "content": user_content}]
        }
        
        if system_message:
            payload["system"] = system_message
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            content = data["content"][0]["text"]
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=data.get("usage"),
                finish_reason=data.get("stop_reason"),
                provider="anthropic"
            )
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise
        except KeyError as e:
            self.logger.error(f"Unexpected response format from Anthropic: {e}")
            raise
    
    def stream_response(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Stream a response using Anthropic API (simplified implementation)."""
        # For now, we'll just return the full response and yield it in chunks
        # Anthropic streaming is more complex and would need SSE handling
        response = self.generate_response(messages, **kwargs)
        words = response.content.split()
        for word in words:
            yield word + " "


class CustomProvider(LLMProvider):
    """Custom API provider for any OpenAI-compatible endpoint."""
    
    def __init__(self, api_key: str, model: str, base_url: str, **kwargs):
        """Initialize custom provider."""
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response using custom API."""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            choice = data["choices"][0]
            
            return LLMResponse(
                content=choice["message"]["content"],
                model=self.model,
                usage=data.get("usage"),
                finish_reason=choice.get("finish_reason"),
                provider="custom"
            )
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Custom API error: {e}")
            raise
        except KeyError as e:
            self.logger.error(f"Unexpected response format from custom API: {e}")
            raise
    
    def stream_response(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Stream a response using custom API."""
        # Simplified streaming for custom provider
        response = self.generate_response(messages, **kwargs)
        words = response.content.split()
        for word in words:
            yield word + " "


class LLMClient:
    """Main LLM client that manages different providers."""
    
    def __init__(self, config):
        """Initialize the LLM client with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.provider = self._create_provider()
    
    def _create_provider(self) -> LLMProvider:
        """Create the appropriate provider based on configuration."""
        provider_config = self.config.llm
        
        if provider_config.provider == "openai":
            return OpenAIProvider(
                api_key=provider_config.api_key,
                model=provider_config.model,
                base_url=provider_config.base_url,
                max_tokens=provider_config.max_tokens,
                temperature=provider_config.temperature
            )
        elif provider_config.provider == "anthropic":
            return AnthropicProvider(
                api_key=provider_config.api_key,
                model=provider_config.model,
                max_tokens=provider_config.max_tokens,
                temperature=provider_config.temperature
            )
        elif provider_config.provider == "custom":
            if not provider_config.base_url:
                raise ValueError("base_url is required for custom provider")
            return CustomProvider(
                api_key=provider_config.api_key,
                model=provider_config.model,
                base_url=provider_config.base_url,
                max_tokens=provider_config.max_tokens,
                temperature=provider_config.temperature
            )
        else:
            raise ValueError(f"Unsupported provider: {provider_config.provider}")
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response using the configured provider."""
        return self.provider.generate_response(messages, **kwargs)
    
    def stream_response(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Stream a response using the configured provider."""
        return self.provider.stream_response(messages, **kwargs)
    
    def process_file_content(self, file_content: str, user_prompt: str = None) -> LLMResponse:
        """
        Process file content with an optional user prompt.
        
        Args:
            file_content: The content of the file to process
            user_prompt: Optional additional prompt from the user
            
        Returns:
            LLMResponse object
        """
        system_prompt = "You are a helpful AI assistant that analyzes and responds to text content provided by the user."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please analyze the following text content:\n\n{file_content}"}
        ]
        
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        
        return self.generate_response(messages)
