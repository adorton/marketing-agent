"""
Main AI Agent class that orchestrates file reading and LLM processing.

This is the core component that ties together file reading, LLM communication,
and provides a high-level interface for processing local files with remote LLMs.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

from .config import Config
from .file_reader import FileReader, FileContent
from .llm_client import LLMClient, LLMResponse


@dataclass
class ProcessingResult:
    """Represents the result of processing a single file."""
    
    file_path: Path
    file_content: FileContent
    llm_response: LLMResponse
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "file_path": str(self.file_path),
            "file_size": self.file_content.file_size,
            "encoding": self.file_content.encoding,
            "llm_response": self.llm_response.to_dict(),
            "processing_time": self.processing_time,
            "success": self.success,
            "error_message": self.error_message
        }


class AIAgent:
    """
    Main AI Agent class for processing local files with remote LLMs.
    
    This class provides a high-level interface for:
    - Reading text files from a directory
    - Sending file content to remote LLMs
    - Collecting and organizing responses
    - Error handling and logging
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the AI Agent.
        
        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or Config.from_env()
        self.config.validate_api_key()
        
        self.logger = logging.getLogger(__name__)
        self.file_reader = FileReader(
            input_directory=self.config.files.input_directory,
            file_extensions=self.config.files.file_extensions,
            recursive=self.config.files.recursive,
            max_file_size=self.config.files.max_file_size,
            encoding=self.config.files.encoding
        )
        self.llm_client = LLMClient(self.config)
        
        self.logger.info(f"AI Agent initialized with {self.config.llm.provider} provider")
    
    def process_single_file(self, file_path: Path, user_prompt: str = None) -> ProcessingResult:
        """
        Process a single file with the LLM.
        
        Args:
            file_path: Path to the file to process
            user_prompt: Optional additional prompt from the user
            
        Returns:
            ProcessingResult object
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing file: {file_path}")
            
            # Read the file
            file_content = self.file_reader.read_file(file_path)
            if not file_content:
                return ProcessingResult(
                    file_path=file_path,
                    file_content=FileContent(file_path, "", 0, "utf-8"),
                    llm_response=None,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="Failed to read file"
                )
            
            # Process with LLM
            llm_response = self.llm_client.process_file_content(
                file_content.content, 
                user_prompt
            )
            
            processing_time = time.time() - start_time
            self.logger.info(f"Successfully processed {file_path} in {processing_time:.2f}s")
            
            return ProcessingResult(
                file_path=file_path,
                file_content=file_content,
                llm_response=llm_response,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.logger.error(error_msg)
            
            return ProcessingResult(
                file_path=file_path,
                file_content=FileContent(file_path, "", 0, "utf-8"),
                llm_response=None,
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )
    
    def process_all_files(self, user_prompt: str = None) -> Generator[ProcessingResult, None, None]:
        """
        Process all files in the input directory.
        
        Args:
            user_prompt: Optional additional prompt from the user
            
        Yields:
            ProcessingResult objects for each processed file
        """
        self.logger.info("Starting batch processing of all files")
        
        files_summary = self.file_reader.get_files_summary()
        self.logger.info(f"Found {files_summary['total_files']} files to process")
        
        for file_content in self.file_reader.read_all_files():
            result = self.process_single_file(file_content.file_path, user_prompt)
            yield result
    
    def process_files_with_prompt(self, files: List[Path], prompt: str) -> List[ProcessingResult]:
        """
        Process specific files with a custom prompt.
        
        Args:
            files: List of file paths to process
            prompt: Custom prompt to use for processing
            
        Returns:
            List of ProcessingResult objects
        """
        self.logger.info(f"Processing {len(files)} files with custom prompt")
        
        results = []
        for file_path in files:
            result = self.process_single_file(file_path, prompt)
            results.append(result)
        
        return results
    
    def get_files_summary(self) -> Dict[str, Any]:
        """Get a summary of files in the input directory."""
        return self.file_reader.get_files_summary()
    
    def stream_file_processing(self, file_path: Path, user_prompt: str = None) -> Generator[str, None, None]:
        """
        Stream the LLM response for a single file.
        
        Args:
            file_path: Path to the file to process
            user_prompt: Optional additional prompt from the user
            
        Yields:
            Chunks of the LLM response as they arrive
        """
        try:
            # Read the file
            file_content = self.file_reader.read_file(file_path)
            if not file_content:
                yield "Error: Failed to read file"
                return
            
            # Prepare messages
            system_prompt = "You are a helpful AI assistant that analyzes and responds to text content provided by the user."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please analyze the following text content:\n\n{file_content.content}"}
            ]
            
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})
            
            # Stream the response
            for chunk in self.llm_client.stream_response(messages):
                yield chunk
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def batch_process_with_summary(self, user_prompt: str = None) -> Dict[str, Any]:
        """
        Process all files and return a summary of results.
        
        Args:
            user_prompt: Optional additional prompt from the user
            
        Returns:
            Dictionary with processing summary and results
        """
        results = list(self.process_all_files(user_prompt))
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        total_processing_time = sum(r.processing_time for r in results)
        total_tokens = sum(
            r.llm_response.usage.get('total_tokens', 0) 
            for r in successful_results 
            if r.llm_response and r.llm_response.usage
        )
        
        return {
            "total_files": len(results),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(results) if results else 0,
            "total_tokens": total_tokens,
            "results": [r.to_dict() for r in results],
            "errors": [r.error_message for r in failed_results if r.error_message]
        }
