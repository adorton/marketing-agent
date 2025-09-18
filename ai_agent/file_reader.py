"""
File reading module for the AI Agent.

Handles scanning directories, reading text files, and preparing content
for LLM processing.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Generator, Optional
from dataclasses import dataclass


@dataclass
class FileContent:
    """Represents the content of a file with metadata."""
    
    file_path: Path
    content: str
    file_size: int
    encoding: str
    
    def __post_init__(self):
        """Validate file content."""
        if not self.content.strip():
            logging.warning(f"File {self.file_path} appears to be empty")


class FileReader:
    """Handles reading and processing text files from directories."""
    
    def __init__(
        self,
        input_directory: str = "./input",
        file_extensions: List[str] = None,
        recursive: bool = True,
        max_file_size: int = 1024 * 1024,
        encoding: str = "utf-8"
    ):
        """
        Initialize the FileReader.
        
        Args:
            input_directory: Directory to scan for files
            file_extensions: List of file extensions to include
            recursive: Whether to search subdirectories
            max_file_size: Maximum file size in bytes
            encoding: File encoding to use
        """
        self.input_directory = Path(input_directory)
        self.file_extensions = file_extensions or [".txt", ".md"]
        self.recursive = recursive
        self.max_file_size = max_file_size
        self.encoding = encoding
        self.logger = logging.getLogger(__name__)
        
        # Ensure extensions start with dot
        self.file_extensions = [
            ext if ext.startswith('.') else f'.{ext}'
            for ext in self.file_extensions
        ]
    
    def scan_directory(self) -> List[Path]:
        """
        Scan the input directory for files matching the criteria.
        
        Returns:
            List of file paths that match the criteria
        """
        if not self.input_directory.exists():
            self.logger.warning(f"Input directory {self.input_directory} does not exist")
            return []
        
        if not self.input_directory.is_dir():
            self.logger.error(f"{self.input_directory} is not a directory")
            return []
        
        files = []
        pattern = "**/*" if self.recursive else "*"
        
        for file_path in self.input_directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.file_extensions:
                # Check file size
                try:
                    file_size = file_path.stat().st_size
                    if file_size <= self.max_file_size:
                        files.append(file_path)
                    else:
                        self.logger.warning(
                            f"Skipping {file_path}: file size ({file_size} bytes) "
                            f"exceeds maximum ({self.max_file_size} bytes)"
                        )
                except OSError as e:
                    self.logger.error(f"Error checking file size for {file_path}: {e}")
        
        self.logger.info(f"Found {len(files)} files to process")
        return files
    
    def read_file(self, file_path: Path) -> Optional[FileContent]:
        """
        Read a single file and return its content.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            FileContent object or None if reading failed
        """
        try:
            # Double-check file size before reading
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                self.logger.warning(f"File {file_path} exceeds maximum size")
                return None
            
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
            
            return FileContent(
                file_path=file_path,
                content=content,
                file_size=file_size,
                encoding=self.encoding
            )
            
        except UnicodeDecodeError as e:
            self.logger.error(f"Encoding error reading {file_path}: {e}")
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    self.logger.info(f"Successfully read {file_path} with {encoding} encoding")
                    return FileContent(
                        file_path=file_path,
                        content=content,
                        file_size=file_size,
                        encoding=encoding
                    )
                except UnicodeDecodeError:
                    continue
            
            self.logger.error(f"Could not read {file_path} with any supported encoding")
            return None
            
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def read_all_files(self) -> Generator[FileContent, None, None]:
        """
        Read all files from the input directory.
        
        Yields:
            FileContent objects for each successfully read file
        """
        files = self.scan_directory()
        
        for file_path in files:
            file_content = self.read_file(file_path)
            if file_content:
                yield file_content
    
    def get_files_summary(self) -> Dict[str, any]:
        """
        Get a summary of files in the input directory.
        
        Returns:
            Dictionary with file statistics
        """
        files = self.scan_directory()
        
        total_size = 0
        successful_reads = 0
        
        for file_path in files:
            try:
                file_size = file_path.stat().st_size
                total_size += file_size
                
                # Try to read the file
                if self.read_file(file_path):
                    successful_reads += 1
            except OSError:
                pass
        
        return {
            "total_files": len(files),
            "successful_reads": successful_reads,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "input_directory": str(self.input_directory),
            "file_extensions": self.file_extensions,
            "recursive": self.recursive
        }
