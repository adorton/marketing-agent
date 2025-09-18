#!/usr/bin/env python3
"""
Command-line interface for the AI Agent.

Provides a user-friendly CLI for processing text files with remote LLMs.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional

import click

from ai_agent import AIAgent, Config


def logging_setup(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """AI Agent - Process local text files with remote LLMs."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    logging_setup(verbose)


@cli.command()
@click.option('--input-dir', '-i', default='./input', help='Input directory containing text files')
@click.option('--prompt', '-p', help='Custom prompt to use for processing')
@click.option('--output', '-o', help='Output file to save results (JSON format)')
@click.option('--stream', '-s', is_flag=True, help='Stream responses in real-time')
@click.pass_context
def process(ctx, input_dir, prompt, output, stream):
    """Process all text files in the input directory."""
    try:
        # Create configuration
        config = Config.from_env()
        config.files.input_directory = input_dir
        
        # Initialize agent
        agent = AIAgent(config)
        
        if stream:
            # Stream processing (process one file at a time with streaming)
            files = agent.file_reader.scan_directory()
            if not files:
                click.echo("No files found in the input directory.", err=True)
                return
            
            click.echo(f"Found {len(files)} files. Processing with streaming...")
            
            for file_path in files:
                click.echo(f"\n--- Processing {file_path} ---")
                for chunk in agent.stream_file_processing(file_path, prompt):
                    click.echo(chunk, nl=False)
                click.echo("\n" + "="*50)
        else:
            # Batch processing
            click.echo("Processing files...")
            results = agent.batch_process_with_summary(prompt)
            
            # Display summary
            click.echo(f"\nProcessing Summary:")
            click.echo(f"  Total files: {results['total_files']}")
            click.echo(f"  Successful: {results['successful']}")
            click.echo(f"  Failed: {results['failed']}")
            click.echo(f"  Total time: {results['total_processing_time']:.2f}s")
            click.echo(f"  Average time: {results['average_processing_time']:.2f}s")
            
            if results['total_tokens']:
                click.echo(f"  Total tokens: {results['total_tokens']}")
            
            # Save results if output file specified
            if output:
                with open(output, 'w') as f:
                    json.dump(results, f, indent=2)
                click.echo(f"\nResults saved to {output}")
            
            # Display errors if any
            if results['errors']:
                click.echo("\nErrors encountered:")
                for error in results['errors']:
                    click.echo(f"  - {error}")
    
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if ctx.obj['verbose']:
            raise
        sys.exit(1)


@cli.command()
@click.option('--input-dir', '-i', default='./input', help='Input directory to scan')
@click.pass_context
def scan(ctx, input_dir):
    """Scan and display information about files in the input directory."""
    try:
        config = Config.from_env()
        config.files.input_directory = input_dir
        
        agent = AIAgent(config)
        summary = agent.get_files_summary()
        
        click.echo("File Summary:")
        click.echo(f"  Directory: {summary['input_directory']}")
        click.echo(f"  Total files: {summary['total_files']}")
        click.echo(f"  Successful reads: {summary['successful_reads']}")
        click.echo(f"  Total size: {summary['total_size_mb']} MB")
        click.echo(f"  File extensions: {', '.join(summary['file_extensions'])}")
        click.echo(f"  Recursive: {summary['recursive']}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if ctx.obj['verbose']:
            raise
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--prompt', '-p', help='Custom prompt to use for processing')
@click.option('--stream', '-s', is_flag=True, help='Stream the response')
@click.pass_context
def single(ctx, file_path, prompt, stream):
    """Process a single file."""
    try:
        config = Config.from_env()
        agent = AIAgent(config)
        
        file_path = Path(file_path)
        
        if stream:
            click.echo(f"Processing {file_path} with streaming...")
            for chunk in agent.stream_file_processing(file_path, prompt):
                click.echo(chunk, nl=False)
            click.echo()
        else:
            click.echo(f"Processing {file_path}...")
            result = agent.process_single_file(file_path, prompt)
            
            if result.success:
                click.echo(f"\nProcessing completed in {result.processing_time:.2f}s")
                click.echo(f"Model: {result.llm_response.model}")
                click.echo(f"Provider: {result.llm_response.provider}")
                
                if result.llm_response.usage:
                    click.echo(f"Tokens used: {result.llm_response.usage.get('total_tokens', 'N/A')}")
                
                click.echo(f"\nResponse:\n{result.llm_response.content}")
            else:
                click.echo(f"Error processing file: {result.error_message}", err=True)
                sys.exit(1)
    
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if ctx.obj['verbose']:
            raise
        sys.exit(1)


@cli.command()
@click.pass_context
def config(ctx):
    """Display current configuration."""
    try:
        config = Config.from_env()
        
        click.echo("Current Configuration:")
        click.echo(f"  LLM Provider: {config.llm.provider}")
        click.echo(f"  Model: {config.llm.model}")
        click.echo(f"  Max Tokens: {config.llm.max_tokens}")
        click.echo(f"  Temperature: {config.llm.temperature}")
        click.echo(f"  Input Directory: {config.files.input_directory}")
        click.echo(f"  File Extensions: {', '.join(config.files.file_extensions)}")
        click.echo(f"  Recursive: {config.files.recursive}")
        click.echo(f"  Max File Size: {config.files.max_file_size} bytes")
        click.echo(f"  Encoding: {config.files.encoding}")
        
        # Check if API key is set
        if config.llm.api_key:
            masked_key = config.llm.api_key[:8] + "..." + config.llm.api_key[-4:]
            click.echo(f"  API Key: {masked_key}")
        else:
            click.echo("  API Key: Not set (set LLM_API_KEY environment variable)")
    
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if ctx.obj['verbose']:
            raise
        sys.exit(1)


@cli.command()
def init():
    """Initialize a new project with example files and configuration."""
    try:
        # Create input directory
        input_dir = Path("./input")
        input_dir.mkdir(exist_ok=True)
        
        # Create example files
        example_files = [
            ("example.txt", "This is an example text file.\n\nIt contains multiple paragraphs and demonstrates how the AI agent can process local text files."),
            ("notes.md", "# Example Notes\n\nThis is a markdown file with some notes.\n\n- Bullet point 1\n- Bullet point 2\n\n## Conclusion\n\nThis demonstrates markdown processing."),
        ]
        
        for filename, content in example_files:
            file_path = input_dir / filename
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    f.write(content)
                click.echo(f"Created example file: {file_path}")
        
        # Create .env template
        env_file = Path(".env.template")
        env_content = """# AI Agent Configuration Template
# Copy this to .env and fill in your values

# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-3.5-turbo
LLM_BASE_URL=
LLM_MAX_TOKENS=1000
LLM_TEMPERATURE=0.7

# File Processing Configuration
INPUT_DIRECTORY=./input
FILE_EXTENSIONS=.txt,.md
RECURSIVE=true
MAX_FILE_SIZE=1048576
FILE_ENCODING=utf-8
"""
        
        if not env_file.exists():
            with open(env_file, 'w') as f:
                f.write(env_content)
            click.echo(f"Created configuration template: {env_file}")
        
        click.echo("\nProject initialized successfully!")
        click.echo("Next steps:")
        click.echo("1. Copy .env.template to .env")
        click.echo("2. Add your API key to the .env file")
        click.echo("3. Run: python cli.py scan")
        click.echo("4. Run: python cli.py process")
    
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
