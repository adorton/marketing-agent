# AI Agent

A Python-based AI agent that automates the task of reading text files from a local directory and using those files to prompt remote Large Language Models (LLMs).

## Features

- **Multi-Provider Support**: Works with OpenAI, Anthropic Claude, and custom OpenAI-compatible endpoints
- **Flexible File Processing**: Supports multiple text file formats (`.txt`, `.md`, etc.)
- **Recursive Directory Scanning**: Can process files in subdirectories
- **Streaming Support**: Real-time response streaming for better user experience
- **Configuration Management**: Environment-based configuration with validation
- **Command-Line Interface**: Easy-to-use CLI for all operations
- **Error Handling**: Robust error handling with detailed logging
- **Batch Processing**: Process multiple files efficiently

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

3. Initialize the project:

```bash
python cli.py init
```

4. Configure your API keys:

```bash
cp env.template .env
# Edit .env and add your API key
```

## Quick Start

### 1. Initialize Project

```bash
python cli.py init
```

This creates:
- `input/` directory with example files
- `env.template` configuration file

### 2. Configure API Key

Edit the `.env` file and add your API key:

```bash
# For OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_api_key_here

# For Anthropic
LLM_PROVIDER=anthropic
LLM_API_KEY=your_anthropic_api_key_here
```

### 3. Process Files

```bash
# Process all files in the input directory
python cli.py process

# Process with a custom prompt
python cli.py process --prompt "Summarize the key points"

# Stream responses in real-time
python cli.py process --stream

# Process a single file
python cli.py single input/example.txt

# Process with custom prompt and streaming
python cli.py single input/example.txt --prompt "What is the main theme?" --stream
```

## Configuration

The agent can be configured via environment variables or by creating a `.env` file:

### LLM Configuration

- `LLM_PROVIDER`: Provider to use (`openai`, `anthropic`, `custom`)
- `LLM_API_KEY`: Your API key for the provider
- `LLM_MODEL`: Model name (e.g., `gpt-3.5-turbo`, `claude-3-sonnet-20240229`)
- `LLM_BASE_URL`: Custom base URL (for custom providers)
- `LLM_MAX_TOKENS`: Maximum tokens in response
- `LLM_TEMPERATURE`: Response creativity (0.0-2.0)

### File Processing Configuration

- `INPUT_DIRECTORY`: Directory containing text files
- `FILE_EXTENSIONS`: Comma-separated list of file extensions
- `RECURSIVE`: Whether to search subdirectories (`true`/`false`)
- `MAX_FILE_SIZE`: Maximum file size in bytes
- `FILE_ENCODING`: File encoding to use

## CLI Commands

### `process`
Process all text files in the input directory.

```bash
python cli.py process [OPTIONS]

Options:
  -i, --input-dir TEXT    Input directory containing text files
  -p, --prompt TEXT       Custom prompt to use for processing
  -o, --output TEXT       Output file to save results (JSON format)
  -s, --stream           Stream responses in real-time
```

### `single`
Process a single file.

```bash
python cli.py single <file_path> [OPTIONS]

Options:
  -p, --prompt TEXT       Custom prompt to use for processing
  -s, --stream           Stream the response
```

### `scan`
Scan and display information about files in the input directory.

```bash
python cli.py scan [OPTIONS]

Options:
  -i, --input-dir TEXT    Input directory to scan
```

### `config`
Display current configuration.

```bash
python cli.py config
```

### `init`
Initialize a new project with example files and configuration.

```bash
python cli.py init
```

## Usage Examples

### Basic File Processing

```bash
# Process all files with default settings
python cli.py process

# Process with custom prompt
python cli.py process --prompt "Extract the main ideas and create a summary"
```

### Streaming Responses

```bash
# Stream all files
python cli.py process --stream

# Stream single file with custom prompt
python cli.py single input/example.txt --prompt "Analyze this text" --stream
```

### Batch Processing with Output

```bash
# Process all files and save results to JSON
python cli.py process --output results.json
```

### Custom Directory

```bash
# Process files from a different directory
python cli.py process --input-dir /path/to/documents
```

## Programmatic Usage

You can also use the AI Agent programmatically:

```python
from ai_agent import AIAgent, Config

# Load configuration from environment
config = Config.from_env()

# Initialize agent
agent = AIAgent(config)

# Process all files
for result in agent.process_all_files("Summarize the key points"):
    if result.success:
        print(f"File: {result.file_path}")
        print(f"Response: {result.llm_response.content}")
        print("-" * 50)

# Process single file
result = agent.process_single_file("path/to/file.txt", "What is this about?")
if result.success:
    print(result.llm_response.content)
```

## Supported Providers

### OpenAI
```bash
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-3.5-turbo  # or gpt-4, gpt-4-turbo, etc.
```

### Anthropic Claude
```bash
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...
LLM_MODEL=claude-3-sonnet-20240229  # or claude-3-haiku-20240307, etc.
```

### Custom OpenAI-Compatible API
```bash
LLM_PROVIDER=custom
LLM_API_KEY=your_api_key
LLM_MODEL=your_model_name
LLM_BASE_URL=https://your-api-endpoint.com/v1
```

## File Support

The agent supports various text file formats:
- `.txt` - Plain text files
- `.md` - Markdown files
- `.json` - JSON files (treated as text)
- `.csv` - CSV files (treated as text)
- `.log` - Log files

You can customize supported extensions in the configuration.

## Error Handling

The agent includes comprehensive error handling:

- **File Reading Errors**: Handles encoding issues, file size limits, and permission errors
- **API Errors**: Manages rate limits, authentication errors, and network issues
- **Configuration Errors**: Validates configuration and provides helpful error messages

## Logging

Enable verbose logging to see detailed information:

```bash
python cli.py --verbose process
```

## Development

### Project Structure

```
ai_agent/
├── __init__.py          # Package initialization
├── agent.py            # Main AI Agent class
├── config.py           # Configuration management
├── file_reader.py      # File reading utilities
└── llm_client.py       # LLM provider clients

cli.py                  # Command-line interface
pyproject.toml          # Project configuration and dependencies
env.template           # Configuration template
README.md              # This file
```

### Adding New Providers

To add a new LLM provider:

1. Create a new class inheriting from `LLMProvider` in `llm_client.py`
2. Implement `generate_response()` and `stream_response()` methods
3. Update the provider creation logic in `LLMClient._create_provider()`

### Testing

```bash
# Install development dependencies
uv sync --extra dev

# Run tests (when available)
pytest

# Check code style
black ai_agent/ cli.py
flake8 ai_agent/ cli.py

# Type checking
mypy ai_agent/ cli.py
```

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
