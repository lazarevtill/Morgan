# Morgan v2 Quick Start Guide
## Get Your CLI Running in 1 Week

> **Goal**: Have users chatting with Morgan via CLI by end of Week 1
> **Effort**: ~5-10 hours focused development
> **Skills**: Python, Click framework, basic LLM integration

---

## What You're Building (Week 1)

A simple but functional CLI that lets users:
```bash
# Chat with Morgan
$ morgan chat
🤖 Morgan: Hello! How can I help?
👤 You: What is Python?
🤖 Morgan: Python is a high-level programming language...

# Ask quick questions
$ morgan ask "How do I use Docker?"

# Check system health
$ morgan health
✅ Overall Status: HEALTHY
```

---

## Prerequisites

```bash
# Python 3.10+
python --version

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install core dependencies
pip install click rich httpx pydantic python-dotenv openai
```

---

## Day-by-Day Plan

### Day 1: Project Structure (2 hours)

```bash
# Create directory structure
mkdir -p morgan-rag/morgan/{cli,core,config,utils}
mkdir -p morgan-rag/tests

# Create files
touch morgan-rag/morgan/__init__.py
touch morgan-rag/morgan/__main__.py
touch morgan-rag/morgan/cli/app.py
touch morgan-rag/morgan/core/assistant.py
touch morgan-rag/morgan/config/settings.py
touch morgan-rag/requirements.txt
touch morgan-rag/.env.example
```

**requirements.txt**:
```
click>=8.1.0
rich>=13.0.0
httpx>=0.27.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
openai>=1.0.0
```

**.env.example**:
```bash
# LLM Configuration
MORGAN_LLM_PROVIDER=ollama  # or openai
MORGAN_LLM_BASE_URL=http://localhost:11434
MORGAN_LLM_MODEL=qwen2.5:7b
MORGAN_OPENAI_API_KEY=sk-...  # if using OpenAI

# Logging
MORGAN_LOG_LEVEL=INFO
```

### Day 2: CLI Framework (2 hours)

**morgan-rag/morgan/cli/app.py**:
```python
import click
from rich.console import Console
from rich.markdown import Markdown

console = Console()

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Morgan AI Assistant - Your intelligent companion"""
    pass

@cli.command()
def chat():
    """Start an interactive chat session"""
    console.print("[bold green]🤖 Morgan:[/bold green] Hello! How can I help?")

    while True:
        try:
            user_input = console.input("[bold blue]👤 You:[/bold blue] ")
            if user_input.lower() in ['/exit', '/quit']:
                console.print("[yellow]Goodbye![/yellow]")
                break

            # TODO: Connect to assistant
            console.print(f"[green]🤖 Morgan:[/green] You said: {user_input}")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

@cli.command()
@click.argument('question')
def ask(question):
    """Ask a single question"""
    console.print(f"[bold blue]Question:[/bold blue] {question}")
    # TODO: Get answer from assistant
    console.print("[green]Answer will appear here[/green]")

@cli.command()
def health():
    """Check system health"""
    from rich.table import Table

    table = Table(title="System Health")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")

    table.add_row("LLM Service", "✅ Healthy")
    table.add_row("Configuration", "✅ Loaded")

    console.print(table)

if __name__ == '__main__':
    cli()
```

**morgan-rag/morgan/__main__.py**:
```python
from morgan.cli.app import cli

if __name__ == '__main__':
    cli()
```

**Test it**:
```bash
cd morgan-rag
python -m morgan --help
python -m morgan chat
```

### Day 3: Configuration System (2 hours)

**morgan-rag/morgan/config/settings.py**:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    """Morgan configuration settings"""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_prefix='MORGAN_',
        case_sensitive=False,
        extra='ignore'
    )

    # LLM Settings
    llm_provider: Literal['ollama', 'openai'] = 'ollama'
    llm_base_url: str = 'http://localhost:11434'
    llm_model: str = 'qwen2.5:7b'
    openai_api_key: str | None = None

    # System Settings
    log_level: str = 'INFO'
    max_history: int = 10
    timeout: int = 30

    @property
    def is_ollama(self) -> bool:
        return self.llm_provider == 'ollama'

# Global settings instance
settings = Settings()
```

**Test it**:
```python
from morgan.config.settings import settings
print(f"Using {settings.llm_provider} with {settings.llm_model}")
```

### Day 4: LLM Integration (2 hours)

**morgan-rag/morgan/core/assistant.py**:
```python
from openai import AsyncOpenAI
from morgan.config.settings import settings
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class MorganAssistant:
    """Core AI assistant logic"""

    def __init__(self):
        # OpenAI SDK works with both OpenAI and Ollama
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url if settings.is_ollama else None,
            api_key=settings.openai_api_key or "ollama"  # Ollama doesn't need key
        )
        self.conversation_history: List[Dict[str, str]] = []

    async def chat(self, user_message: str) -> str:
        """Send message and get response"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Keep only recent history
        recent_history = self.conversation_history[-settings.max_history:]

        try:
            # Get response from LLM
            response = await self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You are Morgan, a helpful AI assistant."},
                    *recent_history
                ],
                temperature=0.7,
                max_tokens=2000
            )

            assistant_message = response.choices[0].message.content

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    async def ask(self, question: str) -> str:
        """One-shot question (no history)"""
        try:
            response = await self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You are Morgan, a helpful AI assistant. Answer concisely."},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
```

### Day 5: Connect CLI to Assistant (2 hours)

**Update morgan-rag/morgan/cli/app.py**:
```python
import click
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from morgan.core.assistant import MorganAssistant
from morgan.config.settings import settings

console = Console()

# Create global assistant instance
assistant = MorganAssistant()

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Morgan AI Assistant - Your intelligent companion"""
    pass

@cli.command()
def chat():
    """Start an interactive chat session"""
    console.print(f"[bold green]🤖 Morgan:[/bold green] Hello! I'm using {settings.llm_model}")
    console.print("[dim]Type /exit or Ctrl+C to quit[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold blue]👤 You:[/bold blue] ")
            if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not user_input.strip():
                continue

            # Get response from assistant (async)
            response = asyncio.run(assistant.chat(user_input))

            # Display with markdown formatting
            console.print("[bold green]🤖 Morgan:[/bold green]")
            console.print(Markdown(response))
            console.print()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

@cli.command()
@click.argument('question')
def ask(question):
    """Ask a single question"""
    console.print(f"[bold blue]❓ Question:[/bold blue] {question}\n")

    with console.status("[bold green]Thinking...[/bold green]"):
        response = asyncio.run(assistant.ask(question))

    console.print("[bold green]💡 Answer:[/bold green]")
    console.print(Markdown(response))

@cli.command()
def health():
    """Check system health"""
    from rich.table import Table

    table = Table(title="System Health")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    # Test LLM connection
    try:
        test_response = asyncio.run(assistant.ask("Say 'OK'"))
        llm_status = "✅ Healthy"
        llm_details = f"Model: {settings.llm_model}"
    except Exception as e:
        llm_status = "❌ Error"
        llm_details = str(e)[:50]

    table.add_row("LLM Service", llm_status, llm_details)
    table.add_row("Configuration", "✅ Loaded", f"Provider: {settings.llm_provider}")
    table.add_row("Chat History", "✅ Ready", f"Max: {settings.max_history} messages")

    console.print(table)

if __name__ == '__main__':
    cli()
```

**Test it**:
```bash
# Make sure Ollama is running
ollama serve  # In another terminal

# Or use OpenAI by setting:
# export MORGAN_LLM_PROVIDER=openai
# export MORGAN_OPENAI_API_KEY=sk-...

# Test the CLI
cd morgan-rag
python -m morgan health
python -m morgan ask "What is Python?"
python -m morgan chat
```

---

## Success Criteria (End of Week 1)

✅ **Must Have**:
- [ ] `morgan chat` works with multi-turn conversation
- [ ] `morgan ask "question"` works for one-shot queries
- [ ] `morgan health` shows service status
- [ ] Conversation history maintained in chat
- [ ] Rich formatted output (colors, markdown)
- [ ] Graceful error handling (no crashes)

✅ **Nice to Have**:
- [ ] Progress spinner while thinking
- [ ] Markdown rendering for code blocks
- [ ] Configuration command to view/edit settings
- [ ] Basic logging to file

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'morgan'"
**Solution**: Install in development mode:
```bash
cd morgan-rag
pip install -e .
```

Add **setup.py**:
```python
from setuptools import setup, find_packages

setup(
    name="morgan-rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.0",
        "rich>=13.0.0",
        "httpx>=0.27.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        "openai>=1.0.0",
    ],
    entry_points={
        'console_scripts': [
            'morgan=morgan.cli.app:cli',
        ],
    },
)
```

Then:
```bash
pip install -e .
morgan --help  # Now works from anywhere!
```

### Issue: "Connection refused" to Ollama
**Solution**: Start Ollama server:
```bash
ollama serve
```

Or use OpenAI instead:
```bash
export MORGAN_LLM_PROVIDER=openai
export MORGAN_OPENAI_API_KEY=sk-your-key-here
```

### Issue: Slow responses
**Solution**: Use smaller model:
```bash
export MORGAN_LLM_MODEL=qwen2.5:1.5b  # Faster, less capable
# or
export MORGAN_LLM_MODEL=qwen2.5:7b    # Good balance
```

---

## Next Steps (Week 2)

Once basic CLI works:
1. Add `morgan init` command to create config file
2. Add `morgan config` command to view/edit settings
3. Improve error messages with suggestions
4. Add conversation export (`morgan chat --save transcript.txt`)
5. Add streaming responses for better UX

**Then move to Phase 2**: RAG system (document ingestion)

---

## Resources

### Documentation
- Click Framework: https://click.palletsprojects.com/
- Rich Console: https://rich.readthedocs.io/
- Pydantic Settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/

### Full Implementation Plan
- See: `V2_IMPLEMENTATION_PLAN.md` for complete roadmap
- See: `CLI_DESIGN_INTENT.md` for design philosophy

---

## Getting Help

If you get stuck:
1. Check `morgan health` - is everything connected?
2. Check logs - what's the actual error?
3. Test Ollama directly: `ollama run qwen2.5:7b`
4. Simplify - comment out complex features, get basic chat working first

**Remember**: Week 1 goal is just to get basic chat working. Don't overcomplicate!

---

**Status**: Ready to Start
**Estimated Time**: 10 hours over 5 days
**Next Action**: Create project structure (Day 1)
