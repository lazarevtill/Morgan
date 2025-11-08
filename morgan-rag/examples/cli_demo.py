#!/usr/bin/env python3
"""
Morgan CLI Demo - Demonstrates CLI functionality.

This script shows how to use the Morgan CLI programmatically
and demonstrates all major features.
"""

import asyncio
from pathlib import Path

from morgan.cli.config import CLIConfig
from morgan.cli.formatters import ConsoleFormatter
from morgan.cli.utils import (
    assistant_context,
    check_health,
    display_response,
)


async def main():
    """Run CLI demo."""
    print("=" * 70)
    print("Morgan CLI Demo")
    print("=" * 70)
    print()

    # 1. Create configuration
    print("1. Creating configuration...")
    config = CLIConfig(
        storage_path=str(Path.home() / ".morgan" / "demo"),
        llm_base_url="http://localhost:11434",
        llm_model="llama3.2:latest",
        vector_db_url="http://localhost:6333",
        enable_emotion_detection=True,
        enable_learning=True,
        enable_rag=True,
        use_rich_formatting=True,
    )
    print(f"   ✓ Config created with storage at: {config.storage_path}")
    print()

    # 2. Create formatter
    print("2. Creating formatter...")
    formatter = ConsoleFormatter(use_rich=config.use_rich_formatting)
    print("   ✓ Formatter initialized")
    print()

    # 3. Initialize assistant
    print("3. Initializing assistant...")
    try:
        async with assistant_context(config) as assistant:
            print("   ✓ Assistant initialized successfully")
            print()

            # 4. Health check
            print("4. Performing health check...")
            health_status = await check_health(assistant)

            if formatter.use_rich:
                table = formatter.format_health(health_status)
                formatter.console.print(table)
            else:
                output = formatter.format_health(health_status)
                print(output)
            print()

            # 5. Process a message
            print("5. Processing sample message...")
            response = await assistant.process_message(
                user_id="demo_user",
                message="What is artificial intelligence?",
                session_id="demo_session",
            )

            print("   ✓ Message processed")
            print()

            # 6. Display response
            print("6. Displaying formatted response...")
            display_response(
                response,
                formatter,
                show_sources=True,
                show_emotion=True,
                show_metrics=True,
            )
            print()

            # 7. Get learning stats
            if assistant.learning_engine:
                print("7. Retrieving learning statistics...")
                metrics = await assistant.learning_engine.get_metrics()

                if formatter.use_rich:
                    stats_panel = formatter.format_learning_stats(metrics)
                    formatter.console.print(stats_panel)
                else:
                    output = formatter.format_learning_stats(metrics)
                    print(output)
                print()

            print("=" * 70)
            print("Demo completed successfully!")
            print("=" * 70)

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
