"""
Main entry point for LLM service
"""
import asyncio
import argparse
import logging
from pathlib import Path

from shared.config.base import ServiceConfig
from shared.utils.logging import setup_logging
from service import LLMService
from api.server import main as server_main


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Morgan LLM Service")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")

    args = parser.parse_args()

    # Setup configuration
    config = ServiceConfig("llm", args.config)

    # Override with command line arguments
    host = args.host
    port = args.port
    if args.log_level:
        config.config["log_level"] = args.log_level

    # Setup logging
    logger = setup_logging(
        "llm_main",
        config.get("log_level", "INFO"),
        "logs/llm_main.log"
    )

    logger.info("Starting Morgan LLM Service...")
    logger.info(f"Configuration: {config.all()}")

    try:
        # Initialize LLM service
        llm_service = LLMService(config)
        await llm_service.start()

        # Start the API server
        await server_main(llm_service, host, port)
    except KeyboardInterrupt:
        logger.info("LLM Service interrupted by user")
    except Exception as e:
        logger.error(f"LLM Service failed: {e}")
        raise
    finally:
        if 'llm_service' in locals():
            await llm_service.stop()


if __name__ == "__main__":
    asyncio.run(main())
