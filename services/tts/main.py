"""
Main entry point for TTS service
"""
import asyncio
import argparse
import logging

from shared.config.base import ServiceConfig
from shared.utils.logging import setup_logging
from .api.server import main as server_main


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Morgan TTS Service")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")

    args = parser.parse_args()

    # Setup configuration
    config = ServiceConfig("tts", args.config)

    # Override with command line arguments
    config.set("host", args.host)
    config.set("port", args.port)
    config.set("log_level", args.log_level)

    # Setup logging
    logger = setup_logging(
        "tts_main",
        config.get("log_level", "INFO"),
        "logs/tts_main.log"
    )

    logger.info("Starting Morgan TTS Service...")
    logger.info(f"Configuration: {config.all()}")

    try:
        # Start the API server
        await server_main()
    except KeyboardInterrupt:
        logger.info("TTS Service interrupted by user")
    except Exception as e:
        logger.error(f"TTS Service failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
