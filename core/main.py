"""
Main entry point for Core service
"""

import argparse
import asyncio

from core.app import main as core_main
from shared.config.base import ServiceConfig
from shared.utils.logging import setup_logging


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Morgan Core Service")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")

    args = parser.parse_args()

    # Setup configuration
    config = ServiceConfig("core", args.config)

    # Override with command line arguments
    config.set("host", args.host)
    config.set("port", args.port)
    config.set("log_level", args.log_level)

    # Setup logging
    logger = setup_logging(
        "core_main", config.get("log_level", "INFO"), "logs/core_main.log"
    )

    logger.info("Starting Morgan Core Service...")
    logger.info(f"Configuration: {config.all()}")

    try:
        # Start the core service
        await core_main()
    except KeyboardInterrupt:
        logger.info("Core Service interrupted by user")
    except Exception as e:
        logger.error(f"Core Service failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
