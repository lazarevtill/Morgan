"""
Main entry point for STT service
"""

import argparse
import asyncio

from api.server import main as server_main
from service import STTService

from shared.config.base import ServiceConfig
from shared.utils.logging import setup_logging


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Morgan STT Service")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind to")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")

    args = parser.parse_args()

    # Setup configuration
    config = ServiceConfig("stt", args.config)

    # Override with command line arguments
    host = args.host
    port = args.port
    log_level = args.log_level if args.log_level else config.get("log_level", "INFO")

    # Setup logging
    logger = setup_logging("stt_main", log_level, "logs/stt_main.log")

    logger.info("Starting Morgan STT Service...")
    logger.info(f"Configuration: {config.all()}")

    try:
        # Initialize STT service
        stt_service = STTService(config)
        await stt_service.start()

        # Start the API server
        await server_main(stt_service, host, port)
    except KeyboardInterrupt:
        logger.info("STT Service interrupted by user")
    except Exception as e:
        logger.error(f"STT Service failed: {e}")
        raise
    finally:
        if "stt_service" in locals():
            await stt_service.stop()


if __name__ == "__main__":
    asyncio.run(main())
