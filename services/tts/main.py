"""
Main entry point for TTS service
"""

import asyncio
import argparse

from shared.config.base import ServiceConfig
from shared.utils.logging import setup_logging
from service import TTSService
from api.server import main as server_main


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
    host = args.host
    port = args.port
    log_level = args.log_level if args.log_level else config.get("log_level", "INFO")

    # Setup logging
    logger = setup_logging("tts_main", log_level, "logs/tts_main.log")

    logger.info("Starting Morgan TTS Service...")
    logger.info(f"Configuration: {config.all()}")

    try:
        # Initialize TTS service
        tts_service = TTSService(config)
        await tts_service.start()

        # Start the API server
        await server_main(tts_service, host, port)
    except KeyboardInterrupt:
        logger.info("TTS Service interrupted by user")
    except Exception as e:
        logger.error(f"TTS Service failed: {e}")
        raise
    finally:
        if "tts_service" in locals():
            await tts_service.stop()


if __name__ == "__main__":
    asyncio.run(main())
