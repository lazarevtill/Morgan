"""
API Server for Morgan Core
"""
import asyncio
import json
import logging
import aiohttp
from aiohttp import web
import base64
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class APIServer:
    """HTTP API Server for Morgan Core"""

    def __init__(self, core, host="0.0.0.0", port=8000):
        self.core = core
        self.host = host
        self.port = port
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        """Configure API routes"""
        self.app.add_routes([
            web.post('/api/text', self.handle_text_input),
            web.post('/api/audio', self.handle_audio_input),
            web.get('/api/health', self.health_check),
            web.get('/api/voices', self.get_voices),
            web.post('/api/conversation/reset', self.reset_conversation),
            web.static('/ui', '/opt/morgan/web-ui', show_index=True)
        ])

    async def start(self):
        """Start the API server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        self.site = web.TCPSite(runner, self.host, self.port)
        await self.site.start()
        logger.info(f"API server started on http://{self.host}:{self.port}")

    async def stop(self):
        """Stop the API server"""
        await self.site.stop()
        logger.info("API server stopped")

    async def handle_text_input(self, request):
        """Handle text input API requests"""
        try:
            data = await request.json()
            text = data.get('text')
            user_id = data.get('user_id', 'default')

            if not text:
                return web.json_response({
                    "error": "Missing required field: text"
                }, status=400)

            # Process the text input
            response = await self.core.process_text_input(text, user_id)

            # If there's audio in the response, encode it as base64
            if 'audio' in response:
                response['audio'] = base64.b64encode(response['audio']).decode('utf-8')

            return web.json_response(response)

        except Exception as e:
            logger.exception(f"Error handling text input: {e}")
            return web.json_response({
                "error": f"Error processing request: {str(e)}"
            }, status=500)

    async def handle_audio_input(self, request):
        """Handle audio input API requests"""
        try:
            data = await request.post()
            user_id = data.get('user_id', 'default')

            if 'audio' not in data:
                return web.json_response({
                    "error": "Missing required field: audio"
                }, status=400)

            # Get audio data
            audio_file = data['audio']
            audio_bytes = audio_file.file.read()

            # Process the audio input
            response = await self.core.process_audio_input(audio_bytes, user_id)

            # If there's audio in the response, encode it as base64
            if 'audio' in response:
                response['audio'] = base64.b64encode(response['audio']).decode('utf-8')

            return web.json_response(response)

        except Exception as e:
            logger.exception(f"Error handling audio input: {e}")
            return web.json_response({
                "error": f"Error processing request: {str(e)}"
            }, status=500)

    async def health_check(self, request):
        """API health check endpoint"""
        return web.json_response({
            "status": "ok",
            "version": "0.1.0"
        })

    async def get_voices(self, request):
        """Get available TTS voices"""
        try:
            voices = await self.core.tts_service.get_available_voices()
            return web.json_response({"voices": voices})
        except Exception as e:
            logger.exception(f"Error getting voices: {e}")
            return web.json_response({
                "error": f"Error getting voices: {str(e)}"
            }, status=500)

    async def reset_conversation(self, request):
        """Reset a conversation context"""
        try:
            data = await request.json()
            user_id = data.get('user_id', 'default')

            success = self.core.state_manager.reset_context(user_id)
            return web.json_response({
                "success": success
            })

        except Exception as e:
            logger.exception(f"Error resetting conversation: {e}")
            return web.json_response({
                "error": f"Error resetting conversation: {str(e)}"
            }, status=500)