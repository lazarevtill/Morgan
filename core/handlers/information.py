"""
Enhanced information retrieval handler for Morgan
"""
from typing import Dict, Any, Optional, List
import logging
import json
import aiohttp
import asyncio
from datetime import datetime, timedelta
import os

from .base_handler import BaseHandler
from conversation.context import ConversationContext

logger = logging.getLogger(__name__)


class InformationHandler(BaseHandler):
    """Handler for information retrieval commands"""

    def __init__(self, core_instance):
        super().__init__(core_instance)
        self.weather_api_key = None
        self.weather_cache = {}
        self.weather_cache_time = 30 * 60  # 30 minutes in seconds

        # Try to load the weather API key from config
        try:
            handlers_config = self.core.config_manager.load_handlers_config()
            info_config = handlers_config.get('handlers', {}).get('information', {})
            self.weather_api_key = info_config.get('weather_api_key', '')
        except Exception as e:
            logger.error(f"Error loading information handler config: {e}")

    async def handle(self, params: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """
        Handle an information retrieval command

        Args:
            params: Command parameters
            context: Conversation context

        Returns:
            Response dictionary
        """
        info_type = params.get("type", "")
        query = params.get("query", "")

        # Process based on information type
        if info_type == "weather":
            location = params.get("location", "")
            days = params.get("days", 1)
            return await self._handle_weather(location, days, context)
        elif info_type == "knowledge":
            return await self._handle_knowledge(query, context)
        elif info_type == "time":
            return await self._handle_time(params.get("timezone"), context)
        elif info_type == "date":
            return await self._handle_date(params.get("format"), context)
        else:
            # Pass to LLM for general information
            history = context.get_history()
            response = await self.core.llm_service.process_input(query, history)
            return await self.format_response(response, True)

    async def _handle_weather(self, location: str, days: int = 1, context: ConversationContext) -> Dict[str, Any]:
        """Handle a weather query"""
        if not location or location.lower() in ("here", "current", "my location"):
            # In a real implementation, we might get the user's location from their profile
            # For now, we'll use a default location
            location = context.get_variable("default_location", "Belgrade, Serbia")

        # Check if we have a cached result that's still valid
        cache_key = f"{location}_{days}"
        cached = self.weather_cache.get(cache_key)
        if cached and (datetime.now() - cached["timestamp"]).total_seconds() < self.weather_cache_time:
            logger.debug(f"Using cached weather data for {location}")
            return await self.format_response(cached["response"], True)

        # If we have a weather API key, try to use a weather API
        if self.weather_api_key:
            try:
                weather_data = await self._fetch_weather_api(location, days)
                if weather_data:
                    response = self._format_weather_response(weather_data, location, days)

                    # Cache the result
                    self.weather_cache[cache_key] = {
                        "timestamp": datetime.now(),
                        "response": response
                    }

                    return await self.format_response(response, True)
            except Exception as e:
                logger.error(f"Error fetching weather data: {e}")
                # Fall through to LLM-based response if API fails

        # Generate response using LLM as fallback
        prompt = self._generate_weather_prompt(location, days)
        response = await self.core.llm_service.process_input(prompt)

        # Cache the LLM response too
        self.weather_cache[cache_key] = {
            "timestamp": datetime.now(),
            "response": response
        }

        return await self.format_response(response, True)

    async def _handle_knowledge(self, query: str, context: ConversationContext) -> Dict[str, Any]:
        """Handle a knowledge query"""
        # For knowledge queries, we use the LLM directly with the conversation history
        # for better context awareness
        history = context.get_history()

        # Enhance query with any active topics from context
        active_topic = context.get_variable("active_topic")
        if active_topic and active_topic not in query:
            enhanced_query = f"{query} in the context of {active_topic}"
        else:
            enhanced_query = query

        response = await self.core.llm_service.process_input(enhanced_query, history)

        return await self.format_response(response, True)

    async def _handle_time(self, timezone: Optional[str], context: ConversationContext) -> Dict[str, Any]:
        """Handle a time query"""
        from datetime import datetime
        import pytz

        # If no timezone specified, use the default timezone
        if not timezone:
            timezone = context.get_variable("timezone", "Europe/Belgrade")

        try:
            # Convert to the requested timezone
            tz = pytz.timezone(timezone)
            current_time = datetime.now(tz)
            formatted_time = current_time.strftime("%I:%M %p")

            response = f"The current time in {timezone} is {formatted_time}."
            return await self.format_response(response, True)
        except Exception as e:
            logger.error(f"Error handling time query: {e}")
            return await self.format_response(
                f"I'm sorry, I couldn't get the current time for {timezone}. Please check that the timezone is valid.",
                True
            )

    async def _handle_date(self, date_format: Optional[str], context: ConversationContext) -> Dict[str, Any]:
        """Handle a date query"""
        from datetime import datetime

        try:
            current_date = datetime.now()

            if date_format == "full":
                formatted_date = current_date.strftime("%A, %B %d, %Y")
            elif date_format == "short":
                formatted_date = current_date.strftime("%m/%d/%Y")
            else:
                # Default format
                formatted_date = current_date.strftime("%B %d, %Y")

            response = f"Today's date is {formatted_date}."
            return await self.format_response(response, True)
        except Exception as e:
            logger.error(f"Error handling date query: {e}")
            return await self.format_response(
                "I'm sorry, I couldn't get the current date.",
                True
            )

    async def _fetch_weather_api(self, location: str, days: int = 1) -> Optional[Dict[str, Any]]:
        """
        Fetch weather data from an API

        Note: This is a placeholder implementation using OpenWeatherMap API.
        In a real implementation, you would use a proper weather API with your API key.
        """
        if not self.weather_api_key:
            return None

        # Example using OpenWeatherMap API
        url = f"https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": location,
            "appid": self.weather_api_key,
            "units": "metric",
            "cnt": min(days * 8, 40)  # 8 forecasts per day, max 5 days (40 forecasts)
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Weather API error: {response.status} - {error_text}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None

    def _format_weather_response(self, weather_data: Dict[str, Any], location: str, days: int) -> str:
        """Format weather data into a readable response"""
        try:
            city = weather_data.get("city", {}).get("name", location)
            country = weather_data.get("city", {}).get("country", "")

            forecasts = weather_data.get("list", [])
            if not forecasts:
                return f"I couldn't find weather information for {location}."

            # Group forecasts by day
            daily_forecasts = {}
            for forecast in forecasts:
                timestamp = forecast.get("dt")
                if not timestamp:
                    continue

                date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                if date not in daily_forecasts:
                    daily_forecasts[date] = []

                daily_forecasts[date].append(forecast)

            # Format the response
            if days == 1:
                # Today's forecast
                today = datetime.now().strftime("%Y-%m-%d")
                if today in daily_forecasts:
                    today_forecasts = daily_forecasts[today]

                    # Get the current/next forecast
                    current = min(today_forecasts, key=lambda x: abs(datetime.now().timestamp() - x.get("dt", 0)))

                    temp = current.get("main", {}).get("temp", 0)
                    feels_like = current.get("main", {}).get("feels_like", 0)
                    description = current.get("weather", [{}])[0].get("description", "")
                    humidity = current.get("main", {}).get("humidity", 0)
                    wind_speed = current.get("wind", {}).get("speed", 0)

                    return f"Current weather in {city}, {country}: {description.capitalize()}. Temperature: {temp}°C (feels like {feels_like}°C). Humidity: {humidity}%. Wind speed: {wind_speed} m/s."
                else:
                    # Use the first day available
                    first_day = sorted(daily_forecasts.keys())[0]
                    first_day_forecasts = daily_forecasts[first_day]

                    # Get the first forecast of the day
                    first = first_day_forecasts[0]

                    temp = first.get("main", {}).get("temp", 0)
                    description = first.get("weather", [{}])[0].get("description", "")

                    return f"Weather forecast for {city}, {country} on {first_day}: {description.capitalize()}. Temperature: {temp}°C."
            else:
                # Multi-day forecast
                response_parts = [f"Weather forecast for {city}, {country}:"]

                for i, date in enumerate(sorted(daily_forecasts.keys())[:days]):
                    day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
                    day_forecasts = daily_forecasts[date]

                    # Calculate average temperature and get the most common weather description
                    avg_temp = sum(f.get("main", {}).get("temp", 0) for f in day_forecasts) / len(day_forecasts)

                    # Get the most common weather description
                    descriptions = [f.get("weather", [{}])[0].get("description", "") for f in day_forecasts]
                    most_common = max(set(descriptions), key=descriptions.count)

                    response_parts.append(f"{day_name}: {most_common.capitalize()}. Average temperature: {avg_temp:.1f}°C.")

                return "\n".join(response_parts)

        except Exception as e:
            logger.error(f"Error formatting weather response: {e}")
            return f"I found weather information for {location}, but had trouble interpreting it."

    def _generate_weather_prompt(self, location: str, days: int) -> str:
        """Generate a prompt for the LLM to create a weather forecast"""
        if days == 1:
            return f"""
            You are a weather forecasting assistant. Please provide a realistic but fictional weather forecast for {location} today.
            Include temperature, general conditions, and any other relevant weather information.
            Keep in mind that I don't have access to real-time weather data, so this is just an approximation.
            Keep your response concise and natural, as if you were a weather presenter.
            """
        else:
            return f"""
            You are a weather forecasting assistant. Please provide a realistic but fictional weather forecast for {location} for the next {days} days.
            Include expected temperatures and general conditions for each day.
            Keep in mind that I don't have access to real-time weather data, so this is just an approximation.
            Keep your response concise and natural, as if you were a weather presenter.
            """


# This import is needed for the _handle_time and _handle_date methods
# In a real implementation, you would add these to the requirements.txt file
try:
    import pytz
except ImportError:
    logger.warning("pytz module not installed, timezone conversion will not work properly")