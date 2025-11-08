"""
CLI Output Formatters.

Provides rich formatting for CLI output using the rich library.
Includes formatters for:
- Messages and conversations
- Assistant responses
- Health checks
- Metrics and statistics
- Emotions
- Sources
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from morgan.core.types import (
    AssistantMetrics,
    AssistantResponse,
    Message,
    MessageRole,
    SearchSource,
)
from morgan.emotions.types import EmotionResult
from morgan.learning.types import LearningMetrics, LearningPattern, UserPreference


# Try to import rich, fallback to plain text if not available
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ConsoleFormatter:
    """Rich console formatter for Morgan CLI."""

    def __init__(self, use_rich: bool = True):
        """
        Initialize formatter.

        Args:
            use_rich: Use rich formatting (falls back to plain if unavailable)
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None

    def print(self, text: str, **kwargs) -> None:
        """Print text with optional rich formatting."""
        if self.console:
            self.console.print(text, **kwargs)
        else:
            print(text)

    def format_message(self, message: Message) -> str:
        """
        Format a message with role, timestamp, and optional emotion.

        Args:
            message: Message to format

        Returns:
            Formatted message string
        """
        if not self.use_rich:
            # Plain text format
            role_prefix = {
                MessageRole.USER: "You:",
                MessageRole.ASSISTANT: "Morgan:",
                MessageRole.SYSTEM: "System:",
            }
            prefix = role_prefix.get(message.role, "Unknown:")

            timestamp = message.timestamp.strftime("%H:%M:%S")
            output = f"[{timestamp}] {prefix} {message.content}"

            if message.emotion:
                output += f"\n  (Emotion: {message.emotion.primary_emotion}, intensity: {message.emotion.intensity:.2f})"

            return output

        # Rich format
        role_colors = {
            MessageRole.USER: "cyan",
            MessageRole.ASSISTANT: "green",
            MessageRole.SYSTEM: "yellow",
        }
        role_icons = {
            MessageRole.USER: "👤",
            MessageRole.ASSISTANT: "🤖",
            MessageRole.SYSTEM: "⚙️",
        }

        color = role_colors.get(message.role, "white")
        icon = role_icons.get(message.role, "•")
        timestamp = message.timestamp.strftime("%H:%M:%S")

        text = Text()
        text.append(f"{icon} ", style="bold")
        text.append(f"{message.role.value.capitalize()}", style=f"bold {color}")
        text.append(f" [{timestamp}]", style="dim")
        text.append("\n")
        text.append(message.content)

        if message.emotion:
            text.append("\n")
            text.append(self._format_emotion_inline(message.emotion), style="dim italic")

        return text

    def format_response(
        self,
        response: AssistantResponse,
        show_sources: bool = True,
        show_metrics: bool = False,
    ) -> str:
        """
        Format assistant response with citations and metadata.

        Args:
            response: Assistant response
            show_sources: Show RAG sources
            show_metrics: Show performance metrics

        Returns:
            Formatted response
        """
        if not self.use_rich:
            # Plain text format
            output = f"\nMorgan: {response.content}\n"

            if response.emotion:
                output += f"\n  Emotion: {response.emotion.primary_emotion} (intensity: {response.emotion.intensity:.2f})\n"

            if show_sources and response.sources:
                output += f"\n  Sources ({len(response.sources)}):\n"
                for i, source in enumerate(response.sources[:3], 1):
                    output += f"    {i}. {source.source} (score: {source.score:.2f})\n"

            if show_metrics:
                output += f"\n  Response time: {response.generation_time_ms:.0f}ms\n"
                output += f"  Confidence: {response.confidence:.2f}\n"

            return output

        # Rich format
        # Main content
        content_panel = Panel(
            Markdown(response.content),
            title="🤖 Morgan",
            border_style="green",
            padding=(1, 2),
        )

        # Emotion indicator
        emotion_text = None
        if response.emotion:
            emotion_text = self._format_emotion_panel(response.emotion)

        # Sources
        sources_text = None
        if show_sources and response.sources:
            sources_text = self._format_sources(response.sources)

        # Metrics
        metrics_text = None
        if show_metrics:
            metrics_data = {
                "Response Time": f"{response.generation_time_ms:.0f}ms",
                "Confidence": f"{response.confidence:.2%}",
                "Tokens": str(response.total_tokens) if response.total_tokens else "N/A",
            }
            metrics_text = self._format_metadata(metrics_data)

        # Combine all parts
        parts = [content_panel]
        if emotion_text:
            parts.append(emotion_text)
        if sources_text:
            parts.append(sources_text)
        if metrics_text:
            parts.append(metrics_text)

        return parts

    def _format_emotion_inline(self, emotion: EmotionResult) -> str:
        """Format emotion as inline text."""
        emotion_icons = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😠",
            "fear": "😨",
            "surprise": "😲",
            "disgust": "😖",
            "neutral": "😐",
            "love": "❤️",
            "curiosity": "🤔",
            "confusion": "😕",
        }

        icon = emotion_icons.get(emotion.primary_emotion, "•")
        return f"{icon} {emotion.primary_emotion} (intensity: {emotion.intensity:.2f})"

    def _format_emotion_panel(self, emotion: EmotionResult) -> Panel:
        """Format emotion as a rich panel."""
        emotion_colors = {
            "joy": "bright_yellow",
            "sadness": "blue",
            "anger": "red",
            "fear": "magenta",
            "surprise": "cyan",
            "disgust": "green",
            "neutral": "white",
            "love": "bright_magenta",
            "curiosity": "bright_cyan",
            "confusion": "yellow",
        }

        color = emotion_colors.get(emotion.primary_emotion, "white")

        text = Text()
        text.append(f"{emotion.primary_emotion.upper()}", style=f"bold {color}")
        text.append(f" • Intensity: {emotion.intensity:.2f}\n", style="dim")

        if emotion.emotions:
            text.append("\nAll emotions:\n", style="dim")
            for emo, score in list(emotion.emotions.items())[:3]:
                bar = "█" * int(score * 10)
                text.append(f"  {emo}: {bar} {score:.2f}\n", style="dim")

        return Panel(
            text,
            title="💭 Emotion",
            border_style=color,
            padding=(0, 1),
        )

    def _format_sources(self, sources: List[SearchSource]) -> Panel:
        """Format RAG sources as a panel."""
        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
        table.add_column("#", style="dim", width=3)
        table.add_column("Source", style="cyan")
        table.add_column("Score", justify="right", style="green")

        for i, source in enumerate(sources[:5], 1):
            # Truncate source name
            source_name = source.source
            if len(source_name) > 50:
                source_name = source_name[:47] + "..."

            table.add_row(
                str(i),
                source_name,
                f"{source.score:.3f}",
            )

        return Panel(
            table,
            title=f"📚 Sources ({len(sources)})",
            border_style="cyan",
            padding=(0, 1),
        )

    def _format_metadata(self, metadata: Dict[str, str]) -> Panel:
        """Format metadata as a panel."""
        text = Text()
        for key, value in metadata.items():
            text.append(f"{key}: ", style="dim")
            text.append(f"{value}\n")

        return Panel(
            text,
            title="📊 Metrics",
            border_style="blue",
            padding=(0, 1),
        )

    def format_health(self, health_status: Dict[str, Any]) -> Any:
        """
        Format health check as table.

        Args:
            health_status: Health status dictionary

        Returns:
            Formatted health status
        """
        if not self.use_rich:
            # Plain text format
            output = "System Health Check:\n"
            output += "=" * 40 + "\n"
            for service, status in health_status.items():
                status_text = "✓" if status.get("healthy", False) else "✗"
                output += f"  {status_text} {service}: {status.get('status', 'unknown')}\n"
                if latency := status.get("latency_ms"):
                    output += f"      Latency: {latency:.0f}ms\n"
            return output

        # Rich format
        table = Table(title="System Health Check", show_header=True, header_style="bold cyan")
        table.add_column("Service", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Latency", justify="right", style="yellow")
        table.add_column("Details", style="dim")

        for service, status in health_status.items():
            healthy = status.get("healthy", False)
            status_icon = "✅" if healthy else "❌"
            status_style = "green" if healthy else "red"

            latency = status.get("latency_ms", 0)
            latency_text = f"{latency:.0f}ms" if latency else "-"

            details = status.get("message", "")

            table.add_row(
                service,
                Text(status_icon, style=status_style),
                latency_text,
                details,
            )

        return table

    def format_learning_stats(self, stats: LearningMetrics) -> Any:
        """
        Format learning statistics.

        Args:
            stats: Learning metrics

        Returns:
            Formatted statistics
        """
        if not self.use_rich:
            # Plain text format
            output = "Learning Statistics:\n"
            output += "=" * 40 + "\n"
            output += f"  Patterns Detected: {stats.patterns_detected}\n"
            output += f"  Feedback Processed: {stats.feedback_processed}\n"
            output += f"  Preferences Learned: {stats.preferences_learned}\n"
            output += f"  Adaptations Made: {stats.adaptations_made}\n"
            output += f"  Avg Confidence: {stats.avg_confidence:.2f}\n"
            return output

        # Rich format
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        metrics = [
            ("Patterns Detected", str(stats.patterns_detected)),
            ("Feedback Processed", str(stats.feedback_processed)),
            ("Preferences Learned", str(stats.preferences_learned)),
            ("Adaptations Made", str(stats.adaptations_made)),
            ("Consolidations", str(stats.consolidations_performed)),
            ("Avg Confidence", f"{stats.avg_confidence:.2%}"),
        ]

        for metric, value in metrics:
            table.add_row(metric, value)

        return Panel(
            table,
            title="📈 Learning Statistics",
            border_style="cyan",
            padding=(1, 2),
        )

    def format_preferences(self, preferences: List[UserPreference]) -> Any:
        """
        Format user preferences.

        Args:
            preferences: List of user preferences

        Returns:
            Formatted preferences
        """
        if not self.use_rich:
            output = "User Preferences:\n"
            output += "=" * 40 + "\n"
            for pref in preferences[:10]:
                output += f"  • {pref.dimension.value}: {pref.value} (confidence: {pref.confidence:.2f})\n"
            return output

        # Rich format
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Dimension", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Confidence", justify="right", style="yellow")

        for pref in preferences[:10]:
            table.add_row(
                pref.dimension.value,
                str(pref.value),
                f"{pref.confidence:.2%}",
            )

        return Panel(
            table,
            title="⚙️ User Preferences",
            border_style="cyan",
            padding=(1, 2),
        )

    def format_patterns(self, patterns: List[LearningPattern]) -> Any:
        """
        Format learning patterns.

        Args:
            patterns: List of learning patterns

        Returns:
            Formatted patterns
        """
        if not self.use_rich:
            output = "Detected Patterns:\n"
            output += "=" * 40 + "\n"
            for pattern in patterns[:10]:
                output += f"  • {pattern.pattern_type.value}: {pattern.description}\n"
                output += f"      Confidence: {pattern.confidence:.2f}, Occurrences: {pattern.occurrences}\n"
            return output

        # Rich format
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Type", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Confidence", justify="right", style="yellow")
        table.add_column("Count", justify="right", style="green")

        for pattern in patterns[:10]:
            table.add_row(
                pattern.pattern_type.value,
                pattern.description[:50] + "..." if len(pattern.description) > 50 else pattern.description,
                f"{pattern.confidence:.2%}",
                str(pattern.occurrences),
            )

        return Panel(
            table,
            title="🔍 Detected Patterns",
            border_style="cyan",
            padding=(1, 2),
        )

    def format_metrics_detailed(self, metrics: AssistantMetrics) -> Any:
        """
        Format detailed assistant metrics.

        Args:
            metrics: Assistant metrics

        Returns:
            Formatted metrics
        """
        if not self.use_rich:
            output = "Performance Metrics:\n"
            output += "=" * 40 + "\n"
            output += f"  Total Duration: {metrics.total_duration_ms:.0f}ms\n"
            output += f"  Emotion Detection: {metrics.emotion_detection_ms:.0f}ms\n"
            output += f"  Memory Retrieval: {metrics.memory_retrieval_ms:.0f}ms\n"
            output += f"  RAG Search: {metrics.rag_search_ms:.0f}ms\n"
            output += f"  Context Building: {metrics.context_building_ms:.0f}ms\n"
            output += f"  Response Generation: {metrics.response_generation_ms:.0f}ms\n"
            output += f"  Learning Update: {metrics.learning_update_ms:.0f}ms\n"
            output += f"\n  Messages Retrieved: {metrics.messages_retrieved}\n"
            output += f"  RAG Sources: {metrics.rag_sources_found}\n"
            output += f"  Cache Hit: {metrics.used_cache}\n"
            output += f"  Degraded Mode: {metrics.degraded_mode}\n"
            return output

        # Rich format - create two tables side by side
        timing_table = Table(title="⏱️ Timing Breakdown", show_header=True, header_style="bold yellow", box=None)
        timing_table.add_column("Stage", style="cyan")
        timing_table.add_column("Duration", justify="right", style="yellow")

        timing_data = [
            ("Emotion Detection", f"{metrics.emotion_detection_ms:.0f}ms"),
            ("Memory Retrieval", f"{metrics.memory_retrieval_ms:.0f}ms"),
            ("RAG Search", f"{metrics.rag_search_ms:.0f}ms"),
            ("Context Building", f"{metrics.context_building_ms:.0f}ms"),
            ("Response Generation", f"{metrics.response_generation_ms:.0f}ms"),
            ("Learning Update", f"{metrics.learning_update_ms:.0f}ms"),
            ("", ""),
            ("Total", f"{metrics.total_duration_ms:.0f}ms"),
        ]

        for stage, duration in timing_data:
            style = "bold green" if stage == "Total" else None
            timing_table.add_row(stage, duration, style=style)

        stats_table = Table(title="📊 Statistics", show_header=False, box=None)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")

        stats_data = [
            ("Messages Retrieved", str(metrics.messages_retrieved)),
            ("RAG Sources", str(metrics.rag_sources_found)),
            ("Cache Hit", "✅" if metrics.used_cache else "❌"),
            ("Degraded Mode", "⚠️" if metrics.degraded_mode else "✅"),
        ]

        for metric, value in stats_data:
            stats_table.add_row(metric, value)

        return [timing_table, stats_table]

    def format_error(self, error: Exception, verbose: bool = False) -> str:
        """
        Format error message.

        Args:
            error: Exception to format
            verbose: Include traceback

        Returns:
            Formatted error message
        """
        if not self.use_rich:
            output = f"❌ Error: {str(error)}\n"
            if verbose:
                import traceback
                output += "\n" + traceback.format_exc()
            return output

        # Rich format
        text = Text()
        text.append("❌ Error: ", style="bold red")
        text.append(str(error), style="red")

        if verbose:
            import traceback
            text.append("\n\nTraceback:\n", style="dim")
            text.append(traceback.format_exc(), style="dim red")

        return Panel(text, border_style="red", padding=(1, 2))

    def create_progress(self, description: str = "Processing...") -> Optional[Progress]:
        """
        Create a progress indicator.

        Args:
            description: Progress description

        Returns:
            Progress object if rich is available, None otherwise
        """
        if not self.use_rich:
            return None

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        )
