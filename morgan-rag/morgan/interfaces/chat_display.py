"""
Chat display and formatting utilities for Morgan interface.

Handles rich console display, message formatting, and visual elements
following KISS principles - focused on display logic only.
"""

from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..emotional.models import CompanionProfile, RelationshipMilestone
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChatDisplay:
    """
    Handles chat display and formatting.

    KISS: Single responsibility - format and display chat elements.
    """

    def __init__(self):
        """Initialize chat display."""
        self.console = Console()

        # Display styling
        self.colors = {
            "primary": "#007bff",
            "success": "#28a745",
            "warning": "#ffc107",
            "danger": "#dc3545",
            "info": "#17a2b8",
            "light": "#f8f9fa",
            "dark": "#343a40",
        }

        logger.debug("Chat display initialized")

    def display_welcome(
        self, user_profile: Optional[CompanionProfile], morgan_name: str
    ):
        """Display personalized welcome message."""
        if user_profile and user_profile.interaction_count > 0:
            # Returning user
            welcome_text = Text()
            welcome_text.append("🤖 ", style="bold blue")
            welcome_text.append(
                f"Welcome back, {user_profile.preferred_name}!", style="bold"
            )

            # Show relationship stats
            stats_table = Table(show_header=False, box=None)
            stats_table.add_column(style="cyan")
            stats_table.add_column(style="white")

            stats_table.add_row(
                "Days together:", str(user_profile.get_relationship_age_days())
            )
            stats_table.add_row("Conversations:", str(user_profile.interaction_count))
            stats_table.add_row("Trust level:", f"{user_profile.trust_level:.1%}")
            stats_table.add_row("Engagement:", f"{user_profile.engagement_score:.1%}")

            welcome_panel = Panel(
                welcome_text,
                title="🎉 Welcome Back!",
                subtitle="Your AI Companion",
                border_style="blue",
            )

            self.console.print(welcome_panel)
            self.console.print(
                Panel(stats_table, title="📊 Relationship Stats", border_style="cyan")
            )

        else:
            # New user
            welcome_text = Text()
            welcome_text.append("🤖 Hello! I'm ", style="bold")
            welcome_text.append(morgan_name, style="bold blue")
            welcome_text.append(
                ", your emotionally intelligent AI companion.\n\n", style="bold"
            )
            welcome_text.append(
                "I'm here to help, learn, and build a meaningful relationship with you over time. ",
                style="white",
            )
            welcome_text.append(
                "I can understand emotions, remember our conversations, and adapt to your preferences.",
                style="white",
            )

            welcome_panel = Panel(
                welcome_text,
                title="👋 Welcome to Morgan!",
                subtitle="Your Emotionally Intelligent AI Companion",
                border_style="green",
            )

            self.console.print(welcome_panel)

    def display_conversation_suggestions(self, suggestions: List[str]):
        """Show conversation topic suggestions."""
        if not suggestions:
            return

        suggestions_table = Table(show_header=False, box=None, padding=(0, 1))
        suggestions_table.add_column("💡", style="yellow")
        suggestions_table.add_column("Suggestion", style="white")

        for i, suggestion in enumerate(suggestions[:3], 1):
            suggestions_table.add_row(f"{i}.", suggestion)

        suggestions_panel = Panel(
            suggestions_table,
            title="💡 Conversation Suggestions",
            border_style="yellow",
        )

        self.console.print(suggestions_panel)

    def display_quick_commands(self):
        """Show available quick commands."""
        commands_table = Table(show_header=False, box=None)
        commands_table.add_column("Command", style="cyan")
        commands_table.add_column("Description", style="white")

        commands = [
            ("help", "Show this help message"),
            ("profile", "View your relationship profile"),
            ("timeline", "View relationship timeline"),
            ("preferences", "Manage your preferences"),
            ("feedback", "Provide feedback"),
            ("quit", "End the conversation"),
        ]

        for cmd, desc in commands:
            commands_table.add_row(f"'{cmd}'", desc)

        commands_panel = Panel(
            commands_table, title="⚡ Quick Commands", border_style="dim"
        )

        self.console.print(commands_panel)

    def display_response(self, response, morgan_name: str):
        """Display Morgan's response with emotional context."""
        # Create response text with emotional indicators
        response_text = Text()
        response_text.append("🤖 ", style="bold green")
        response_text.append(morgan_name, style="bold green")

        # Add emotional tone indicator
        if response.emotional_tone:
            response_text.append(f" ({response.emotional_tone})", style="italic dim")

        response_text.append(":\n", style="bold green")
        response_text.append(response.answer, style="white")

        # Create panel with emotional context
        panel_style = "green"
        if response.empathy_level and response.empathy_level > 0.7:
            panel_style = "magenta"  # High empathy
        elif response.confidence and response.confidence > 0.8:
            panel_style = "blue"  # High confidence

        subtitle = ""
        if response.empathy_level:
            subtitle += f"Empathy: {response.empathy_level:.1%}"
        if response.confidence:
            if subtitle:
                subtitle += " • "
            subtitle += f"Confidence: {response.confidence:.1%}"

        response_panel = Panel(
            response_text,
            title="💬 Response",
            subtitle=subtitle,
            border_style=panel_style,
        )

        self.console.print(response_panel)

        # Show personalization elements if any
        if response.personalization_elements:
            personalization_text = "🎯 " + " • ".join(response.personalization_elements)
            self.console.print(personalization_text, style="dim cyan")

        # Show sources if available
        if response.sources:
            sources_text = "📚 Sources: " + " • ".join(response.sources[:3])
            if len(response.sources) > 3:
                sources_text += f" (+{len(response.sources) - 3} more)"
            self.console.print(sources_text, style="dim blue")

    def display_milestone_celebration(
        self, milestone: RelationshipMilestone, celebration_message: str
    ):
        """Celebrate a relationship milestone."""
        celebration_text = Text()
        celebration_text.append("🎉 ", style="bold yellow")
        celebration_text.append("Milestone Achieved!", style="bold yellow")
        celebration_text.append("\n\n", style="white")
        celebration_text.append(celebration_message, style="white")
        celebration_text.append("\n\n", style="white")
        celebration_text.append("Emotional significance: ", style="dim")
        celebration_text.append(f"{milestone.emotional_significance:.1%}", style="bold")

        celebration_panel = Panel(
            celebration_text,
            title="🏆 Relationship Milestone",
            border_style="yellow",
            padding=(1, 2),
        )

        self.console.print(celebration_panel)

        # Animate celebration
        import time

        for i in range(3):
            self.console.print("✨", end="", style="bold yellow")
            time.sleep(0.3)
        self.console.print()

    def display_help(self, morgan_name: str):
        """Show detailed help information."""
        help_text = Text()
        help_text.append(
            f"🤖 {morgan_name} - Emotionally Intelligent AI Companion\n\n",
            style="bold blue",
        )
        help_text.append(
            "I'm designed to understand emotions, remember our conversations, ",
            style="white",
        )
        help_text.append(
            "and build a meaningful relationship with you over time.\n\n", style="white"
        )

        help_text.append("Features:\n", style="bold")
        help_text.append("• Emotional intelligence and empathy\n", style="white")
        help_text.append("• Conversation memory and context\n", style="white")
        help_text.append("• Relationship milestone tracking\n", style="white")
        help_text.append("• Personalized responses and suggestions\n", style="white")
        help_text.append("• Preference learning and adaptation\n", style="white")

        help_panel = Panel(
            help_text, title="❓ Help & Information", border_style="blue"
        )

        self.console.print(help_panel)

    def display_profile(self, user_profile: CompanionProfile):
        """Show user relationship profile."""
        if not user_profile:
            self.console.print(
                "📝 No profile found. Start chatting to build your profile!",
                style="yellow",
            )
            return

        # Create profile display
        profile_table = Table(show_header=False, box=None)
        profile_table.add_column("Attribute", style="cyan")
        profile_table.add_column("Value", style="white")

        profile_data = [
            ("Preferred Name", user_profile.preferred_name),
            ("Relationship Age", f"{user_profile.get_relationship_age_days()} days"),
            ("Conversations", str(user_profile.interaction_count)),
            ("Trust Level", f"{user_profile.trust_level:.1%}"),
            ("Engagement Score", f"{user_profile.engagement_score:.1%}"),
            (
                "Communication Style",
                user_profile.communication_preferences.communication_style.value.title(),
            ),
            (
                "Response Length",
                user_profile.communication_preferences.preferred_response_length.value.title(),
            ),
        ]

        for attr, value in profile_data:
            profile_table.add_row(attr, value)

        # Show interests
        if user_profile.communication_preferences.topics_of_interest:
            interests = ", ".join(
                user_profile.communication_preferences.topics_of_interest[:5]
            )
            if len(user_profile.communication_preferences.topics_of_interest) > 5:
                interests += f" (+{len(user_profile.communication_preferences.topics_of_interest) - 5} more)"
            profile_table.add_row("Interests", interests)

        profile_panel = Panel(
            profile_table, title="👤 Your Profile", border_style="cyan"
        )

        self.console.print(profile_panel)

    def display_timeline(self, user_profile: CompanionProfile):
        """Show relationship timeline."""
        if not user_profile or not user_profile.relationship_milestones:
            self.console.print(
                "📈 No milestones yet. Keep chatting to create memorable moments!",
                style="yellow",
            )
            return

        # Display milestones
        timeline_table = Table(show_header=True, box=None)
        timeline_table.add_column("Date", style="cyan")
        timeline_table.add_column("Milestone", style="white")
        timeline_table.add_column("Significance", style="yellow")

        for milestone in user_profile.relationship_milestones[
            -10:
        ]:  # Last 10 milestones
            date_str = milestone.timestamp.strftime("%Y-%m-%d %H:%M")
            milestone_type = milestone.milestone_type.value.replace("_", " ").title()
            significance = f"{milestone.emotional_significance:.1%}"

            timeline_table.add_row(date_str, milestone_type, significance)

        timeline_panel = Panel(
            timeline_table, title="📈 Relationship Timeline", border_style="green"
        )

        self.console.print(timeline_panel)

    def display_error(self, error_message: str):
        """Display error message."""
        error_panel = Panel(f"❌ {error_message}", title="Error", border_style="red")
        self.console.print(error_panel)

    def display_success(self, success_message: str):
        """Display success message."""
        self.console.print(f"✅ {success_message}", style="green")

    def display_info(self, info_message: str):
        """Display info message."""
        self.console.print(f"ℹ️ {info_message}", style="blue")

    def get_user_input(self, prompt: str, user_name: str = "You") -> str:
        """Get user input with rich prompt."""
        from rich.prompt import Prompt

        return Prompt.ask(f"[bold cyan]{user_name}[/bold cyan]", default="")

    def confirm(self, message: str) -> bool:
        """Get user confirmation."""
        from rich.prompt import Confirm

        return Confirm.ask(message)
