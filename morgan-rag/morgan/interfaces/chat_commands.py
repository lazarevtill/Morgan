"""
Chat command handlers for Morgan interface.

Handles special commands like preferences, profile, timeline, etc.
following KISS principles - focused on command processing only.
"""

from typing import Optional, Dict, Any
from datetime import datetime

from ..core.assistant import MorganAssistant
from ..emotional.models import CompanionProfile, CommunicationStyle, ResponseLength
from ..utils.logger import get_logger
from .chat_display import ChatDisplay

logger = get_logger(__name__)


class ChatCommandHandler:
    """
    Handles chat commands and special interactions.
    
    KISS: Single responsibility - process chat commands.
    """
    
    def __init__(self, morgan: MorganAssistant, display: ChatDisplay):
        """Initialize command handler."""
        self.morgan = morgan
        self.display = display
        logger.debug("Chat command handler initialized")
    
    def handle_help_command(self):
        """Handle help command."""
        self.display.display_help(self.morgan.name)
    
    def handle_profile_command(self, user_id: str):
        """Handle profile command."""
        user_profile = self.morgan.relationship_manager.profiles.get(user_id)
        self.display.display_profile(user_profile)
    
    def handle_timeline_command(self, user_id: str):
        """Handle timeline command."""
        user_profile = self.morgan.relationship_manager.profiles.get(user_id)
        self.display.display_timeline(user_profile)
    
    def handle_preferences_command(self, user_id: str):
        """Interactive preference management."""
        user_profile = self.morgan.relationship_manager.profiles.get(user_id)
        
        if not user_profile:
            self.display.display_error("No profile found. Preferences will be created after our first conversation.")
            return
        
        from rich.panel import Panel
        self.display.console.print(Panel("⚙️ Preference Management", border_style="blue"))
        
        # Show current preferences
        from rich.table import Table
        current_prefs = Table(show_header=False, box=None)
        current_prefs.add_column("Setting", style="cyan")
        current_prefs.add_column("Current Value", style="white")
        
        prefs = user_profile.communication_preferences
        current_prefs.add_row("Preferred Name", user_profile.preferred_name)
        current_prefs.add_row("Communication Style", prefs.communication_style.value.title())
        current_prefs.add_row("Response Length", prefs.preferred_response_length.value.title())
        
        self.display.console.print(current_prefs)
        
        # Ask if user wants to update
        if self.display.confirm("\nWould you like to update your preferences?"):
            self._update_preferences_interactive(user_profile)
    
    def handle_feedback_command(self, user_id: str, conversation_id: str):
        """Collect user feedback."""
        from rich.panel import Panel
        self.display.console.print(Panel("💭 Feedback Collection", border_style="blue"))
        
        # Get rating
        rating = None
        while rating is None:
            try:
                from rich.prompt import Prompt
                rating_input = Prompt.ask("Rate our conversation (1-5 stars)", default="5")
                rating = int(rating_input)
                if not 1 <= rating <= 5:
                    raise ValueError
            except ValueError:
                self.display.display_error("Please enter a number between 1 and 5")
        
        # Get optional comment
        from rich.prompt import Prompt
        comment = Prompt.ask("Any additional comments? (optional)", default="")
        
        # Submit feedback
        success = self.morgan.provide_feedback(
            conversation_id=conversation_id,
            rating=rating,
            comment=comment if comment else None,
            user_id=user_id
        )
        
        if success:
            self.display.display_success("Thank you for your feedback! 😊")
        else:
            self.display.display_error("Failed to submit feedback")
    
    def handle_stats_command(self, user_id: str):
        """Show user statistics and insights."""
        try:
            insights = self.morgan.get_relationship_insights(user_id)
            milestone_stats = self.morgan.get_milestone_statistics(user_id)
            
            from rich.table import Table
            from rich.panel import Panel
            
            # Create stats table
            stats_table = Table(show_header=False, box=None)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            if isinstance(insights, dict) and "error" not in insights:
                stats_table.add_row("Relationship Age", f"{insights.get('relationship_age_days', 0)} days")
                stats_table.add_row("Total Conversations", str(insights.get('interaction_count', 0)))
                stats_table.add_row("Trust Level", f"{insights.get('trust_level', 0):.1%}")
                stats_table.add_row("Engagement Score", f"{insights.get('engagement_score', 0):.1%}")
                
                if insights.get('dominant_emotions'):
                    emotions = ", ".join(insights['dominant_emotions'][:3])
                    stats_table.add_row("Dominant Emotions", emotions)
            
            if isinstance(milestone_stats, dict) and "error" not in milestone_stats:
                stats_table.add_row("Total Milestones", str(milestone_stats.get('total_milestones', 0)))
                
                if milestone_stats.get('milestone_types'):
                    types = ", ".join(milestone_stats['milestone_types'][:3])
                    stats_table.add_row("Milestone Types", types)
            
            stats_panel = Panel(
                stats_table,
                title="📊 Your Statistics",
                border_style="blue"
            )
            
            self.display.console.print(stats_panel)
            
        except Exception as e:
            logger.error(f"Error showing stats: {e}")
            self.display.display_error("Unable to retrieve statistics")
    
    def handle_topics_command(self, user_id: str):
        """Show and manage conversation topics."""
        try:
            suggestions = self.morgan.suggest_conversation_topics(user_id)
            
            from rich.table import Table
            from rich.panel import Panel
            
            topics_table = Table(show_header=False, box=None)
            topics_table.add_column("💡", style="yellow")
            topics_table.add_column("Topic", style="white")
            
            for i, topic in enumerate(suggestions[:5], 1):
                topics_table.add_row(f"{i}.", topic)
            
            topics_panel = Panel(
                topics_table,
                title="💡 Suggested Topics",
                border_style="yellow"
            )
            
            self.display.console.print(topics_panel)
            
        except Exception as e:
            logger.error(f"Error showing topics: {e}")
            self.display.display_error("Unable to retrieve topic suggestions")
    
    def _update_preferences_interactive(self, user_profile: CompanionProfile):
        """Interactive preference update process."""
        from rich.prompt import Prompt
        
        # Update preferred name
        new_name = Prompt.ask("Preferred name", default=user_profile.preferred_name)
        if new_name != user_profile.preferred_name:
            user_profile.preferred_name = new_name
            self.display.display_success(f"Updated preferred name to '{new_name}'")
        
        # Update communication style
        style_options = ["friendly", "casual", "professional", "technical", "formal"]
        current_style = user_profile.communication_preferences.communication_style.value
        
        self.display.console.print("\nCommunication styles:")
        for i, style in enumerate(style_options, 1):
            marker = "→" if style == current_style else " "
            self.display.console.print(f"{marker} {i}. {style.title()}")
        
        style_choice = Prompt.ask("Choose communication style (1-5)", default=str(style_options.index(current_style) + 1))
        try:
            new_style_index = int(style_choice) - 1
            if 0 <= new_style_index < len(style_options):
                user_profile.communication_preferences.communication_style = CommunicationStyle(style_options[new_style_index])
                self.display.display_success(f"Updated communication style to '{style_options[new_style_index]}'")
        except (ValueError, IndexError):
            self.display.display_error("Invalid choice")
        
        # Update response length preference
        length_options = ["brief", "detailed", "comprehensive"]
        current_length = user_profile.communication_preferences.preferred_response_length.value
        
        self.display.console.print("\nResponse length preferences:")
        for i, length in enumerate(length_options, 1):
            marker = "→" if length == current_length else " "
            self.display.console.print(f"{marker} {i}. {length.title()}")
        
        length_choice = Prompt.ask("Choose response length (1-3)", default=str(length_options.index(current_length) + 1))
        try:
            new_length_index = int(length_choice) - 1
            if 0 <= new_length_index < len(length_options):
                user_profile.communication_preferences.preferred_response_length = ResponseLength(length_options[new_length_index])
                self.display.display_success(f"Updated response length to '{length_options[new_length_index]}'")
        except (ValueError, IndexError):
            self.display.display_error("Invalid choice")
        
        # Update topics of interest
        current_topics = ", ".join(user_profile.communication_preferences.topics_of_interest)
        self.display.console.print(f"\nCurrent topics of interest: {current_topics}")
        
        if self.display.confirm("Would you like to add new topics of interest?"):
            new_topics_input = Prompt.ask("Enter topics separated by commas", default="")
            if new_topics_input.strip():
                new_topics = [topic.strip() for topic in new_topics_input.split(",")]
                # Add new topics to existing ones (avoid duplicates)
                for topic in new_topics:
                    if topic and topic not in user_profile.communication_preferences.topics_of_interest:
                        user_profile.communication_preferences.topics_of_interest.append(topic)
                
                self.display.display_success(f"Added new topics: {', '.join(new_topics)}")
        
        user_profile.communication_preferences.last_updated = datetime.utcnow()
        self.display.display_success("\n✅ Preferences updated successfully!")
    
    def get_available_commands(self) -> Dict[str, str]:
        """Get list of available commands."""
        return {
            "help": "Show help information",
            "profile": "View your relationship profile",
            "timeline": "View relationship timeline",
            "preferences": "Manage your preferences",
            "feedback": "Provide feedback",
            "stats": "Show statistics and insights",
            "topics": "Show conversation topic suggestions",
            "quit": "End the conversation"
        }
    
    def is_command(self, user_input: str) -> bool:
        """Check if user input is a command."""
        commands = self.get_available_commands().keys()
        return user_input.lower().strip() in commands
    
    def execute_command(self, command: str, user_id: str, conversation_id: str) -> bool:
        """
        Execute a command.
        
        Returns:
            True if command was executed, False if not recognized
        """
        command = command.lower().strip()
        
        try:
            if command == "help":
                self.handle_help_command()
            elif command == "profile":
                self.handle_profile_command(user_id)
            elif command == "timeline":
                self.handle_timeline_command(user_id)
            elif command == "preferences":
                self.handle_preferences_command(user_id)
            elif command == "feedback":
                self.handle_feedback_command(user_id, conversation_id)
            elif command == "stats":
                self.handle_stats_command(user_id)
            elif command == "topics":
                self.handle_topics_command(user_id)
            elif command in ["quit", "exit", "bye"]:
                return False  # Signal to quit
            else:
                return False  # Command not recognized
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            self.display.display_error(f"Error executing command: {e}")
            return True  # Command was recognized but failed