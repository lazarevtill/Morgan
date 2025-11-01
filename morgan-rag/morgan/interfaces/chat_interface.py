"""
Enhanced chat interface with emotional awareness and personalization.

Provides a rich command-line chat experience with emotional intelligence,
relationship tracking, and companion features.

KISS: Clean interface that uses modular display and command components.
"""

import uuid
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass

from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.assistant import MorganAssistant
from ..utils.logger import get_logger
from .chat_display import ChatDisplay
from .chat_commands import ChatCommandHandler

logger = get_logger(__name__)


@dataclass
class ChatSession:
    """Represents an active chat session."""
    user_id: str
    conversation_id: str
    start_time: datetime
    message_count: int = 0
    emotional_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.emotional_context is None:
            self.emotional_context = {}


class EmotionalChatInterface:
    """
    Enhanced chat interface with emotional intelligence and companion features.
    
    Provides:
    - Real-time emotional awareness display
    - Relationship milestone celebrations
    - Personalized conversation suggestions
    - Rich visual feedback and progress tracking
    
    KISS: Clean orchestrator using modular display and command components.
    """
    
    def __init__(self, morgan_assistant: Optional[MorganAssistant] = None):
        """Initialize the emotional chat interface."""
        self.morgan = morgan_assistant or MorganAssistant()
        self.active_sessions: Dict[str, ChatSession] = {}
        
        # Initialize modular components
        self.display = ChatDisplay()
        self.command_handler = ChatCommandHandler(self.morgan, self.display)
        
        logger.info("Emotional chat interface initialized")
    
    def start_session(self, user_id: Optional[str] = None) -> str:
        """Start a new chat session with emotional awareness."""
        if not user_id:
            user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        # Create conversation with Morgan
        conversation_id = self.morgan.start_conversation(user_id=user_id)
        
        # Create session
        session = ChatSession(
            user_id=user_id,
            conversation_id=conversation_id,
            start_time=datetime.utcnow()
        )
        self.active_sessions[user_id] = session
        
        # Display welcome with personalization
        user_profile = self.morgan.relationship_manager.profiles.get(user_id)
        self.display.display_welcome(user_profile, self.morgan.name)
        
        # Show conversation suggestions
        suggestions = self.morgan.suggest_conversation_topics(user_id)
        self.display.display_conversation_suggestions(suggestions)
        
        # Show available commands
        self.display.display_quick_commands()
        
        return user_id
    
    def chat_loop(self, user_id: str):
        """Main chat loop with emotional intelligence."""
        session = self.active_sessions.get(user_id)
        if not session:
            self.display.display_error("Session not found. Please start a new session.")
            return
        
        try:
            while True:
                # Get user input
                user_profile = self.morgan.relationship_manager.profiles.get(user_id)
                user_name = user_profile.preferred_name if user_profile else "You"
                user_input = self.display.get_user_input("", user_name)
                
                if not user_input.strip():
                    continue
                
                # Check if it's a command
                if self.command_handler.is_command(user_input):
                    should_continue = self.command_handler.execute_command(
                        user_input, user_id, session.conversation_id
                    )
                    if not should_continue:
                        self._handle_goodbye(user_id)
                        break
                    continue
                
                # Process regular message with emotional intelligence
                self._process_message(user_id, user_input)
                session.message_count += 1
                
        except KeyboardInterrupt:
            self._handle_goodbye(user_id)
        except Exception as e:
            logger.error(f"Chat loop error: {e}")
            self.display.display_error(f"An error occurred: {e}")
    
    def _process_message(self, user_id: str, message: str):
        """Process user message with emotional intelligence."""
        session = self.active_sessions[user_id]
        
        # Show thinking indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.display.console,
            transient=True
        ) as progress:
            progress.add_task("🧠 Morgan is thinking...", total=None)
            
            # Get response from Morgan
            response = self.morgan.ask(
                question=message,
                conversation_id=session.conversation_id,
                user_id=user_id
            )
        
        # Display response with emotional context
        self.display.display_response(response, self.morgan.name)
        
        # Handle milestone celebrations
        if response.milestone_celebration:
            celebration_msg = self.morgan.milestone_tracker.generate_celebration_message(
                response.milestone_celebration
            )
            self.display.display_milestone_celebration(response.milestone_celebration, celebration_msg)
        
        # Update session emotional context
        if response.emotional_tone:
            session.emotional_context['last_emotion'] = response.emotional_tone
            session.emotional_context['empathy_level'] = response.empathy_level
    
    def _handle_goodbye(self, user_id: str):
        """Handle conversation goodbye."""
        user_profile = self.morgan.relationship_manager.profiles.get(user_id)
        session = self.active_sessions.get(user_id)
        
        # Generate personalized goodbye
        if user_profile and user_profile.preferred_name != user_id:
            goodbye_text = f"Goodbye, {user_profile.preferred_name}! "
        else:
            goodbye_text = "Goodbye! "
        
        if session and session.message_count > 0:
            goodbye_text += f"I enjoyed our {session.message_count} message conversation. "
        
        goodbye_text += "Feel free to chat with me anytime! 🤖💙"
        
        from rich.panel import Panel
        goodbye_panel = Panel(
            goodbye_text,
            title="👋 Until Next Time!",
            border_style="blue"
        )
        
        self.display.console.print(goodbye_panel)
        
        # Clean up session
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]


def start_emotional_chat(user_id: Optional[str] = None, morgan_assistant: Optional[MorganAssistant] = None):
    """Start an emotional chat session."""
    interface = EmotionalChatInterface(morgan_assistant)
    
    if not user_id:
        user_id = interface.start_session()
    else:
        interface.start_session(user_id)
    
    interface.chat_loop(user_id)


if __name__ == "__main__":
    # Demo the emotional chat interface
    start_emotional_chat()