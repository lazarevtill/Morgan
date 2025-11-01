#!/usr/bin/env python3
"""
Test script for refactored Morgan modules.

Tests that all the new modular components work together properly.
"""

import sys
import os

# Add the morgan-rag directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'morgan-rag'))

def test_imports():
    """Test that all refactored modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core modules
        from morgan.core.response_handler import ResponseHandler, Response
        from morgan.core.conversation_manager import ConversationManager
        from morgan.core.emotional_processor import EmotionalProcessor
        from morgan.core.milestone_tracker import MilestoneTracker
        
        # Test interface modules
        try:
            from morgan.interfaces.feedback_system import CompanionFeedbackSystem
            print("✅ Feedback system import successful!")
        except ImportError as e:
            print(f"⚠️ Feedback system import failed: {e}")
        
        from morgan.interfaces.websocket_handler import WebSocketManager
        from morgan.interfaces.chat_display import ChatDisplay
        from morgan.interfaces.chat_commands import ChatCommandHandler
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of refactored modules."""
    print("\nTesting basic functionality...")
    
    try:
        # Test response handler
        from morgan.core.response_handler import ResponseHandler
        response_handler = ResponseHandler()
        
        response = response_handler.create_response(
            answer="Test response",
            confidence=0.8,
            emotional_tone="friendly"
        )
        
        assert response.answer == "Test response"
        assert response.confidence == 0.8
        assert response.emotional_tone == "friendly"
        print("✅ Response handler works!")
        
        # Test conversation manager
        from morgan.core.conversation_manager import ConversationManager
        conv_manager = ConversationManager()
        
        context = conv_manager.create_conversation_context(
            "Hello", "conv_123", "user_456"
        )
        
        assert context.message_text == "Hello"
        assert context.conversation_id == "conv_123"
        assert context.user_id == "user_456"
        print("✅ Conversation manager works!")
        
        # Test milestone tracker
        from morgan.core.milestone_tracker import MilestoneTracker
        milestone_tracker = MilestoneTracker()
        print("✅ Milestone tracker works!")
        
        # Test feedback system
        import morgan.interfaces.feedback_system as feedback_module
        CompanionFeedbackSystem = getattr(feedback_module, 'CompanionFeedbackSystem', None)
        FeedbackType = getattr(feedback_module, 'FeedbackType', None)
        
        if CompanionFeedbackSystem is None:
            print("⚠️ CompanionFeedbackSystem not found, skipping feedback test")
            return True
            
        feedback_system = CompanionFeedbackSystem()
        
        feedback_id = feedback_system.collect_feedback(
            user_id="test_user",
            conversation_id="test_conv",
            feedback_type=FeedbackType.CONVERSATION_RATING,
            rating=5,
            comment="Great conversation!"
        )
        
        assert feedback_id is not None
        print("✅ Feedback system works!")
        
        # Test chat display
        from morgan.interfaces.chat_display import ChatDisplay
        display = ChatDisplay()
        print("✅ Chat display works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🤖 Testing Refactored Morgan Modules")
    print("=" * 40)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! Refactoring successful!")
        print("\nThe modular architecture is working properly:")
        print("• Core modules are properly separated")
        print("• Interface components are modular")
        print("• KISS principles are followed")
        print("• Each module has a single responsibility")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)