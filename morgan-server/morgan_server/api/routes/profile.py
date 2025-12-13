"""
Profile API Routes

This module implements the profile endpoints for Morgan server:
- GET /api/profile/{user_id}: Get user profile
- PUT /api/profile/{user_id}: Update user preferences
- GET /api/timeline/{user_id}: Get relationship timeline
"""

from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Path
from fastapi.responses import JSONResponse

from morgan_server.api.models import (
    ProfileResponse,
    PreferenceUpdate,
    TimelineResponse,
    TimelineEvent,
    ErrorResponse,
)
from morgan_server.personalization.profile import ProfileManager, CommunicationStyle, ResponseLength


router = APIRouter(prefix="/api", tags=["profile"])


# Global profile manager instance (will be injected via dependency injection)
_profile_manager: Optional[ProfileManager] = None


def set_profile_manager(profile_manager: ProfileManager) -> None:
    """
    Set the global profile manager instance.
    
    This should be called during application startup.
    
    Args:
        profile_manager: ProfileManager instance
    """
    global _profile_manager
    _profile_manager = profile_manager


def get_profile_manager() -> ProfileManager:
    """
    Get the global profile manager instance.
    
    Returns:
        ProfileManager instance
        
    Raises:
        HTTPException: If profile manager is not initialized
    """
    if _profile_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Profile manager not initialized",
        )
    return _profile_manager


@router.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_user_profile(
    user_id: str = Path(..., description="User identifier")
) -> ProfileResponse:
    """
    Get user profile.
    
    Retrieves the profile for the specified user, including preferences,
    metrics, and relationship information.
    
    Args:
        user_id: User identifier
        
    Returns:
        ProfileResponse with user profile data
        
    Raises:
        HTTPException: If profile manager is not initialized or user not found
    """
    try:
        profile_manager = get_profile_manager()
        
        # Validate user_id
        if not user_id or not user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id cannot be empty",
            )
        
        # Get or create profile
        profile = profile_manager.get_or_create_profile(user_id)
        
        # Calculate relationship age
        relationship_age = profile_manager.calculate_relationship_age(user_id)
        
        # Convert to API model
        return ProfileResponse(
            user_id=profile.user_id,
            preferred_name=profile.preferred_name,
            relationship_age_days=relationship_age,
            interaction_count=profile.interaction_count,
            trust_level=profile.trust_level,
            engagement_score=profile.engagement_score,
            communication_style=profile.communication_style.value,
            response_length=profile.response_length.value,
            topics_of_interest=profile.topics_of_interest,
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Error retrieving profile: {e}")
        
        # Return structured error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve profile: {str(e)}",
        )


@router.put("/profile/{user_id}", response_model=ProfileResponse)
async def update_user_profile(
    user_id: str = Path(..., description="User identifier"),
    preferences: PreferenceUpdate = ...
) -> ProfileResponse:
    """
    Update user preferences.
    
    Updates the user's profile preferences including communication style,
    response length, topics of interest, and preferred name.
    
    Args:
        user_id: User identifier
        preferences: Preference updates
        
    Returns:
        ProfileResponse with updated profile data
        
    Raises:
        HTTPException: If profile manager is not initialized, validation fails, or update fails
    """
    try:
        profile_manager = get_profile_manager()
        
        # Validate user_id
        if not user_id or not user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id cannot be empty",
            )
        
        # Get or create profile
        profile = profile_manager.get_or_create_profile(user_id)
        
        # Build update kwargs
        update_kwargs = {}
        
        if preferences.preferred_name is not None:
            update_kwargs["preferred_name"] = preferences.preferred_name
        
        if preferences.communication_style is not None:
            # Convert string to enum
            style_map = {
                "casual": CommunicationStyle.CASUAL,
                "professional": CommunicationStyle.PROFESSIONAL,
                "friendly": CommunicationStyle.FRIENDLY,
                "technical": CommunicationStyle.TECHNICAL,
                "playful": CommunicationStyle.PLAYFUL,
            }
            update_kwargs["communication_style"] = style_map[preferences.communication_style]
        
        if preferences.response_length is not None:
            # Convert string to enum
            length_map = {
                "brief": ResponseLength.BRIEF,
                "moderate": ResponseLength.MODERATE,
                "detailed": ResponseLength.DETAILED,
            }
            update_kwargs["response_length"] = length_map[preferences.response_length]
        
        if preferences.topics_of_interest is not None:
            update_kwargs["topics_of_interest"] = preferences.topics_of_interest
        
        # Update profile
        if update_kwargs:
            updated_profile = profile_manager.update_profile(user_id, **update_kwargs)
        else:
            updated_profile = profile
        
        # Calculate relationship age
        relationship_age = profile_manager.calculate_relationship_age(user_id)
        
        # Convert to API model
        return ProfileResponse(
            user_id=updated_profile.user_id,
            preferred_name=updated_profile.preferred_name,
            relationship_age_days=relationship_age,
            interaction_count=updated_profile.interaction_count,
            trust_level=updated_profile.trust_level,
            engagement_score=updated_profile.engagement_score,
            communication_style=updated_profile.communication_style.value,
            response_length=updated_profile.response_length.value,
            topics_of_interest=updated_profile.topics_of_interest,
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Handle validation errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Error updating profile: {e}")
        
        # Return structured error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update profile: {str(e)}",
        )


@router.get("/timeline/{user_id}", response_model=TimelineResponse)
async def get_user_timeline(
    user_id: str = Path(..., description="User identifier")
) -> TimelineResponse:
    """
    Get user relationship timeline.
    
    Retrieves significant events and milestones in the user's relationship
    with Morgan, including first interaction, trust milestones, etc.
    
    Args:
        user_id: User identifier
        
    Returns:
        TimelineResponse with timeline events
        
    Raises:
        HTTPException: If profile manager is not initialized or user not found
    """
    try:
        profile_manager = get_profile_manager()
        
        # Validate user_id
        if not user_id or not user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id cannot be empty",
            )
        
        # Get or create profile
        profile = profile_manager.get_or_create_profile(user_id)
        
        # Build timeline events
        events: List[TimelineEvent] = []
        
        # Add profile creation event
        events.append(
            TimelineEvent(
                event_type="profile_created",
                timestamp=profile.created_at,
                description="First interaction with Morgan",
                metadata={"user_id": user_id}
            )
        )
        
        # Add trust level milestones
        if profile.trust_level >= 0.25:
            events.append(
                TimelineEvent(
                    event_type="trust_milestone",
                    timestamp=profile.last_updated,
                    description="Reached 25% trust level",
                    metadata={"trust_level": 0.25}
                )
            )
        
        if profile.trust_level >= 0.5:
            events.append(
                TimelineEvent(
                    event_type="trust_milestone",
                    timestamp=profile.last_updated,
                    description="Reached 50% trust level",
                    metadata={"trust_level": 0.5}
                )
            )
        
        if profile.trust_level >= 0.75:
            events.append(
                TimelineEvent(
                    event_type="trust_milestone",
                    timestamp=profile.last_updated,
                    description="Reached 75% trust level",
                    metadata={"trust_level": 0.75}
                )
            )
        
        # Add interaction milestones
        if profile.interaction_count >= 10:
            events.append(
                TimelineEvent(
                    event_type="interaction_milestone",
                    timestamp=profile.last_updated,
                    description="Reached 10 interactions",
                    metadata={"interaction_count": 10}
                )
            )
        
        if profile.interaction_count >= 50:
            events.append(
                TimelineEvent(
                    event_type="interaction_milestone",
                    timestamp=profile.last_updated,
                    description="Reached 50 interactions",
                    metadata={"interaction_count": 50}
                )
            )
        
        if profile.interaction_count >= 100:
            events.append(
                TimelineEvent(
                    event_type="interaction_milestone",
                    timestamp=profile.last_updated,
                    description="Reached 100 interactions",
                    metadata={"interaction_count": 100}
                )
            )
        
        # Sort events by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        return TimelineResponse(
            user_id=user_id,
            events=events,
            total_events=len(events)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Error retrieving timeline: {e}")
        
        # Return structured error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve timeline: {str(e)}",
        )
