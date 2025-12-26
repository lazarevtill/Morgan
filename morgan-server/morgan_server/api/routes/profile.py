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
from morgan_server.api.routes.chat import get_assistant
# Import Enums from Core models for mapping
from morgan.intelligence.core.models import CommunicationStyle, ResponseLength


router = APIRouter(prefix="/api", tags=["profile"])


@router.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_user_profile(
    user_id: str = Path(..., description="User identifier")
) -> ProfileResponse:
    """
    Get user profile.
    """
    try:
        assistant = get_assistant()
        
        # Validate user_id
        if not user_id or not user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id cannot be empty",
            )
        
        # Get profile
        profile = assistant.get_user_profile(user_id)
        if not profile:
             # Should we create one? get_user_profile usually creates if not exists in Core.
             # But if something failed:
             raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found",
            )
        
        # Calculate relationship age
        # Core profile has get_relationship_age_days
        relationship_age = 0
        if hasattr(profile, 'get_relationship_age_days'):
            relationship_age = profile.get_relationship_age_days()
        elif hasattr(profile, 'profile_created'):
             delta = datetime.utcnow() - profile.profile_created
             relationship_age = delta.days
        
        # Get preferences safely
        prefs = profile.communication_preferences
        
        # Convert to API model
        return ProfileResponse(
            user_id=profile.user_id,
            preferred_name=profile.preferred_name,
            relationship_age_days=relationship_age,
            interaction_count=profile.interaction_count,
            trust_level=profile.trust_level,
            engagement_score=profile.engagement_score,
            communication_style=prefs.communication_style.value if prefs else "friendly",
            response_length=prefs.preferred_response_length.value if prefs else "moderate",
            topics_of_interest=prefs.topics_of_interest if prefs else [],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving profile: {e}")
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
    """
    try:
        assistant = get_assistant()
        
        if not user_id or not user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id cannot be empty",
            )
        
        # Build update kwargs with correct Enum types
        update_kwargs = {}
        
        if preferences.preferred_name is not None:
            update_kwargs["preferred_name"] = preferences.preferred_name
        
        if preferences.communication_style is not None:
            # Map string to Core Enum
            try:
                # Use value matching, assuming Core Enums use lowercase strings as values?
                # or verify map manually.
                # Assuming simple mapping:
                update_kwargs["communication_style"] = CommunicationStyle(preferences.communication_style.lower())
            except ValueError:
                # Fallback or error?
                pass
        
        if preferences.response_length is not None:
             try:
                update_kwargs["response_length"] = ResponseLength(preferences.response_length.lower())
             except ValueError:
                 pass
        
        if preferences.topics_of_interest is not None:
            update_kwargs["topics_of_interest"] = preferences.topics_of_interest
        
        # Update profile
        updated_profile = assistant.update_user_profile(user_id, **update_kwargs)
        
        if not updated_profile:
             raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Failed to update profile",
            )

        # Calculate relationship age
        relationship_age = 0
        if hasattr(updated_profile, 'get_relationship_age_days'):
            relationship_age = updated_profile.get_relationship_age_days()
        
        prefs = updated_profile.communication_preferences

        # Convert to API model
        return ProfileResponse(
            user_id=updated_profile.user_id,
            preferred_name=updated_profile.preferred_name,
            relationship_age_days=relationship_age,
            interaction_count=updated_profile.interaction_count,
            trust_level=updated_profile.trust_level,
            engagement_score=updated_profile.engagement_score,
            communication_style=prefs.communication_style.value if prefs else "friendly",
            response_length=prefs.preferred_response_length.value if prefs else "moderate",
            topics_of_interest=prefs.topics_of_interest if prefs else [],
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        print(f"Error updating profile: {e}")
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
    """
    try:
        assistant = get_assistant()
        
        if not user_id or not user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id cannot be empty",
            )
        
        profile = assistant.get_user_profile(user_id)
        if not profile:
             raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found",
            )
        
        events: List[TimelineEvent] = []
        
        # Add profile creation event
        events.append(
            TimelineEvent(
                event_type="profile_created",
                timestamp=profile.profile_created,
                description="First interaction with Morgan",
                metadata={"user_id": user_id}
            )
        )
        
        # Add actual milestones from profile
        if hasattr(profile, 'relationship_milestones'):
            for milestone in profile.relationship_milestones:
                events.append(
                    TimelineEvent(
                        event_type=milestone.milestone_type.value,
                        timestamp=milestone.timestamp,
                        description=milestone.description,
                        metadata={
                            "significance": milestone.emotional_significance,
                            "milestone_id": milestone.milestone_id
                        }
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
        raise
    except Exception as e:
        print(f"Error retrieving timeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve timeline: {str(e)}",
        )
