"""The UserModel — the stable, learned representation of *who the user is*.

Produced and maintained asynchronously by the Learning subsystem; read (never written) on the
request path by Personalization. This is the concrete answer to "knows me".
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from morgan_brain.models.base import UserScoped
from morgan_brain.models.emotion import SentimentScore


class RelationshipStage(str, Enum):
    """How well Morgan knows the user. Gates how proactive it is allowed to be."""

    NEW = "new"
    ACQUAINTED = "acquainted"
    FAMILIAR = "familiar"
    TRUSTED = "trusted"


class Trait(BaseModel):
    name: str
    value: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CommunicationPrefs(BaseModel):
    tone: str = "neutral"
    length: str = "balanced"  # terse | balanced | thorough
    formality: str = "neutral"
    code_vs_prose: str = "balanced"  # code_first | balanced | prose_first


class BehavioralPattern(BaseModel):
    description: str
    cue: str = ""  # e.g. "weekday 09:00"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class UserModel(UserScoped):
    traits: list[Trait] = Field(default_factory=list)
    comm_prefs: CommunicationPrefs = Field(default_factory=CommunicationPrefs)
    topics_of_interest: dict[str, float] = Field(default_factory=dict)
    behavioral_patterns: list[BehavioralPattern] = Field(default_factory=list)
    emotional_baseline: SentimentScore = Field(default_factory=SentimentScore)
    relationship_stage: RelationshipStage = RelationshipStage.NEW
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
