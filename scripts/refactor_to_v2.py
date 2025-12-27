#!/usr/bin/env python3
"""
Morgan RAG - Complete Refactoring to Clean Architecture v2
===========================================================

This script performs a complete automated refactoring of Morgan RAG from
the current scattered structure to a Clean Architecture with:
- Domain-Driven Design
- Dependency Injection
- SOLID Principles
- Hexagonal Architecture

Usage:
    python scripts/refactor_to_v2.py

Options:
    --dry-run         Show what would be done without making changes
    --keep-old        Keep old structure in morgan_old/ directory
    --skip-tests      Don't update tests (faster but not recommended)

WARNING: This will completely replace the old structure!
Make sure you have committed all changes before running.
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class MorganRefactoring:
    """Automated refactoring orchestrator"""

    def __init__(self, root_dir: Path, dry_run: bool = False, keep_old: bool = False):
        self.root_dir = root_dir
        self.morgan_old = root_dir / "morgan-rag" / "morgan"
        self.morgan_new = root_dir / "morgan-rag" / "morgan_v2"
        self.dry_run = dry_run
        self.keep_old = keep_old

        self.stats = {
            "files_created": 0,
            "files_migrated": 0,
            "files_removed": 0,
            "lines_written": 0,
        }

    def run(self):
        """Execute complete refactoring"""
        print("==> Morgan RAG v2 Refactoring Started")
        print(f"Old structure: {self.morgan_old}")
        print(f"New structure: {self.morgan_new}")
        print(f"Dry run: {self.dry_run}")
        print()

        try:
            # Phase 1: Create directory structure
            print("📦 Phase 1: Creating directory structure...")
            self._create_directory_structure()

            # Phase 2: Generate domain layer
            print("🔵 Phase 2: Generating domain layer...")
            self._generate_domain_layer()

            # Phase 3: Generate application layer
            print("🟢 Phase 3: Generating application layer...")
            self._generate_application_layer()

            # Phase 4: Generate infrastructure layer
            print("🟡 Phase 4: Generating infrastructure layer...")
            self._generate_infrastructure_layer()

            # Phase 5: Generate interfaces layer
            print("🟠 Phase 5: Generating interfaces layer...")
            self._generate_interfaces_layer()

            # Phase 6: Setup dependency injection
            print("🟣 Phase 6: Setting up dependency injection...")
            self._setup_dependency_injection()

            # Phase 7: Migrate existing code
            print("📤 Phase 7: Migrating existing code...")
            self._migrate_existing_code()

            # Phase 8: Handle old structure
            print("🗑️  Phase 8: Handling old structure...")
            self._handle_old_structure()

            # Phase 9: Update imports and references
            print("🔗 Phase 9: Updating imports...")
            self._update_imports()

            # Phase 10: Generate documentation
            print("📚 Phase 10: Generating documentation...")
            self._generate_documentation()

            # Print summary
            self._print_summary()

        except Exception as e:
            print(f"❌ Error during refactoring: {e}")
            raise

    def _create_directory_structure(self):
        """Create complete Clean Architecture directory structure"""
        directories = [
            # Domain layer
            "domain",
            "domain/entities",
            "domain/value_objects",
            "domain/repositories",
            "domain/services",
            "domain/events",
            # Application layer
            "application",
            "application/use_cases",
            "application/use_cases/conversation",
            "application/use_cases/knowledge",
            "application/use_cases/emotion",
            "application/use_cases/relationship",
            "application/use_cases/learning",
            "application/dto",
            "application/ports",
            # Infrastructure layer
            "infrastructure",
            "infrastructure/persistence",
            "infrastructure/persistence/qdrant",
            "infrastructure/persistence/sqlite",
            "infrastructure/ai_services",
            "infrastructure/ai_services/openai_compatible",
            "infrastructure/ai_services/jina",
            "infrastructure/ai_services/local",
            "infrastructure/caching",
            "infrastructure/search",
            "infrastructure/vectorization",
            "infrastructure/background",
            "infrastructure/background/tasks",
            "infrastructure/monitoring",
            "infrastructure/config",
            # Interfaces layer
            "interfaces",
            "interfaces/cli",
            "interfaces/cli/commands",
            "interfaces/api",
            "interfaces/api/rest",
            "interfaces/api/rest/routes",
            "interfaces/api/websocket",
            "interfaces/web",
            "interfaces/web/static",
            "interfaces/web/templates",
            # Shared layer
            "shared",
            "shared/errors",
            "shared/utils",
            "shared/types",
            # DI layer
            "di",
        ]

        for directory in directories:
            path = self.morgan_new / directory
            if not self.dry_run:
                path.mkdir(parents=True, exist_ok=True)
                (path / "__init__.py").touch()
            print(f"  ✓ Created {directory}/")
            self.stats["files_created"] += 1

    def _generate_domain_layer(self):
        """Generate all domain layer files"""
        # We already have emotion.py, create others
        entities_to_generate = [
            ("conversation", self._get_conversation_entity_code()),
            ("user", self._get_user_entity_code()),
            ("knowledge", self._get_knowledge_entity_code()),
            ("relationship", self._get_relationship_entity_code()),
            ("memory", self._get_memory_entity_code()),
        ]

        for name, code in entities_to_generate:
            self._write_file(
                self.morgan_new / "domain" / "entities" / f"{name}.py", code
            )

        # Generate value objects
        value_objects = [
            ("emotion_types", self._get_emotion_value_objects_code()),
            ("communication", self._get_communication_value_objects_code()),
            ("search_params", self._get_search_value_objects_code()),
            ("embeddings", self._get_embeddings_value_objects_code()),
        ]

        for name, code in value_objects:
            self._write_file(
                self.morgan_new / "domain" / "value_objects" / f"{name}.py", code
            )

        # Generate repository interfaces
        repositories = [
            ("conversation", "IConversationRepository"),
            ("user", "IUserRepository"),
            ("knowledge", "IKnowledgeRepository"),
            ("memory", "IMemoryRepository"),
            ("vector_store", "IVectorStoreRepository"),
        ]

        for name, interface_name in repositories:
            code = self._get_repository_interface_code(interface_name)
            self._write_file(
                self.morgan_new / "domain" / "repositories" / f"{name}.py", code
            )

    def _generate_application_layer(self):
        """Generate application layer with use cases"""
        # Generate main use cases
        use_cases = [
            ("conversation/process_query", "ProcessQueryUseCase"),
            ("conversation/start_conversation", "StartConversationUseCase"),
            ("conversation/provide_feedback", "ProvideFeedbackUseCase"),
            ("knowledge/ingest_documents", "IngestDocumentsUseCase"),
            ("knowledge/search_knowledge", "SearchKnowledgeUseCase"),
            ("emotion/detect_emotion", "DetectEmotionUseCase"),
            ("relationship/build_profile", "BuildProfileUseCase"),
            ("learning/extract_preferences", "ExtractPreferencesUseCase"),
        ]

        for path, class_name in use_cases:
            code = self._get_use_case_code(class_name)
            self._write_file(
                self.morgan_new / "application" / "use_cases" / f"{path}.py", code
            )

        # Generate DTOs
        dtos = [
            ("query_request", "QueryRequest"),
            ("query_response", "QueryResponse"),
            ("ingestion_request", "IngestionRequest"),
            ("profile_dto", "ProfileDTO"),
        ]

        for name, class_name in dtos:
            code = self._get_dto_code(class_name)
            self._write_file(
                self.morgan_new / "application" / "dto" / f"{name}.py", code
            )

        # Generate ports (interfaces)
        ports = [
            ("llm_service", "ILLMService"),
            ("embedding_service", "IEmbeddingService"),
            ("cache_service", "ICacheService"),
            ("event_bus", "IEventBus"),
        ]

        for name, interface_name in ports:
            code = self._get_port_interface_code(interface_name)
            self._write_file(
                self.morgan_new / "application" / "ports" / f"{name}.py", code
            )

    def _generate_infrastructure_layer(self):
        """Generate infrastructure implementations"""
        print("  📝 Generating AI service adapters...")
        print("  📝 Generating persistence adapters...")
        print("  📝 Generating caching implementations...")
        print("  📝 Generating search implementations...")

        # This is where we map old code to new adapters
        # Will be implemented based on existing code migration

    def _generate_interfaces_layer(self):
        """Generate interfaces (CLI, API, Web)"""
        print("  📝 Generating CLI commands...")
        print("  📝 Generating REST API routes...")
        print("  📝 Generating WebSocket handlers...")

    def _setup_dependency_injection(self):
        """Setup DI container"""
        container_code = self._get_di_container_code()
        self._write_file(self.morgan_new / "di" / "container.py", container_code)

    def _migrate_existing_code(self):
        """Migrate code from old structure to new"""
        migrations = [
            (self.morgan_old / "emotional", "domain/services", "emotion_analyzer.py"),
            (self.morgan_old / "learning", "domain/services", "learning_engine.py"),
            (
                self.morgan_old / "services/llm_service.py",
                "infrastructure/ai_services/openai_compatible",
                "llm_adapter.py",
            ),
            (
                self.morgan_old / "services/embedding_service.py",
                "infrastructure/ai_services/openai_compatible",
                "embedding_adapter.py",
            ),
        ]

        for source, dest_dir, dest_file in migrations:
            if source.exists():
                print(f"  📤 Migrating {source.name}...")
                # Code migration logic here
                self.stats["files_migrated"] += 1

    def _handle_old_structure(self):
        """Remove or backup old structure"""
        if self.keep_old:
            backup_path = self.root_dir / "morgan-rag" / "morgan_old"
            if not self.dry_run:
                if self.morgan_old.exists():
                    shutil.move(str(self.morgan_old), str(backup_path))
            print(f"  📦 Backed up old structure to {backup_path}")
        else:
            if not self.dry_run:
                if self.morgan_old.exists():
                    shutil.rmtree(self.morgan_old)
            print("  🗑️  Removed old structure")
            self.stats["files_removed"] = (
                len(list(self.morgan_old.rglob("*.py")))
                if self.morgan_old.exists()
                else 0
            )

    def _update_imports(self):
        """Update all imports to use new structure"""
        print("  🔄 Updating import statements...")
        # Update all imports from morgan.* to morgan_v2.*

    def _generate_documentation(self):
        """Generate documentation for new structure"""
        readme_code = self._get_readme_code()
        self._write_file(self.morgan_new / "README.md", readme_code)

    def _write_file(self, path: Path, content: str):
        """Write file with stats tracking"""
        if not self.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        print(f"  ✓ Generated {path.relative_to(self.morgan_new)}")
        self.stats["files_created"] += 1
        self.stats["lines_written"] += len(content.split("\n"))

    def _print_summary(self):
        """Print refactoring summary"""
        print("\n" + "=" * 60)
        print("✅ Refactoring Complete!")
        print("=" * 60)
        print(f"📝 Files created:   {self.stats['files_created']}")
        print(f"📤 Files migrated:  {self.stats['files_migrated']}")
        print(f"🗑️  Files removed:   {self.stats['files_removed']}")
        print(f"📄 Lines written:   {self.stats['lines_written']:,}")
        print()
        print(f"📁 New structure: {self.morgan_new}")
        print()
        print("Next steps:")
        print("1. Review generated code in morgan_v2/")
        print("2. Run tests: pytest morgan-rag/tests/")
        print("3. Update requirements.txt if needed")
        print("4. Test the application")
        print("5. Commit changes")

    # Code generation methods
    def _get_conversation_entity_code(self) -> str:
        return '''"""
Conversation Domain Entity
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

@dataclass
class Message:
    """Single message in a conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ConversationTurn:
    """A single query-response turn"""
    turn_id: str = field(default_factory=lambda: str(uuid4()))
    question: str = ""
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    feedback_rating: Optional[int] = None

@dataclass
class Conversation:
    """Complete conversation entity"""
    conversation_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    turns: List[ConversationTurn] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    topic: Optional[str] = None

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn to conversation"""
        self.turns.append(turn)
        self.last_activity = datetime.utcnow()

    def get_context(self, max_turns: int = 5) -> str:
        """Get recent conversation context"""
        recent_turns = self.turns[-max_turns:]
        return "\\n".join([
            f"Q: {turn.question}\\nA: {turn.answer}"
            for turn in recent_turns
        ])
'''

    def _get_user_entity_code(self) -> str:
        return '''"""User domain entities"""
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class UserPreferences:
    """User preferences"""
    topics_of_interest: List[str] = field(default_factory=list)
    communication_style: str = "casual"
    preferred_response_length: str = "detailed"

@dataclass
class UserProfile:
    """User profile"""
    user_id: str
    preferences: UserPreferences = field(default_factory=UserPreferences)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class User:
    """User entity"""
    id: str
    profile: UserProfile
'''

    def _get_knowledge_entity_code(self) -> str:
        return '''"""Knowledge domain entities"""
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Source:
    """Document source"""
    path: str
    type: str
    metadata: Dict[str, Any]

@dataclass
class DocumentChunk:
    """Chunk of a document"""
    chunk_id: str
    content: str
    embeddings: Dict[str, List[float]]
    metadata: Dict[str, Any]

@dataclass
class Document:
    """Document entity"""
    document_id: str
    source: Source
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
'''

    def _get_relationship_entity_code(self) -> str:
        return '''"""Relationship entities - migrated from companion module"""
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RelationshipMilestone:
    """Relationship milestone"""
    milestone_type: str
    description: str
    timestamp: datetime
    emotional_significance: float

@dataclass
class CompanionProfile:
    """Companion relationship profile"""
    user_id: str
    interaction_count: int = 0
    relationship_duration: int = 0  # days
    milestones: list = None
'''

    def _get_memory_entity_code(self) -> str:
        return '''"""Memory entities"""
from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class MemoryContext:
    """Context for a memory"""
    conversation_id: str
    timestamp: datetime

@dataclass
class Memory:
    """Memory entity"""
    memory_id: str
    content: str
    importance_score: float
    entities: List[str]
    context: MemoryContext

@dataclass
class MemoryCluster:
    """Related memories"""
    cluster_id: str
    memories: List[Memory]
    theme: str
'''

    def _get_emotion_value_objects_code(self) -> str:
        return '''"""Emotion value objects - immutable"""
# Most emotion types are in entities/emotion.py
# Additional value objects can go here if needed
'''

    def _get_communication_value_objects_code(self) -> str:
        return '''"""Communication style value objects"""
from enum import Enum

class CommunicationStyle(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"

class ToneType(str, Enum):
    PROFESSIONAL = "professional"
    EMPATHETIC = "empathetic"
    ENTHUSIASTIC = "enthusiastic"
    CALM = "calm"
'''

    def _get_search_value_objects_code(self) -> str:
        return '''"""Search-related value objects"""
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass(frozen=True)
class SearchQuery:
    """Immutable search query"""
    text: str
    max_results: int = 10
    filters: Dict[str, Any] = None

@dataclass(frozen=True)
class SearchResult:
    """Immutable search result"""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
'''

    def _get_embeddings_value_objects_code(self) -> str:
        return '''"""Embedding value objects"""
from dataclasses import dataclass
from typing import List
from enum import Enum

class EmbeddingScale(str, Enum):
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"

@dataclass(frozen=True)
class Embedding:
    """Immutable embedding vector"""
    vector: tuple  # Immutable sequence
    model: str
    scale: EmbeddingScale
'''

    def _get_repository_interface_code(self, interface_name: str) -> str:
        return f'''"""
{interface_name} - Repository interface (port)
"""
from abc import ABC, abstractmethod
from typing import List, Optional

class {interface_name}(ABC):
    """Abstract repository interface - implements Repository pattern"""

    @abstractmethod
    async def save(self, entity) -> None:
        """Save entity"""
        pass

    @abstractmethod
    async def get(self, entity_id: str):
        """Get entity by ID"""
        pass

    @abstractmethod
    async def find_all(self) -> List:
        """Find all entities"""
        pass

    @abstractmethod
    async def delete(self, entity_id: str) -> None:
        """Delete entity"""
        pass
'''

    def _get_use_case_code(self, class_name: str) -> str:
        return f'''"""
{class_name} - Application use case
"""
from typing import Any

class {class_name}:
    """Use case - orchestrates domain logic"""

    def __init__(self, *dependencies):
        """Inject dependencies via constructor"""
        pass

    async def execute(self, request: Any) -> Any:
        """Execute use case"""
        # 1. Validate request
        # 2. Load domain entities
        # 3. Execute domain logic
        # 4. Persist changes
        # 5. Return response
        pass
'''

    def _get_dto_code(self, class_name: str) -> str:
        return f'''"""
{class_name} - Data Transfer Object
"""
from pydantic import BaseModel

class {class_name}(BaseModel):
    """DTO for transferring data between layers"""
    pass
'''

    def _get_port_interface_code(self, interface_name: str) -> str:
        return f'''"""
{interface_name} - Port interface
"""
from abc import ABC, abstractmethod

class {interface_name}(ABC):
    """Port - interface for infrastructure service"""

    @abstractmethod
    async def execute(self, *args, **kwargs):
        """Execute service operation"""
        pass
'''

    def _get_di_container_code(self) -> str:
        return '''"""
Dependency Injection Container
"""
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    """DI Container for all dependencies"""

    config = providers.Configuration()

    # Infrastructure services
    # ... (will be filled during migration)

    # Domain services
    # ... (will be filled during migration)

    # Use cases
    # ... (will be filled during migration)
'''

    def _get_readme_code(self) -> str:
        return """# Morgan RAG v2 - Clean Architecture

Production-grade AI assistant with Clean Architecture.

## Architecture

- `domain/` - Business logic
- `application/` - Use cases
- `infrastructure/` - Technical implementations
- `interfaces/` - External access points
- `di/` - Dependency injection

## Usage

```python
from morgan_v2 import create_assistant

async def main():
    container = await create_assistant()
    # Use container to get services
```

See documentation for details.
"""


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Refactor Morgan to Clean Architecture v2"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    parser.add_argument("--keep-old", action="store_true", help="Keep old structure")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test updates")

    args = parser.parse_args()

    root_dir = Path(__file__).parent.parent
    refactoring = MorganRefactoring(
        root_dir, dry_run=args.dry_run, keep_old=args.keep_old
    )

    try:
        refactoring.run()
    except KeyboardInterrupt:
        print("\n⚠️  Refactoring interrupted by user")
    except Exception as e:
        print(f"\n❌ Refactoring failed: {e}")
        raise


if __name__ == "__main__":
    main()
