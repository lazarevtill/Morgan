"""
Code Intelligence Engine for Repository Processing

Advanced code understanding and search using specialized code embeddings.
Integrates jina-code-embeddings-1.5b for semantic code analysis.
"""

import ast
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import time

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("tree-sitter not available, falling back to AST parsing")

try:
    from pygments.lexers import get_lexer_for_filename, guess_lexer
    from pygments.util import ClassNotFound
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    logging.warning("pygments not available, using basic language detection")

from .service import JinaEmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class CodeFun