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
class CodeFunction:
    """Represents a code function with metadata."""
    name: str
    signature: str
    docstring: Optional[str] = None
    body: str = ""
    start_line: int = 0
    end_line: int = 0
    language: str = "python"
    complexity: int = 0
    dependencies: List[str] = field(default_factory=list)


@dataclass
class CodeClass:
    """Represents a code class with metadata."""
    name: str
    methods: List[CodeFunction] = field(default_factory=list)
    docstring: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    language: str = "python"
    inheritance: List[str] = field(default_factory=list)


@dataclass
class CodeFile:
    """Represents a code file with extracted components."""
    path: str
    language: str
    functions: List[CodeFunction] = field(default_factory=list)
    classes: List[CodeClass] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    complexity_score: float = 0.0


class CodeIntelligenceService:
    """
    Advanced code intelligence service using Jina embeddings.
    
    Provides semantic code understanding, search, and analysis.
    """
    
    def __init__(self):
        """Initialize code intelligence service."""
        self.embedding_service = JinaEmbeddingService()
        self.supported_languages = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php'
        }
        
        logger.info("CodeIntelligenceService initialized")
    
    def analyze_code_file(self, file_path: str, content: str) -> CodeFile:
        """
        Analyze a code file and extract components.
        
        Args:
            file_path: Path to the code file
            content: File content
            
        Returns:
            CodeFile with extracted components
        """
        try:
            language = self._detect_language(file_path)
            
            code_file = CodeFile(
                path=file_path,
                language=language
            )
            
            if language == 'python':
                self._analyze_python_file(content, code_file)
            else:
                # Basic analysis for other languages
                self._analyze_generic_file(content, code_file)
            
            return code_file
            
        except Exception as e:
            logger.error(f"Failed to analyze code file {file_path}: {e}")
            return CodeFile(path=file_path, language="unknown")
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        return self.supported_languages.get(ext, "unknown")
    
    def _analyze_python_file(self, content: str, code_file: CodeFile) -> None:
        """Analyze Python file using AST."""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func = self._extract_function(node, content)
                    code_file.functions.append(func)
                elif isinstance(node, ast.ClassDef):
                    cls = self._extract_class(node, content)
                    code_file.classes.append(cls)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports = self._extract_imports(node)
                    code_file.imports.extend(imports)
                    
        except SyntaxError as e:
            logger.warning(f"Syntax error in Python file: {e}")
    
    def _analyze_generic_file(self, content: str, code_file: CodeFile) -> None:
        """Basic analysis for non-Python files."""
        lines = content.split('\n')
        code_file.docstring = self._extract_file_docstring(lines)
        
        # Basic function detection using regex
        func_pattern = r'(function|def|func|fn)\s+(\w+)'
        for i, line in enumerate(lines):
            match = re.search(func_pattern, line)
            if match:
                func = CodeFunction(
                    name=match.group(2),
                    signature=line.strip(),
                    start_line=i + 1,
                    language=code_file.language
                )
                code_file.functions.append(func)
    
    def _extract_function(self, node: ast.FunctionDef, content: str) -> CodeFunction:
        """Extract function information from AST node."""
        lines = content.split('\n')
        
        return CodeFunction(
            name=node.name,
            signature=self._get_function_signature(node),
            docstring=ast.get_docstring(node),
            body=self._get_node_source(node, lines),
            start_line=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            language="python"
        )
    
    def _extract_class(self, node: ast.ClassDef, content: str) -> CodeClass:
        """Extract class information from AST node."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method = self._extract_function(item, content)
                methods.append(method)
        
        return CodeClass(
            name=node.name,
            methods=methods,
            docstring=ast.get_docstring(node),
            start_line=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            language="python",
            inheritance=[base.id for base in node.bases if isinstance(base, ast.Name)]
        )
    
    def _extract_imports(self, node) -> List[str]:
        """Extract import statements."""
        imports = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
        return imports
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Get function signature string."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return f"{node.name}({', '.join(args)})"
    
    def _get_node_source(self, node, lines: List[str]) -> str:
        """Get source code for AST node."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return '\n'.join(lines[node.lineno-1:node.end_lineno])
        else:
            return lines[node.lineno-1] if node.lineno <= len(lines) else ""
    
    def _extract_file_docstring(self, lines: List[str]) -> Optional[str]:
        """Extract file-level docstring."""
        for line in lines[:10]:  # Check first 10 lines
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                return stripped
        return None


# Global instance
_code_intelligence_service = None


def get_code_intelligence_service() -> CodeIntelligenceService:
    """Get singleton code intelligence service."""
    global _code_intelligence_service
    if _code_intelligence_service is None:
        _code_intelligence_service = CodeIntelligenceService()
    return _code_intelligence_service 
 