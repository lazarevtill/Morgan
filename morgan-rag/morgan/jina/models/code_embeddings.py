"""
Jina Code Embeddings Model

Following official Jina AI example:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("jinaai/jina-code-embeddings-1.5b", trust_remote_code=True)
code_snippets = [
    "def hello_world(): print('Hello, World!')",
    "function greet(name) { return `Hello, ${name}!`; }",
    "class Calculator: def add(self, a, b): return a + b"
]
embeddings = model.encode(code_snippets)
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CodeEmbeddingResult:
    """Result from code embedding operation."""
    embeddings: np.ndarray
    code_snippets: List[str]
    languages: List[str]
    model_name: str
    processing_time: float


@dataclass
class CodeStructure:
    """Analyzed code structure and components."""
    language: str
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[str]
    docstrings: List[str]
    comments: List[str]
    complexity_metrics: Dict[str, float]


@dataclass
class CodeSearchResult:
    """Code search result with context."""
    code_snippet: str
    function_name: Optional[str]
    class_name: Optional[str]
    file_path: str
    line_numbers: Tuple[int, int]
    relevance_score: float
    context: str


class JinaCodeEmbeddings:
    """
    Jina Code Embeddings model implementation following official examples.
    
    This model is specialized for understanding source code across multiple
    programming languages with semantic code search capabilities.
    """
    
    MODEL_NAME = "jinaai/jina-code-embeddings-1.5b"
    
    def __init__(self, cache_dir: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize Jina Code Embeddings.
        
        Args:
            cache_dir: Directory to cache the model
            token: Hugging Face token for authentication
        """
        self.cache_dir = cache_dir
        self.token = token or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.model = None
        self._is_loaded = False
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def load_model(self) -> bool:
        """
        Load the Jina Code Embeddings model following official example.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._is_loaded and self.model is not None:
            return True
        
        try:
            # Following official Jina AI example:
            # from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer("jinaai/jina-code-embeddings-1.5b", trust_remote_code=True)
            from sentence_transformers import SentenceTransformer
            
            model_kwargs = {
                "trust_remote_code": True
            }
            
            if self.cache_dir:
                model_kwargs["cache_folder"] = self.cache_dir
            
            if self.token:
                model_kwargs["token"] = self.token
            
            logger.info(f"Loading {self.MODEL_NAME}...")
            self.model = SentenceTransformer(self.MODEL_NAME, **model_kwargs)
            self._is_loaded = True
            
            logger.info(f"Successfully loaded {self.MODEL_NAME}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {self.MODEL_NAME}: {e}")
            self._is_loaded = False
            return False
    
    def encode(self, code_snippets: List[str], languages: Optional[List[str]] = None, 
               batch_size: int = 32) -> CodeEmbeddingResult:
        """
        Encode code snippets into embeddings following official example.
        
        Args:
            code_snippets: List of code snippets to encode
            languages: Optional list of programming languages for each snippet
            batch_size: Batch size for processing
            
        Returns:
            CodeEmbeddingResult with embeddings and metadata
        """
        import time
        
        start_time = time.time()
        
        if not self._is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Failed to load {self.MODEL_NAME}")
        
        try:
            # Following official Jina AI example:
            # code_snippets = [
            #     "def hello_world(): print('Hello, World!')",
            #     "function greet(name) { return `Hello, ${name}!`; }",
            #     "class Calculator: def add(self, a, b): return a + b"
            # ]
            # embeddings = model.encode(code_snippets)
            
            embeddings = self.model.encode(code_snippets, batch_size=batch_size)
            
            # Detect languages if not provided
            if languages is None:
                languages = [self._detect_language(snippet) for snippet in code_snippets]
            
            processing_time = time.time() - start_time
            
            logger.debug(f"Encoded {len(code_snippets)} code snippets in {processing_time:.3f}s")
            
            return CodeEmbeddingResult(
                embeddings=embeddings,
                code_snippets=code_snippets,
                languages=languages,
                model_name=self.MODEL_NAME,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Code encoding failed: {e}")
            raise
    
    def analyze_code_structure(self, code: str, language: str) -> CodeStructure:
        """
        Analyze code structure and extract components.
        
        Args:
            code: Source code to analyze
            language: Programming language
            
        Returns:
            CodeStructure with analyzed components
        """
        try:
            # Simple code analysis (in practice, you'd use AST parsers)
            functions = self._extract_functions(code, language)
            classes = self._extract_classes(code, language)
            imports = self._extract_imports(code, language)
            docstrings = self._extract_docstrings(code, language)
            comments = self._extract_comments(code, language)
            complexity = self._calculate_complexity(code)
            
            return CodeStructure(
                language=language,
                functions=functions,
                classes=classes,
                imports=imports,
                docstrings=docstrings,
                comments=comments,
                complexity_metrics=complexity
            )
            
        except Exception as e:
            logger.error(f"Code structure analysis failed: {e}")
            return CodeStructure(
                language=language,
                functions=[],
                classes=[],
                imports=[],
                docstrings=[],
                comments=[],
                complexity_metrics={}
            )
    
    def search_code_semantically(self, query: str, code_collection: List[str], 
                                top_k: int = 10) -> List[CodeSearchResult]:
        """
        Perform semantic code search.
        
        Args:
            query: Search query (natural language or code)
            code_collection: Collection of code snippets to search
            top_k: Number of top results to return
            
        Returns:
            List of CodeSearchResult objects
        """
        # Encode query and code collection
        query_result = self.encode([query])
        code_result = self.encode(code_collection)
        
        # Compute similarities
        query_embedding = query_result.embeddings[0]
        code_embeddings = code_result.embeddings
        
        # Calculate cosine similarities
        similarities = np.dot(code_embeddings, query_embedding) / (
            np.linalg.norm(code_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            code_snippet = code_collection[idx]
            language = self._detect_language(code_snippet)
            structure = self.analyze_code_structure(code_snippet, language)
            
            # Extract function/class names
            function_name = structure.functions[0]["name"] if structure.functions else None
            class_name = structure.classes[0]["name"] if structure.classes else None
            
            result = CodeSearchResult(
                code_snippet=code_snippet,
                function_name=function_name,
                class_name=class_name,
                file_path=f"snippet_{idx}",  # Placeholder
                line_numbers=(1, len(code_snippet.split('\n'))),
                relevance_score=float(similarities[idx]),
                context=f"Language: {language}"
            )
            results.append(result)
        
        return results
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code snippet."""
        # Simple heuristic-based language detection
        code_lower = code.lower().strip()
        
        if 'def ' in code_lower and ':' in code_lower:
            return 'python'
        elif 'function ' in code_lower and '{' in code_lower:
            return 'javascript'
        elif 'class ' in code_lower and '{' in code_lower and ';' in code_lower:
            return 'java'
        elif '#include' in code_lower or 'int main(' in code_lower:
            return 'c++'
        elif 'func ' in code_lower and '{' in code_lower:
            return 'go'
        elif 'fn ' in code_lower and '{' in code_lower:
            return 'rust'
        else:
            return 'unknown'
    
    def _extract_functions(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract function definitions from code."""
        functions = []
        lines = code.split('\n')
        
        if language == 'python':
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    func_name = line.split('def ')[1].split('(')[0].strip()
                    functions.append({
                        "name": func_name,
                        "line": i + 1,
                        "signature": line.strip()
                    })
        elif language == 'javascript':
            for i, line in enumerate(lines):
                if 'function ' in line:
                    func_name = line.split('function ')[1].split('(')[0].strip()
                    functions.append({
                        "name": func_name,
                        "line": i + 1,
                        "signature": line.strip()
                    })
        
        return functions
    
    def _extract_classes(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract class definitions from code."""
        classes = []
        lines = code.split('\n')
        
        if language == 'python':
            for i, line in enumerate(lines):
                if line.strip().startswith('class '):
                    class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                    classes.append({
                        "name": class_name,
                        "line": i + 1,
                        "signature": line.strip()
                    })
        
        return classes
    
    def _extract_imports(self, code: str, language: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            if language == 'python' and (line.startswith('import ') or line.startswith('from ')):
                imports.append(line)
            elif language == 'javascript' and ('import ' in line or 'require(' in line):
                imports.append(line)
        
        return imports
    
    def _extract_docstrings(self, code: str, language: str) -> List[str]:
        """Extract docstrings from code."""
        docstrings = []
        
        if language == 'python':
            # Simple docstring extraction (triple quotes)
            import re
            pattern = r'"""(.*?)"""'
            matches = re.findall(pattern, code, re.DOTALL)
            docstrings.extend(matches)
        
        return docstrings
    
    def _extract_comments(self, code: str, language: str) -> List[str]:
        """Extract comments from code."""
        comments = []
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            if language == 'python' and line.startswith('#'):
                comments.append(line)
            elif language in ['javascript', 'java', 'c++'] and line.startswith('//'):
                comments.append(line)
        
        return comments
    
    def _calculate_complexity(self, code: str) -> Dict[str, float]:
        """Calculate basic complexity metrics."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        return {
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "complexity_score": len(non_empty_lines) / max(len(lines), 1)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.MODEL_NAME,
            "description": "Jina Code Embeddings 1.5B for semantic code understanding",
            "use_case": "Code search, similarity, and analysis across programming languages",
            "performance": "High quality code embeddings with semantic understanding",
            "supported_languages": ["python", "javascript", "java", "c++", "go", "rust"],
            "embedding_dimension": 1536,  # Typical dimension for this model size
            "is_loaded": self._is_loaded
        }
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._is_loaded = False
            logger.info(f"Unloaded {self.MODEL_NAME}")