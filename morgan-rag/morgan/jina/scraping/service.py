"""
Jina AI Web Scraping Service

Intelligent web scraping service using ReaderLM-v2 for clean content extraction.
Implements content quality assessment, metadata extraction, and robust error handling.
Single responsibility: web scraping only.
"""

from typing import List, Dict, Any, Optional
import logging
import time
import re
import os
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse, urljoin
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Web scraping dependencies
import requests
from bs4 import BeautifulSoup
import html2text

# Local ReaderLM model
try:
    from ..models.reader_lm import JinaReaderLM
    READERLM_AVAILABLE = True
except ImportError:
    READERLM_AVAILABLE = False
    logging.warning("ReaderLM model not available - will use HTML parsing only")

logger = logging.getLogger(__name__)


@dataclass
class WebContent:
    """Extracted web content structure."""
    url: str
    title: str
    content: str
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_quality: float = 0.0
    images: List[str] = field(default_factory=list)
    extraction_method: str = "unknown"
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "author": self.author,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "metadata": self.metadata,
            "extraction_quality": self.extraction_quality,
            "images": self.images,
            "extraction_method": self.extraction_method,
            "processing_time": self.processing_time
        }


@dataclass
class WebMetadata:
    """Metadata extracted from web content."""
    title: str
    description: str
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    language: str = "en"
    content_type: str = "article"
    word_count: int = 0
    readability_score: float = 0.0
    has_structured_data: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "author": self.author,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "language": self.language,
            "content_type": self.content_type,
            "word_count": self.word_count,
            "readability_score": self.readability_score,
            "has_structured_data": self.has_structured_data
        }


@dataclass
class ContentQuality:
    """Assessment of web content quality."""
    readability_score: float
    completeness_score: float
    structure_score: float
    noise_level: float
    overall_quality: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "readability_score": self.readability_score,
            "completeness_score": self.completeness_score,
            "structure_score": self.structure_score,
            "noise_level": self.noise_level,
            "overall_quality": self.overall_quality
        }


class JinaWebScrapingService:
    """Self-hosted web scraping service using only local HTML parsing."""
    
    def __init__(
        self, 
        max_workers: int = 5,
        timeout: int = 30,
        user_agent: str = None
    ):
        """
        Initialize the self-hosted web scraping service.
        
        Args:
            max_workers: Maximum number of concurrent scraping workers
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._session = requests.Session()
        
        # Configure session headers with realistic user agent
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        self._session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
        
        # HTML to text converter
        self._html_converter = html2text.HTML2Text()
        self._html_converter.ignore_links = False
        self._html_converter.ignore_images = False
        self._html_converter.body_width = 0  # No line wrapping
        self._html_converter.unicode_snob = True
        self._html_converter.decode_errors = 'ignore'
        
        # Initialize local ReaderLM if available
        self._readerlm = None
        if READERLM_AVAILABLE:
            try:
                self._readerlm = JinaReaderLM(cache_dir=os.path.expanduser("~/.cache/morgan/readerlm"))
                logger.info("Local ReaderLM initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ReaderLM: {str(e)}")
                self._readerlm = None
        
        logger.info(f"Initialized Self-Hosted Web Scraping Service - Workers: {max_workers}, ReaderLM: {self._readerlm is not None}")
    
    def scrape_url(
        self,
        url: str,
        extract_images: bool = True,
        preserve_structure: bool = True
    ) -> WebContent:
        """
        Scrape content from URL using local HTML parsing.
        
        Args:
            url: URL to scrape
            extract_images: Whether to extract image URLs
            preserve_structure: Whether to preserve document structure
            
        Returns:
            Extracted web content
        """
        if not self._is_valid_url(url):
            logger.error(f"Invalid URL provided: {url}")
            raise ValueError(f"Invalid URL: {url}")
        
        logger.info(f"Scraping URL: {url}")
        start_time = time.time()
        
        try:
            # Try local ReaderLM first, then fallback to HTML parsing
            content = None
            extraction_method = "unknown"
            
            try:
                content = self._extract_with_local_readerlm(url, preserve_structure)
                extraction_method = "readerlm-v2-local"
                logger.debug(f"Successfully extracted content using local ReaderLM for {url}")
            except Exception as e:
                logger.warning(f"Local ReaderLM extraction failed for {url}: {str(e)}")
                
                # Fallback to HTML parsing
                content = self._extract_with_html_parsing(url, preserve_structure)
                extraction_method = "html_parsing"
                logger.debug(f"Used HTML parsing fallback for {url}")
            
            if content is None:
                raise Exception("Both ReaderLM and HTML parsing failed")
            
            # Set extraction method
            content.extraction_method = extraction_method
            
            # Extract metadata
            metadata = self._extract_metadata(url, content.content, content.title)
            content.metadata.update(metadata.to_dict())
            
            # Extract images if requested
            if extract_images:
                content.images = self._extract_image_urls(content.content, url)
            
            # Assess content quality
            quality_assessment = self._assess_content_quality_detailed(content.content)
            content.extraction_quality = quality_assessment.overall_quality
            content.metadata["quality_assessment"] = quality_assessment.to_dict()
            
            # Set processing time
            content.processing_time = time.time() - start_time
            
            logger.info(f"Scraped URL in {content.processing_time:.2f}s, quality: {content.extraction_quality:.2f}, method: {extraction_method}")
            
            return content
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to scrape URL {url}: {str(e)}")
            
            # Return error content
            return WebContent(
                url=url,
                title="Scraping Failed",
                content=f"Failed to scrape content: {str(e)}",
                extraction_quality=0.0,
                extraction_method="error",
                processing_time=processing_time,
                metadata={
                    "error": str(e), 
                    "scraped_at": datetime.now().isoformat(),
                    "attempted_methods": ["readerlm-v2" if use_readerlm and self.api_key else None, "html_parsing" if self.enable_fallback else None]
                }
            )
    
    def batch_scrape(
        self,
        urls: List[str],
        max_concurrent: int = 5,
        extract_images: bool = True
    ) -> List[WebContent]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum concurrent scraping operations
            extract_images: Whether to extract image URLs
            
        Returns:
            List of extracted web content
        """
        if not urls:
            logger.warning("Empty URL list provided for batch scraping")
            return []
        
        logger.info(f"Batch scraping {len(urls)} URLs with {max_concurrent} concurrent workers")
        start_time = time.time()
        
        # Limit concurrent operations
        actual_concurrent = min(max_concurrent, self.max_workers, len(urls))
        
        results = []
        for i in range(0, len(urls), actual_concurrent):
            batch = urls[i:i + actual_concurrent]
            batch_results = self._scrape_batch_concurrent(batch, extract_images)
            results.extend(batch_results)
            
            logger.debug(f"Completed batch {i//actual_concurrent + 1}/{(len(urls) + actual_concurrent - 1)//actual_concurrent}")
        
        elapsed_time = time.time() - start_time
        successful = sum(1 for r in results if r.extraction_quality > 0.5)
        logger.info(f"Batch scraped {len(urls)} URLs in {elapsed_time:.2f}s, {successful} successful")
        
        return results
    
    async def scrape_url_async(
        self,
        url: str,
        extract_images: bool = True,
        preserve_structure: bool = True
    ) -> WebContent:
        """
        Asynchronous URL scraping.
        
        Args:
            url: URL to scrape
            extract_images: Whether to extract image URLs
            preserve_structure: Whether to preserve document structure
            
        Returns:
            Extracted web content
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.scrape_url,
            url,
            extract_images,
            preserve_structure
        )
    
    def _extract_with_local_readerlm(
        self,
        url: str,
        preserve_structure: bool = True
    ) -> WebContent:
        """
        Extract content using local ReaderLM-v2 model.
        
        Args:
            url: URL to extract content from
            preserve_structure: Whether to preserve document structure
            
        Returns:
            Extracted web content
            
        Raises:
            Exception: If local ReaderLM processing fails
        """
        if not self._readerlm:
            raise Exception("Local ReaderLM not available")
        
        logger.debug(f"Extracting content using local ReaderLM for: {url}")
        
        try:
            # Use local ReaderLM to extract content
            result = self._readerlm.extract_content(url)
            
            # Convert ReaderLM result to WebContent
            return WebContent(
                url=url,
                title=result.title or f"Content from {urlparse(url).netloc}",
                content=result.content,
                author=result.metadata.get('author'),
                publication_date=None,  # ReaderLM doesn't extract dates yet
                metadata={
                    "scraped_at": datetime.now().isoformat(),
                    "scraper": "readerlm-v2-local",
                    "domain": urlparse(url).netloc,
                    "readerlm_metadata": result.metadata,
                    "processing_time": result.processing_time,
                    "readerlm_quality": result.quality_score
                }
            )
            
        except Exception as e:
            logger.error(f"Local ReaderLM extraction failed for {url}: {str(e)}")
            raise Exception(f"Local ReaderLM failed: {str(e)}")
    
    def _extract_with_html_parsing(
        self,
        url: str,
        preserve_structure: bool = True
    ) -> WebContent:
        """
        Fallback content extraction using basic HTML parsing.
        
        Args:
            url: URL to extract content from
            preserve_structure: Whether to preserve document structure
            
        Returns:
            Extracted web content
            
        Raises:
            Exception: If HTML parsing fails
        """
        logger.debug(f"Extracting content using HTML parsing for: {url}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            self._remove_noise_elements(soup)
            
            # Extract title
            title = self._extract_title_from_soup(soup)
            
            # Extract main content
            content = self._extract_main_content(soup, preserve_structure)
            
            # Extract metadata
            author = self._extract_author_from_soup(soup)
            pub_date = self._extract_publication_date_from_soup(soup)
            
            return WebContent(
                url=url,
                title=title,
                content=content,
                author=author,
                publication_date=pub_date,
                metadata={
                    "scraped_at": datetime.now().isoformat(),
                    "scraper": "html_parser",
                    "domain": urlparse(url).netloc,
                    "content_length": len(content),
                    "response_status": response.status_code
                }
            )
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"HTML parsing request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"HTML parsing failed: {str(e)}")
    
    def _remove_noise_elements(self, soup: BeautifulSoup) -> None:
        """Remove noise elements from HTML soup."""
        # Remove common noise elements
        noise_selectors = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            '.advertisement', '.ads', '.sidebar', '.menu', '.navigation',
            '.social-share', '.comments', '.related-posts', '.popup',
            '[class*="ad-"]', '[id*="ad-"]', '[class*="advertisement"]'
        ]
        
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()
    
    def _extract_title_from_soup(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML soup."""
        # Try different title sources in order of preference
        title_selectors = [
            'h1',
            'title',
            '[property="og:title"]',
            '[name="twitter:title"]',
            '.article-title',
            '.post-title',
            '.entry-title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title and len(title) > 5:  # Reasonable title length
                    return title
        
        return "Untitled"
    
    def _extract_main_content(self, soup: BeautifulSoup, preserve_structure: bool) -> str:
        """Extract main content from HTML soup."""
        # Try to find main content area
        content_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main',
            '.main-content',
            '#content'
        ]
        
        content_element = None
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                content_element = element
                break
        
        # If no specific content area found, use body
        if not content_element:
            content_element = soup.find('body') or soup
        
        if preserve_structure:
            # Convert to markdown while preserving structure
            return self._html_converter.handle(str(content_element))
        else:
            # Extract plain text
            return content_element.get_text(separator=' ', strip=True)
    
    def _extract_author_from_soup(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author from HTML soup."""
        author_selectors = [
            '[rel="author"]',
            '[property="article:author"]',
            '[name="author"]',
            '.author',
            '.byline',
            '.post-author'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                author = element.get_text(strip=True) or element.get('content', '')
                if author and len(author) < 100:  # Reasonable author length
                    return author
        
        return None
    
    def _extract_publication_date_from_soup(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract publication date from HTML soup."""
        date_selectors = [
            '[property="article:published_time"]',
            '[property="article:modified_time"]',
            '[name="publish_date"]',
            'time[datetime]',
            '.publish-date',
            '.post-date'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get('datetime') or element.get('content') or element.get_text(strip=True)
                if date_str:
                    try:
                        # Try to parse various date formats
                        return self._parse_date_string(date_str)
                    except (ValueError, TypeError):
                        continue
        
        return None
    
    def _parse_date_string(self, date_str: str) -> datetime:
        """Parse date string into datetime object."""
        # Common date formats
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y"
        ]
        
        # Clean the date string
        date_str = date_str.strip()
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse date: {date_str}")
    
    def _extract_title_from_content(self, content: str) -> str:
        """Extract title from content text."""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line.replace('# ', '').strip()
            elif line and len(line) > 10 and len(line) < 200:
                # First substantial line might be title
                return line
        return "Untitled"
    
    def _extract_metadata(self, url: str, content: str, title: str = "") -> WebMetadata:
        """Extract comprehensive metadata from content."""
        lines = content.split('\n')
        
        # Use provided title or extract from content
        if not title:
            for line in lines:
                if line.startswith('# '):
                    title = line.replace('# ', '').strip()
                    break
            if not title:
                title = "Untitled"
        
        # Calculate word count
        words = content.split()
        word_count = len(words)
        
        # Extract description (first substantial paragraph)
        description = ""
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('!['):
                if len(line) > 50:  # Substantial content
                    description = line[:300]
                    break
        
        # Detect language (simple heuristic)
        language = self._detect_language(content)
        
        # Determine content type
        content_type = self._determine_content_type(content, url)
        
        # Calculate readability score
        readability_score = self._calculate_readability_score(content)
        
        # Check for structured data
        has_structured_data = self._has_structured_data(content)
        
        return WebMetadata(
            title=title,
            description=description,
            word_count=word_count,
            language=language,
            content_type=content_type,
            readability_score=readability_score,
            has_structured_data=has_structured_data
        )
    
    def _detect_language(self, content: str) -> str:
        """Simple language detection based on common words."""
        # Simple heuristic - in production, use proper language detection library
        english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        spanish_indicators = ['el', 'la', 'y', 'o', 'pero', 'en', 'de', 'con', 'por', 'para']
        french_indicators = ['le', 'la', 'et', 'ou', 'mais', 'dans', 'de', 'avec', 'par', 'pour']
        
        content_lower = content.lower()
        words = content_lower.split()
        
        if len(words) < 10:
            return "en"  # Default to English for short content
        
        english_count = sum(1 for word in english_indicators if word in content_lower)
        spanish_count = sum(1 for word in spanish_indicators if word in content_lower)
        french_count = sum(1 for word in french_indicators if word in content_lower)
        
        if spanish_count > english_count and spanish_count > french_count:
            return "es"
        elif french_count > english_count and french_count > spanish_count:
            return "fr"
        else:
            return "en"
    
    def _determine_content_type(self, content: str, url: str) -> str:
        """Determine the type of content."""
        url_lower = url.lower()
        parsed_host = urlparse(url).hostname
        parsed_host = parsed_host.lower() if parsed_host else ""
        
        # Check URL patterns
        if any(pattern in url_lower for pattern in ['/blog/', '/news/', '/article/', '/post/']):
            return "article"
        elif any(pattern in url_lower for pattern in ['/docs/', '/documentation/', '/guide/', '/tutorial/']):
            return "documentation"
        elif any(pattern in url_lower for pattern in ['/api/', '/reference/']):
            return "reference"
        elif (
            parsed_host == "github.com" or parsed_host.endswith(".github.com")
            or parsed_host == "gitlab.com" or parsed_host.endswith(".gitlab.com")
        ):
            return "code_repository"
        elif any(pattern in url_lower for pattern in ['/wiki/', 'wikipedia.org']):
            return "wiki"
        
        # Check content patterns
        if content.count('```') >= 2 or content.count('`') >= 10:
            return "technical_documentation"
        elif content.count('#') >= 3:  # Multiple headings
            return "structured_article"
        elif len(content.split()) < 100:
            return "short_content"
        else:
            return "article"
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate a simple readability score."""
        if not content.strip():
            return 0.0
        
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        words = content.split()
        
        # Simple metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Readability score (0-1, higher is more readable)
        # Penalize very long sentences and very long words
        sentence_penalty = max(0, (avg_sentence_length - 20) / 20)
        word_penalty = max(0, (avg_word_length - 6) / 6)
        
        score = 1.0 - (sentence_penalty * 0.3 + word_penalty * 0.2)
        return max(0.0, min(1.0, score))
    
    def _has_structured_data(self, content: str) -> bool:
        """Check if content has structured data indicators."""
        structure_indicators = [
            content.count('#') >= 2,  # Multiple headings
            content.count('- ') >= 3,  # Lists
            content.count('1. ') >= 2,  # Numbered lists
            '```' in content,  # Code blocks
            content.count('|') >= 6,  # Tables
            content.count('\n\n') >= 3  # Multiple paragraphs
        ]
        
        return sum(structure_indicators) >= 2
    
    def _extract_image_urls(self, content: str, base_url: str) -> List[str]:
        """Extract image URLs from content."""
        images = []
        
        # Extract markdown image references
        markdown_images = re.findall(r'!\[.*?\]\((.*?)\)', content)
        for img_url in markdown_images:
            if img_url.startswith('http'):
                images.append(img_url)
            else:
                # Convert relative URLs to absolute
                images.append(urljoin(base_url, img_url))
        
        # Extract HTML img tags if present
        html_images = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE)
        for img_url in html_images:
            if img_url.startswith('http'):
                images.append(img_url)
            else:
                images.append(urljoin(base_url, img_url))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for img in images:
            if img not in seen:
                seen.add(img)
                unique_images.append(img)
        
        return unique_images
    
    def _assess_content_quality(self, content: str) -> float:
        """Simple content quality assessment (backward compatibility)."""
        quality_assessment = self._assess_content_quality_detailed(content)
        return quality_assessment.overall_quality
    
    def _assess_content_quality_detailed(self, content: str) -> ContentQuality:
        """Comprehensive content quality assessment."""
        if not content.strip():
            return ContentQuality(0.0, 0.0, 0.0, 1.0, 0.0)
        
        # Basic metrics
        words = content.split()
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        word_count = len(words)
        line_count = len(non_empty_lines)
        
        # 1. Readability Score
        readability_score = self._calculate_readability_score(content)
        
        # 2. Completeness Score
        completeness_factors = [
            min(1.0, word_count / 200),  # Substantial content
            min(1.0, line_count / 15),   # Multiple lines
            1.0 if word_count >= 50 else word_count / 50,  # Minimum viable content
        ]
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        # 3. Structure Score
        structure_indicators = {
            'headings': len([line for line in lines if line.strip().startswith('#')]),
            'paragraphs': content.count('\n\n'),
            'lists': content.count('- ') + content.count('* ') + len(re.findall(r'\d+\. ', content)),
            'code_blocks': content.count('```'),
            'links': content.count('[') + content.count('http'),
            'emphasis': content.count('**') + content.count('*') + content.count('_')
        }
        
        structure_score = 0.0
        structure_score += min(0.3, structure_indicators['headings'] * 0.1)
        structure_score += min(0.2, structure_indicators['paragraphs'] * 0.05)
        structure_score += min(0.2, structure_indicators['lists'] * 0.02)
        structure_score += min(0.1, structure_indicators['code_blocks'] * 0.05)
        structure_score += min(0.1, structure_indicators['links'] * 0.01)
        structure_score += min(0.1, structure_indicators['emphasis'] * 0.01)
        
        # 4. Noise Level (lower is better)
        noise_indicators = [
            content.lower().count('advertisement'),
            content.lower().count('subscribe'),
            content.lower().count('cookie'),
            content.lower().count('privacy policy'),
            len(re.findall(r'[A-Z]{3,}', content)),  # Excessive caps
            content.count('!!!'),  # Excessive punctuation
        ]
        
        noise_level = min(1.0, sum(noise_indicators) / 20)
        
        # 5. Overall Quality
        # Weight the factors
        overall_quality = (
            readability_score * 0.25 +
            completeness_score * 0.35 +
            structure_score * 0.25 +
            (1.0 - noise_level) * 0.15
        )
        
        return ContentQuality(
            readability_score=readability_score,
            completeness_score=completeness_score,
            structure_score=structure_score,
            noise_level=noise_level,
            overall_quality=min(1.0, max(0.0, overall_quality))
        )
    
    def _scrape_batch_concurrent(
        self,
        urls: List[str],
        extract_images: bool
    ) -> List[WebContent]:
        """Scrape a batch of URLs concurrently."""
        import concurrent.futures
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as executor:
            future_to_url = {
                executor.submit(self.scrape_url, url, extract_images): url
                for url in urls
            }
            
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    result = future.result(timeout=30)  # 30 second timeout per URL
                    results.append(result)
                except Exception as e:
                    url = future_to_url[future]
                    logger.error(f"Failed to scrape {url}: {str(e)}")
                    results.append(WebContent(
                        url=url,
                        title="Scraping Failed",
                        content=f"Timeout or error: {str(e)}",
                        extraction_quality=0.0
                    ))
        
        return results
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    def get_supported_domains(self) -> List[str]:
        """
        Get list of domains that work well with the scraper.
        
        Returns:
            List of recommended domains
        """
        return [
            "wikipedia.org",
            "github.com",
            "stackoverflow.com",
            "medium.com",
            "arxiv.org",
            "news.ycombinator.com",
            "reddit.com"
        ]
    
    def validate_url(self, url: str) -> Dict[str, Any]:
        """
        Validate and analyze a URL before scraping.
        
        Args:
            url: URL to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "is_valid": False,
            "domain": "",
            "scheme": "",
            "is_supported": False,
            "estimated_quality": 0.0,
            "warnings": []
        }
        
        try:
            parsed = urlparse(url)
            result["domain"] = parsed.netloc
            result["scheme"] = parsed.scheme
            result["is_valid"] = bool(parsed.scheme and parsed.netloc)
            
            if result["is_valid"]:
                # Check if domain is in supported list
                supported_domains = self.get_supported_domains()
                result["is_supported"] = any(domain in parsed.netloc for domain in supported_domains)
                
                # Estimate quality based on domain
                if result["is_supported"]:
                    result["estimated_quality"] = 0.8
                else:
                    result["estimated_quality"] = 0.6
                
                # Add warnings for potential issues
                if parsed.scheme != "https":
                    result["warnings"].append("Non-HTTPS URL may have security issues")
                
                if not result["is_supported"]:
                    result["warnings"].append("Domain not in tested list, quality may vary")
            
        except Exception as e:
            result["warnings"].append(f"URL parsing error: {str(e)}")
        
        return result
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about extraction performance."""
        # This would be implemented with actual tracking in production
        return {
            "total_extractions": 0,
            "successful_extractions": 0,
            "readerlm_extractions": 0,
            "fallback_extractions": 0,
            "average_quality": 0.0,
            "average_processing_time": 0.0
        }
    
    def close(self):
        """Clean up resources."""
        if self._readerlm:
            try:
                self._readerlm.unload_model()
                logger.debug("Unloaded local ReaderLM model")
            except Exception as e:
                logger.warning(f"Error unloading ReaderLM: {str(e)}")
        
        if self._executor:
            self._executor.shutdown(wait=True)
        if self._session:
            self._session.close()
        logger.info("Closed Self-Hosted Web Scraping Service")