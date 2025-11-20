"""
Exa API Client for web search and content retrieval
"""
import logging
from typing import List, Dict, Optional
from exa_py import Exa
from config import EXA_API_KEY, SEARCH_CONFIG
from models import SearchResult

logger = logging.getLogger(__name__)


class ExaClient:
    """Wrapper for Exa API with error handling and caching"""
    
    def __init__(self, api_key: str = EXA_API_KEY):
        """Initialize Exa client"""
        self.client = Exa(api_key=api_key)
        self.cache = {}  # Simple in-memory cache
    
    def search(
        self,
        query: str,
        num_results: int = None,
        include_text: bool = True,
        include_highlights: bool = True,
    ) -> List[SearchResult]:
        """
        Search the web using Exa
        
        Args:
            query: Search query
            num_results: Number of results to return
            include_text: Include full text content
            include_highlights: Include highlighted passages
            
        Returns:
            List of SearchResult objects
        """
        if num_results is None:
            num_results = SEARCH_CONFIG["default_num_results"]
        
        # Check cache
        cache_key = f"{query}:{num_results}"
        if cache_key in self.cache:
            logger.info(f"Cache hit for query: {query}")
            return self.cache[cache_key]
        
        try:
            logger.info(f"Searching Exa for: {query}")
            
            # Configure highlights
            highlights_config = None
            if include_highlights and SEARCH_CONFIG["enable_highlights"]:
                highlights_config = {
                    "num_sentences": 3,
                    "highlights_per_url": SEARCH_CONFIG["highlights_per_url"],
                }
            
            # Perform search
            response = self.client.search_and_contents(
                query=query,
                num_results=num_results,
                text=include_text,
                highlights=highlights_config,
            )
            
            # Convert to SearchResult objects
            results = []
            for r in response.results:
                result = SearchResult(
                    url=r.url,
                    title=r.title,
                    text=r.text if include_text else "",
                    score=getattr(r, 'score', 1.0),
                    highlights=getattr(r, 'highlights', []),
                    author=getattr(r, 'author', None),
                    published_date=getattr(r, 'published_date', None),
                )
                results.append(result)
            
            # Cache results
            self.cache[cache_key] = results
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Exa: {str(e)}")
            return []
    
    def get_contents(
        self,
        urls: List[str],
        include_text: bool = True,
    ) -> List[SearchResult]:
        """
        Fetch content from specific URLs
        
        Args:
            urls: List of URLs to fetch
            include_text: Include full text content
            
        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"Fetching content from {len(urls)} URLs")
            
            response = self.client.get_contents(
                urls,
                text=include_text,
            )
            
            results = []
            for r in response.results:
                result = SearchResult(
                    url=r.url,
                    title=getattr(r, 'title', r.url),
                    text=r.text if include_text else "",
                    score=1.0,
                    highlights=[],
                )
                results.append(result)
            
            logger.info(f"Fetched {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching content: {str(e)}")
            return []
    
    def find_similar(
        self,
        url: str,
        num_results: int = 5,
    ) -> List[SearchResult]:
        """
        Find similar content to a given URL
        
        Args:
            url: Reference URL
            num_results: Number of similar results
            
        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"Finding similar content to: {url}")
            
            response = self.client.find_similar(
                url=url,
                num_results=num_results,
            )
            
            results = []
            for r in response.results:
                result = SearchResult(
                    url=r.url,
                    title=r.title,
                    text=getattr(r, 'text', ""),
                    score=getattr(r, 'score', 1.0),
                    highlights=[],
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} similar results")
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar content: {str(e)}")
            return []
    
    def clear_cache(self):
        """Clear the search cache"""
        self.cache.clear()
        logger.info("Search cache cleared")


# Singleton instance
_exa_client = None


def get_exa_client() -> ExaClient:
    """Get or create ExaClient instance"""
    global _exa_client
    if _exa_client is None:
        _exa_client = ExaClient()
    return _exa_client
