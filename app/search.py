"""
app/search.py
-------------
External Search Integration Module
==================================
Provides integration with Tavily search API and Wikipedia API to extend
the knowledge base with real-time web search capabilities.

Features
--------
- Tavily API integration for web search
- Wikipedia API integration for encyclopedia content
- Fallback mechanisms and error handling
- Rate limiting and caching support
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import List, Optional, Dict, Any
import aiohttp
import wikipedia
from wikipedia.exceptions import WikipediaException

logger = logging.getLogger(__name__)


class ExternalSearcher:
    """
    External search integration for extending knowledge base with web sources.
    
    Provides unified interface for Tavily web search and Wikipedia search
    with proper error handling and fallback mechanisms.
    """
    
    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        max_web_results: int = 3,
        max_wiki_results: int = 2,
        timeout: float = 10.0,
    ) -> None:
        """
        Initialize external search capabilities.
        
        Parameters
        ----------
        tavily_api_key : Optional[str]
            API key for Tavily search. If None, Tavily search is disabled.
        max_web_results : int
            Maximum number of web search results to return.
        max_wiki_results : int
            Maximum number of Wikipedia results to return.
        timeout : float
            Request timeout in seconds.
        """
        self.tavily_api_key = tavily_api_key
        self.max_web_results = max_web_results
        self.max_wiki_results = max_wiki_results
        self.timeout = timeout
        
        # Configure Wikipedia
        wikipedia.set_lang("en")
        
        logger.info(
            "ExternalSearcher initialized - Tavily: %s, Max web: %d, Max wiki: %d",
            "enabled" if tavily_api_key else "disabled",
            max_web_results,
            max_wiki_results,
        )
    
    async def search_tavily(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using Tavily API for web sources.
        
        Parameters
        ----------
        query : str
            Search query string.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of search results with content and metadata.
        """
        if not self.tavily_api_key:
            logger.debug("Tavily search disabled (no API key)")
            return []
        
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.tavily_api_key}"
        }
        
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": self.max_web_results,
            "include_domains": None,
            "exclude_domains": None,
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        
                        for result in data.get("results", []):
                            # Clean and format the content
                            content = self._clean_text(result.get("content", ""))
                            if content and len(content) > 50:  # Filter very short results
                                results.append({
                                    "text": content,
                                    "source": f"Web: {result.get('title', 'Unknown')} - {result.get('url', '')}",
                                    "url": result.get("url", ""),
                                    "title": result.get("title", ""),
                                    "score": result.get("score", 0.0),
                                })
                        
                        logger.info("Tavily search returned %d results for query: %s", len(results), query[:50])
                        return results
                    else:
                        logger.error("Tavily API error: status %d", response.status)
                        return []
                        
        except asyncio.TimeoutError:
            logger.error("Tavily search timeout for query: %s", query[:50])
            return []
        except Exception as e:
            logger.error("Tavily search failed for query %s: %s", query[:50], e)
            return []
    
    async def search_wikipedia(self, query: str) -> List[Dict[str, Any]]:
        """
        Search Wikipedia for encyclopedia content.
        
        Parameters
        ----------
        query : str
            Search query string.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of Wikipedia search results with content and metadata.
        """
        try:
            # Search for relevant pages
            search_results = wikipedia.search(query, results=self.max_wiki_results)
            results = []
            
            for page_title in search_results:
                try:
                    # Get page summary
                    page = wikipedia.page(page_title, auto_suggest=False)
                    summary = self._clean_text(page.summary[:1000])  # Limit summary length
                    
                    if summary and len(summary) > 50:
                        results.append({
                            "text": summary,
                            "source": f"Wikipedia: {page.title}",
                            "url": page.url,
                            "title": page.title,
                            "score": 0.0,  # Wikipedia doesn't provide relevance scores
                        })
                        
                except (WikipediaException, KeyError) as e:
                    logger.debug("Wikipedia page error for %s: %s", page_title, e)
                    continue
            
            logger.info("Wikipedia search returned %d results for query: %s", len(results), query[:50])
            return results
            
        except Exception as e:
            logger.error("Wikipedia search failed for query %s: %s", query[:50], e)
            return []
    
    async def search_all(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform comprehensive search using all available sources.
        
        Parameters
        ----------
        query : str
            Search query string.
            
        Returns
        -------
        List[Dict[str, Any]]
            Combined results from all search sources.
        """
        # Run searches concurrently
        tasks = []
        if self.tavily_api_key:
            tasks.append(self.search_tavily(query))
        tasks.append(self.search_wikipedia(query))
        
        try:
            results_lists = await asyncio.gather(*tasks, return_exceptions=True)
            all_results = []
            
            for i, results in enumerate(results_lists):
                if isinstance(results, Exception):
                    logger.error("Search method %d failed: %s", i, results)
                    continue
                all_results.extend(results)
            
            # Sort by score if available, otherwise keep original order
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            logger.info("Combined external search returned %d total results", len(all_results))
            return all_results
            
        except Exception as e:
            logger.error("Combined search failed for query %s: %s", query[:50], e)
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Parameters
        ----------
        text : str
            Raw text content.
            
        Returns
        -------
        str
            Cleaned text.
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\[\]\{\}\:\;\"\'\/\\]', '', text)
        
        # Ensure reasonable length
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text


# Convenience function for backward compatibility
async def search_external(
    query: str,
    tavily_api_key: Optional[str] = None,
    max_results: int = 5,
) -> List[str]:
    """
    Simple interface for external search.
    
    Parameters
    ----------
    query : str
        Search query.
    tavily_api_key : Optional[str]
        Tavily API key.
    max_results : int
        Maximum total results.
        
    Returns
    -------
    List[str]
        List of text content from search results.
    """
    searcher = ExternalSearcher(
        tavily_api_key=tavily_api_key,
        max_web_results=max_results // 2,
        max_wiki_results=max_results // 2,
    )
    
    results = await searcher.search_all(query)
    return [result["text"] for result in results[:max_results]]
