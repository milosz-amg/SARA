from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import time

from semantic_scholar.settings import SEMANTIC_SCHOLAR_API_URL, DEFAULT_TIMEOUT, DEFAULT_LIMIT, DEFAULT_FIELDS_AUTHOR

class SemanticScholarAPIError(RuntimeError):
    """Custom exception for Semantic Scholar API errors."""
    pass

class RateLimitError(SemanticScholarAPIError):
    """Exception raised when hitting rate limits."""
    pass

class SemanticScholarAPIClient:
    def __init__(
        self,
        api_url: str = SEMANTIC_SCHOLAR_API_URL,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        rate_limit_delay: float = 1.0,
    ):
        """
        Initialize Semantic Scholar API client.
        
        Args:
            api_url: Base API URL
            api_key: API key for authenticated requests (recommended)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for exponential retry strategy
            rate_limit_delay: Delay between requests to avoid rate limiting
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0
        
        self.logger = logging.getLogger(__name__)
        
        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"])
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))

    def _headers(self) -> Dict[str, str]:
        """Generate headers for API requests."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _rate_limit(self):
        """Implement client-side rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make GET request with proper error handling and rate limiting."""
        self._rate_limit()
        
        url = f"{self.api_url}{path}"
        clean_params = {k: v for k, v in params.items() if v is not None}
        
        try:
            response = self.session.get(
                url, 
                headers=self._headers(), 
                params=clean_params, 
                timeout=self.timeout
            )
            self._handle_response(response, url)
            return response.json()
        except requests.exceptions.RequestException as e:
            raise SemanticScholarAPIError(f"Request failed: {str(e)}")

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request for batch endpoints."""
        self._rate_limit()
        
        url = f"{self.api_url}{path}"
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        
        try:
            response = self.session.post(
                url, 
                headers=headers, 
                json=data, 
                timeout=self.timeout
            )
            self._handle_response(response, url)
            return response.json()
        except requests.exceptions.RequestException as e:
            raise SemanticScholarAPIError(f"Request failed: {str(e)}")

    def _handle_response(self, response: requests.Response, url: str):
        """Handle HTTP response with proper error messages."""
        if response.status_code == 429:
            raise RateLimitError("Rate limited by Semantic Scholar (HTTP 429). Consider using an API key or reducing request frequency.")
        elif response.status_code == 400:
            raise SemanticScholarAPIError(f"Bad request (400): Check your parameters. URL: {url}")
        elif response.status_code == 401:
            raise SemanticScholarAPIError("Unauthorized (401): Invalid API key")
        elif response.status_code == 403:
            raise SemanticScholarAPIError("Forbidden (403): Access denied")
        elif response.status_code == 404:
            raise SemanticScholarAPIError(f"Not found (404): Resource doesn't exist. URL: {url}")
        elif not response.ok:
            raise SemanticScholarAPIError(f"API error {response.status_code}: {response.text[:300]}")

    # -------- Author Endpoints --------
    
    def search_authors(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search for authors by name.
        
        Args:
            query: Author name to search for
            limit: Maximum number of results (1-100)
            offset: Number of results to skip
            fields: List of fields to return
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        params = {
            "query": query.strip(),
            "limit": min(max(limit, 1), 100),
            "offset": max(offset, 0),
            "fields": ",".join(fields or DEFAULT_FIELDS_AUTHOR),
        }
        return self._get("/author/search", params)

    def get_author(
        self,
        author_id: str,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get detailed information about an author.
        
        Args:
            author_id: Semantic Scholar author ID
            fields: List of fields to return
        """
        if not author_id.strip():
            raise ValueError("Author ID cannot be empty")
            
        params = {"fields": ",".join(fields or DEFAULT_FIELDS_AUTHOR)}
        return self._get(f"/author/{author_id.strip()}", params)

    def get_author_papers(
        self,
        author_id: str,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get papers by an author.
        
        Args:
            author_id: Semantic Scholar author ID
            limit: Maximum number of results (1-100)
            offset: Number of results to skip
            fields: List of paper fields to return
        """
        if not author_id.strip():
            raise ValueError("Author ID cannot be empty")

        default_paper_fields = ["paperId", "title", "year", "authors", "citationCount"]
        
        params = {
            "limit": min(max(limit, 1), 100),
            "offset": max(offset, 0),
            "fields": ",".join(fields or default_paper_fields),
        }
        return self._get(f"/author/{author_id.strip()}/papers", params)

    # -------- Paper Endpoints --------
    
    def search_papers(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        fields: Optional[List[str]] = None,
        year: Optional[str] = None,
        venue: Optional[str] = None,
        min_citation_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Search for papers using relevance search.
        
        Args:
            query: Search query (supports advanced operators)
            limit: Maximum number of results (1-100)
            offset: Number of results to skip
            fields: List of fields to return
            year: Publication year or range (e.g., "2020", "2018-2020")
            venue: Publication venue filter
            min_citation_count: Minimum citation count filter
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        default_paper_fields = ["paperId", "title", "year", "authors", "citationCount", "abstract"]
        
        params = {
            "query": query.strip(),
            "limit": min(max(limit, 1), 100),
            "offset": max(offset, 0),
            "fields": ",".join(fields or default_paper_fields),
        }

        if year:
            params["year"] = year
        if venue:
            params["venue"] = venue
        if min_citation_count is not None:
            params["minCitationCount"] = max(min_citation_count, 0)
            
        return self._get("/paper/search", params)

    def get_paper(
        self,
        paper_id: str,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get detailed information about a paper.
        
        Args:
            paper_id: Semantic Scholar paper ID, DOI, arXiv ID, etc.
            fields: List of fields to return
        """
        if not paper_id.strip():
            raise ValueError("Paper ID cannot be empty")
            
        default_paper_fields = ["paperId", "title", "year", "authors", "citationCount", "abstract"]
        params = {"fields": ",".join(fields or default_paper_fields)}
        return self._get(f"/paper/{paper_id.strip()}", params)

    def get_paper_citations(
        self,
        paper_id: str,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get papers that cite the given paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of results (1-1000)
            offset: Number of results to skip
            fields: List of fields to return for citing papers
        """
        if not paper_id.strip():
            raise ValueError("Paper ID cannot be empty")
            
        default_citation_fields = ["paperId", "title", "year", "authors", "citationCount"]
        
        params = {
            "limit": min(max(limit, 1), 1000),  # Citations endpoint allows up to 1000
            "offset": max(offset, 0),
            "fields": ",".join(fields or default_citation_fields),
        }
        return self._get(f"/paper/{paper_id.strip()}/citations", params)

    def get_paper_references(
        self,
        paper_id: str,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get papers referenced by the given paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of results (1-1000)
            offset: Number of results to skip
            fields: List of fields to return for referenced papers
        """
        if not paper_id.strip():
            raise ValueError("Paper ID cannot be empty")
            
        default_reference_fields = ["paperId", "title", "year", "authors", "citationCount"]
        
        params = {
            "limit": min(max(limit, 1), 1000),
            "offset": max(offset, 0),
            "fields": ",".join(fields or default_reference_fields),
        }
        return self._get(f"/paper/{paper_id.strip()}/references", params)

    # -------- Batch Endpoints --------
    
    def get_papers_batch(
        self,
        paper_ids: List[str],
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get details for multiple papers in a single request.
        
        Args:
            paper_ids: List of paper IDs (max 500)
            fields: List of fields to return
        """
        if not paper_ids:
            raise ValueError("Paper IDs list cannot be empty")
        if len(paper_ids) > 500:
            raise ValueError("Maximum 500 paper IDs allowed per batch request")
            
        clean_ids = [pid.strip() for pid in paper_ids if pid.strip()]
        if not clean_ids:
            raise ValueError("No valid paper IDs provided")
            
        default_paper_fields = ["paperId", "title", "year", "authors", "citationCount", "abstract"]
        
        data = {
            "ids": clean_ids,
            "fields": fields or default_paper_fields,
        }
        return self._post("/paper/batch", data)

    # -------- Utility Methods --------
    
    def get_paginated_results(
        self,
        search_function,
        max_results: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Helper method to automatically handle pagination and collect all results.
        
        Args:
            search_function: The search method to use (e.g., self.search_papers)
            max_results: Maximum total results to collect (None for all)
            **kwargs: Arguments to pass to the search function
        """
        all_results = []
        offset = kwargs.get('offset', 0)
        limit = min(kwargs.get('limit', DEFAULT_LIMIT), 100)
        
        while True:
            kwargs['offset'] = offset
            kwargs['limit'] = limit
            
            response = search_function(**kwargs)
            
            if 'data' in response:
                batch = response['data']
            elif 'results' in response:
                batch = response['results']
            else:
                batch = response if isinstance(response, list) else []
            
            if not batch:
                break
                
            all_results.extend(batch)
            
            if max_results and len(all_results) >= max_results:
                all_results = all_results[:max_results]
                break
                
            total = response.get('total', len(all_results))
            if len(all_results) >= total:
                break
                
            offset += len(batch)
            
            if len(batch) < limit:
                break
        
        return all_results

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self.session.close()
