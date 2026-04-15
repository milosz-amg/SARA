"""
ArXiv API Client

Wrapper for fetching papers from ArXiv API with rate limiting,
retry logic, and XML parsing.

Based on patterns from semantic_scholar/semanticscholar_api_client.py
"""

from typing import Dict, List, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArXivAPIError(RuntimeError):
    """Custom exception for ArXiv API errors."""
    pass


class RateLimitError(ArXivAPIError):
    """Exception raised when hitting rate limits."""
    pass


class ArXivAPIClient:
    """
    Client for interacting with ArXiv API.

    The ArXiv API returns results in Atom XML format.
    API documentation: https://arxiv.org/help/api/index
    """

    # XML namespaces used by ArXiv
    NAMESPACES = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom',
        'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
    }

    def __init__(
        self,
        base_url: str = "http://export.arxiv.org/api/query",
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        rate_limit_delay: float = 3.0,
    ):
        """
        Initialize ArXiv API client.

        Args:
            base_url: Base API URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for exponential retry strategy
            rate_limit_delay: Delay between requests (ArXiv requires 3 seconds minimum)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0

        self.logger = logger

        # Setup session with retry strategy
        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"])
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _rate_limit(self):
        """Implement client-side rate limiting (ArXiv requires 3 seconds minimum)."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _build_query_params(
        self,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
        start: int = 0,
        max_results: int = 100,
        sort_by: str = "submittedDate",
        sort_order: str = "descending"
    ) -> Dict[str, Any]:
        """
        Build query parameters for ArXiv API.

        Args:
            category: ArXiv category (e.g., 'cs.AI', 'physics', 'math')
            search_query: Custom search query (overrides category if provided)
            start: Starting index for pagination
            max_results: Number of results to return (max 2000 per request)
            sort_by: Sort field ('relevance', 'lastUpdatedDate', 'submittedDate')
            sort_order: Sort order ('ascending' or 'descending')

        Returns:
            Dictionary of query parameters
        """
        params = {
            'start': start,
            'max_results': min(max_results, 2000),  # ArXiv max is 2000
            'sortBy': sort_by,
            'sortOrder': sort_order,
        }

        # Build search query
        if search_query:
            params['search_query'] = search_query
        elif category:
            params['search_query'] = f'cat:{category}'
        else:
            raise ValueError("Either category or search_query must be provided")

        return params

    def _parse_entry(self, entry: ET.Element) -> Dict[str, Any]:
        """
        Parse a single paper entry from XML.

        Args:
            entry: XML Element representing a paper

        Returns:
            Dictionary with paper metadata
        """
        ns = self.NAMESPACES

        # Extract ID (format: http://arxiv.org/abs/2301.12345v1)
        arxiv_id_url = entry.find('atom:id', ns).text
        arxiv_id = arxiv_id_url.split('/abs/')[-1]
        # Remove version number (e.g., v1, v2)
        arxiv_id_clean = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id

        # Extract basic fields
        paper = {
            'id': arxiv_id_clean,
            'arxiv_id': arxiv_id,
            'title': entry.find('atom:title', ns).text.strip().replace('\n', ' '),
            'abstract': entry.find('atom:summary', ns).text.strip().replace('\n', ' '),
            'published': entry.find('atom:published', ns).text,
            'updated': entry.find('atom:updated', ns).text,
        }

        # Extract authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text)
        paper['authors'] = authors

        # Extract categories
        categories = []
        primary_category = None

        # Primary category
        primary_cat_elem = entry.find('arxiv:primary_category', ns)
        if primary_cat_elem is not None:
            primary_category = primary_cat_elem.get('term')
            categories.append(primary_category)

        # All categories
        for category in entry.findall('atom:category', ns):
            cat_term = category.get('term')
            if cat_term and cat_term not in categories:
                categories.append(cat_term)

        paper['categories'] = categories
        paper['primary_category'] = primary_category or (categories[0] if categories else None)

        # Extract optional fields
        doi_elem = entry.find('arxiv:doi', ns)
        paper['doi'] = doi_elem.text if doi_elem is not None else None

        comment_elem = entry.find('arxiv:comment', ns)
        paper['comment'] = comment_elem.text if comment_elem is not None else None

        journal_ref_elem = entry.find('arxiv:journal_ref', ns)
        paper['journal_ref'] = journal_ref_elem.text if journal_ref_elem is not None else None

        # Extract links
        paper['arxiv_url'] = arxiv_id_url.replace('/abs/', '/abs/').split('v')[0]
        paper['pdf_url'] = arxiv_id_url.replace('/abs/', '/pdf/').split('v')[0] + '.pdf'

        return paper

    def _parse_response(self, xml_content: str) -> Dict[str, Any]:
        """
        Parse ArXiv API XML response.

        Args:
            xml_content: Raw XML response string

        Returns:
            Dictionary with:
                - papers: List of paper dictionaries
                - total_results: Total number of results available
                - start_index: Starting index of this batch
                - items_per_page: Number of items in this batch
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise ArXivAPIError(f"Failed to parse XML response: {str(e)}")

        ns = self.NAMESPACES

        # Extract metadata
        total_results = int(root.find('opensearch:totalResults', ns).text)
        start_index = int(root.find('opensearch:startIndex', ns).text)
        items_per_page = int(root.find('opensearch:itemsPerPage', ns).text)

        # Parse entries
        papers = []
        for entry in root.findall('atom:entry', ns):
            try:
                paper = self._parse_entry(entry)
                papers.append(paper)
            except Exception as e:
                self.logger.warning(f"Failed to parse entry: {str(e)}")
                continue

        return {
            'papers': papers,
            'total_results': total_results,
            'start_index': start_index,
            'items_per_page': items_per_page,
        }

    def search_papers(
        self,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
        start: int = 0,
        max_results: int = 100,
        sort_by: str = "submittedDate",
        sort_order: str = "descending"
    ) -> Dict[str, Any]:
        """
        Search for papers on ArXiv.

        Args:
            category: ArXiv category (e.g., 'cs.AI', 'physics.gen-ph')
            search_query: Custom search query (overrides category)
            start: Starting index for pagination
            max_results: Number of results to return
            sort_by: Sort field ('relevance', 'lastUpdatedDate', 'submittedDate')
            sort_order: Sort order ('ascending' or 'descending')

        Returns:
            Dictionary with papers and metadata
        """
        self._rate_limit()

        params = self._build_query_params(
            category=category,
            search_query=search_query,
            start=start,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order
        )

        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Parse XML response
            result = self._parse_response(response.text)

            self.logger.debug(
                f"Fetched {len(result['papers'])} papers "
                f"(start={start}, total={result['total_results']})"
            )

            return result

        except requests.exceptions.Timeout:
            raise ArXivAPIError("Request timed out")
        except requests.exceptions.RequestException as e:
            raise ArXivAPIError(f"Request failed: {str(e)}")

    def fetch_all_papers(
        self,
        category: str,
        max_papers: int = 10000,
        batch_size: int = 1000,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple papers with pagination.

        Args:
            category: ArXiv category to fetch
            max_papers: Maximum number of papers to fetch
            batch_size: Papers per request (max 2000)
            progress_callback: Optional function called after each batch
                               with (current_count, total_expected)

        Returns:
            List of paper dictionaries
        """
        all_papers = []
        start = 0
        batch_size = min(batch_size, 1000)  # Conservative limit

        self.logger.info(f"Fetching up to {max_papers} papers from category '{category}'")

        while len(all_papers) < max_papers:
            # Fetch batch
            result = self.search_papers(
                category=category,
                start=start,
                max_results=batch_size
            )

            papers = result['papers']
            if not papers:
                self.logger.info("No more papers available")
                break

            all_papers.extend(papers)

            # Progress callback
            if progress_callback:
                progress_callback(len(all_papers), max_papers)

            # Log progress
            self.logger.info(
                f"Fetched {len(all_papers)}/{max_papers} papers "
                f"(total available: {result['total_results']})"
            )

            # Check if we've reached the limit
            if len(all_papers) >= max_papers:
                all_papers = all_papers[:max_papers]
                break

            # Check if there are more results
            if len(papers) < batch_size:
                self.logger.info("Reached end of available papers")
                break

            start += batch_size

        self.logger.info(f"Successfully fetched {len(all_papers)} papers")
        return all_papers


if __name__ == '__main__':
    # Test the client
    print("Testing ArXiv API Client")
    print("=" * 60)

    client = ArXivAPIClient()

    # Test with a small query
    print("Fetching 5 papers from cs.AI category...")
    result = client.search_papers(category='cs.AI', max_results=5)

    print(f"\nTotal results available: {result['total_results']}")
    print(f"Fetched: {len(result['papers'])} papers\n")

    # Print first paper
    if result['papers']:
        paper = result['papers'][0]
        print("Sample paper:")
        print(f"  ID: {paper['id']}")
        print(f"  Title: {paper['title'][:80]}...")
        print(f"  Authors: {', '.join(paper['authors'][:3])}...")
        print(f"  Categories: {', '.join(paper['categories'])}")
        print(f"  Published: {paper['published']}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
