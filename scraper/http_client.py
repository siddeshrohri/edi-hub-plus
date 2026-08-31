"""
HTTP Client
Responsible for:
  - Creating and managing the requests session
  - Checking robots.txt before any fetch
  - Enforcing rate limiting between requests
  - Returning raw responses (no parsing here)
"""

import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import USER_AGENT, REQUEST_TIMEOUT_SEC, RATE_LIMIT_SEC
from utils.logger import get_logger

logger = get_logger(__name__)

# Cache robots.txt results so we don't re-fetch for every URL on the same domain
_robots_cache: dict[str, RobotFileParser] = {}


def _get_robots(domain: str) -> RobotFileParser:
    """Fetch and cache the robots.txt for a given domain."""
    if domain in _robots_cache:
        return _robots_cache[domain]

    rp = RobotFileParser()
    robots_url = f"{domain}/robots.txt"
    rp.set_url(robots_url)

    try:
        rp.read()
        logger.debug(f"Fetched robots.txt for {domain}")
    except Exception as e:
        logger.debug(f"Could not read robots.txt for {domain}: {e} — assuming allowed")

    _robots_cache[domain] = rp
    return rp


def is_allowed(url: str) -> bool:
    """Return True if our bot is allowed to fetch this URL per robots.txt."""
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    rp = _get_robots(domain)
    allowed = rp.can_fetch(USER_AGENT, url)
    if not allowed:
        logger.warning(f"robots.txt disallows: {url}")
    return allowed


def build_session() -> requests.Session:
    """Create a requests session with our bot headers pre-set."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.5",
    })
    return session


def fetch(url: str, session: requests.Session) -> tuple[requests.Response | None, str | None]:
    """
    Fetch a URL ethically:
      1. Check robots.txt
      2. Wait (rate limit)
      3. Make the request
      4. Return (response, None) on success, or (None, failure_reason) on failure.

    Failure reasons:
      robots-blocked   — robots.txt disallows this URL
      http-error       — server returned a 4xx/5xx response
      timeout          — request timed out
      connection-error — could not connect to the server
      unknown-error    — any other unexpected exception
    """
    if not is_allowed(url):
        logger.info(f"Skipping (robots.txt): {url}")
        return None, "robots-blocked"

    logger.debug(f"Rate limit pause ({RATE_LIMIT_SEC}s)")
    time.sleep(RATE_LIMIT_SEC)

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        logger.debug(f"Fetched {url} — {response.status_code} — {len(response.text)} chars")
        return response, None

    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP error fetching {url}: {e}")
        return None, f"http-error ({e.response.status_code if e.response is not None else 'unknown'})"
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching {url}")
        return None, "timeout"
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Connection error fetching {url}: {e}")
        return None, "connection-error"
    except Exception as e:
        logger.warning(f"Unexpected error fetching {url}: {e}")
        return None, f"unknown-error ({type(e).__name__})"