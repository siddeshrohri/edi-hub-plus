"""
Classifier
Responsible for:
  - Determining the type of a URL before we fetch it
  - Detecting JS-heavy pages after fetching
  - Nothing else

URL types:
  pdf     — direct link to a PDF file
  webpage — standard HTML page (static or JS-rendered)
"""

import re
from urllib.parse import urlparse

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.logger import get_logger

logger = get_logger(__name__)

# Extensions that indicate a direct file download
PDF_EXTENSIONS = {".pdf"}

# If extracted plain text word count is below this, assume JS-rendered
JS_WORD_COUNT_THRESHOLD = 150


def classify_url(url: str) -> str:
    """
    Classify a URL by its extension before fetching.
    Returns: 'pdf' | 'webpage'
    """
    path = urlparse(url).path.lower()
    ext  = os.path.splitext(path)[1]

    if ext in PDF_EXTENSIONS:
        logger.debug(f"Classified as PDF: {url}")
        return "pdf"

    logger.debug(f"Classified as webpage: {url}")
    return "webpage"


def is_js_heavy(text: str) -> bool:
    """
    Given the plain text extracted from a fetched page,
    return True if there's too little content to be useful
    (strongly suggests JS-rendered content).
    """
    word_count = len(text.split())
    if word_count < JS_WORD_COUNT_THRESHOLD:
        logger.debug(f"JS-heavy detected — only {word_count} words extracted")
        return True
    return False