from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Resource:
    """
    Represents a single EDI resource.
    Populated progressively across pipeline stages:
      Stage 1 — title, url, author, resource_id
      Stage 2 — status, url_type, raw_html
      Stage 3 — cleaned_text
      Stage 4 — tags (OLMo output)
    """
    title:       str
    url:         str
    author:      str = "Unknown"
    resource_id: str = ""

    # Stage 2
    status:   str = "pending"   # pending | success | pdf-skip | js-skip | failed
    url_type: str = "unknown"   # webpage | pdf | unknown
    raw_html: Optional[str] = field(default=None, repr=False)

    # Stage 3
    cleaned_text: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_html", None)   # never serialise raw HTML — too large
        return d