"""
EDI Hub+ Pipeline — Stage 1 Entry Point

Usage:
    python main.py --excel "Data-used-in-the-prototype-12-June-2026.xlsm"
    python main.py --excel path/to/file.xlsm --output my_output.json
"""

import argparse
import json
import sys

from config import STAGE1_OUTPUT_FILE
from scraper.discovery import discover_resources
from utils.logger import get_logger

logger = get_logger(__name__)


def save_resources(resources, output_path: str) -> None:
    """Serialise discovered resources to JSON."""
    data = [r.to_dict() for r in resources]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(data)} resources → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="EDI Hub+ Pipeline — Stage 1: Resource Discovery"
    )
    parser.add_argument(
        "--excel",
        required=True,
        help="Path to the prototype Excel file (.xlsm)",
    )
    parser.add_argument(
        "--output",
        default=STAGE1_OUTPUT_FILE,
        help=f"Output JSON path (default: {STAGE1_OUTPUT_FILE})",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EDI Hub+ Pipeline — Stage 1: Resource Discovery")
    logger.info("=" * 60)

    resources = discover_resources(args.excel)

    if not resources:
        logger.error("No resources loaded — check the Excel file path and sheet name")
        sys.exit(1)

    save_resources(resources, args.output)

    logger.info("-" * 60)
    logger.info(f"Total resources loaded : {len(resources)}")
    logger.info(f"Output saved to        : {args.output}")
    logger.info("-" * 60)
    logger.info("Next step: run Stage 2 (fetch + classify each URL)")


if __name__ == "__main__":
    main()