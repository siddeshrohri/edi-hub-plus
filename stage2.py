"""
EDI Hub+ Pipeline — Stage 2: Fetch and Classify

Reads stage1_resources.json, fetches each URL, classifies it,
and saves updated results + a skipped log.

Usage:
    python stage2.py
    python stage2.py --input stage1_resources.json --output stage2_resources.json
"""

import argparse
import json
import sys

from models import Resource
from scraper.fetcher import fetch_resources
from utils.logger import get_logger

logger = get_logger(__name__)

STAGE2_INPUT_FILE   = "stage1_resources.json"
STAGE2_OUTPUT_FILE  = "stage2_resources.json"
STAGE2_SKIPPED_FILE = "stage2_skipped.json"


def load_resources(input_path: str) -> list[Resource]:
    """Load Resource objects from Stage 1 JSON output."""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        resources = [Resource(**d) for d in data]
        logger.info(f"Loaded {len(resources)} resources from {input_path}")
        return resources
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        logger.error("Run Stage 1 first: python main.py --excel <path_to_excel>")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Could not load input file: {e}")
        sys.exit(1)


def save_results(resources: list[Resource], skipped: list[dict],
                 output_path: str, skipped_path: str) -> None:
    """Save updated resources and skipped log to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in resources], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(resources)} resources → {output_path}")

    with open(skipped_path, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(skipped)} skipped entries → {skipped_path}")


def main():
    parser = argparse.ArgumentParser(
        description="EDI Hub+ Pipeline — Stage 2: Fetch and Classify"
    )
    parser.add_argument(
        "--input",
        default=STAGE2_INPUT_FILE,
        help=f"Stage 1 output JSON (default: {STAGE2_INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        default=STAGE2_OUTPUT_FILE,
        help=f"Stage 2 output JSON (default: {STAGE2_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--skipped",
        default=STAGE2_SKIPPED_FILE,
        help=f"Skipped resources log (default: {STAGE2_SKIPPED_FILE})",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EDI Hub+ Pipeline — Stage 2: Fetch and Classify")
    logger.info("=" * 60)

    resources = load_resources(args.input)
    resources, skipped = fetch_resources(resources)
    save_results(resources, skipped, args.output, args.skipped)

    logger.info(f"Next step: run Stage 3 (text cleaning)")
    logger.info(f"  python stage3.py --input {args.output}")


if __name__ == "__main__":
    main()