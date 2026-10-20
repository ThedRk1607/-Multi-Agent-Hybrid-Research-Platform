#!/usr/bin/env python3
"""
OmniRAG — CLI Ingestion Script
Usage:
  python scripts/ingest.py --file path/to/file.pdf
  python scripts/ingest.py --dir path/to/folder/
  python scripts/ingest.py --text "Some raw text" --label "my-doc"
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag.ingestion import ingest_file, ingest_directory, ingest_documents, extract_from_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="OmniRAG document ingestion")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single file")
    group.add_argument("--dir", help="Path to a directory of files")
    group.add_argument("--text", help="Raw text to ingest")

    parser.add_argument("--label", default="cli-input", help="Source label for raw text")
    parser.add_argument("--extensions", nargs="+", default=[".pdf", ".docx", ".txt", ".md"])

    args = parser.parse_args()

    if args.file:
        logger.info("Ingesting file: %s", args.file)
        count = ingest_file(args.file)
        print(f"✅ Ingested {count} chunks from {args.file}")

    elif args.dir:
        logger.info("Ingesting directory: %s", args.dir)
        count = ingest_directory(args.dir, extensions=args.extensions)
        print(f"✅ Ingested {count} total chunks from {args.dir}")

    elif args.text:
        docs = extract_from_text(args.text, source=args.label)
        count = ingest_documents(docs)
        print(f"✅ Ingested {count} chunks from raw text")


if __name__ == "__main__":
    main()
