"""Audit payloads in the configured Qdrant collection for sensitive values."""

import argparse
import os
import sys
from collections import defaultdict
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import models

from app.core.config import settings
from app.providers.factory import get_vector_db_provider
from app.services.secret_scanner import scan_text


def _scan_payload(payload: dict[str, Any]) -> dict[str, int]:
    findings: dict[str, int] = defaultdict(int)
    for field, value in payload.items():
        if isinstance(value, (dict, list)):
            values = [(str(index), item) for index, item in enumerate(value)] if isinstance(value, list) else list(value.items())
            for nested_field, nested_value in values:
                for finding in scan_text(str(nested_value)):
                    findings[f"{field}.{nested_field}:{finding['type']}"] += 1
        else:
            for finding in scan_text(str(value)):
                findings[f"{field}:{finding['type']}"] += 1
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete", action="store_true", help="confirm before deleting points with findings")
    args = parser.parse_args(argv)

    try:
        provider = get_vector_db_provider()
        client = provider.client
        flagged: dict[Any, dict[str, int]] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                findings = _scan_payload(point.payload or {})
                if findings:
                    flagged[point.id] = findings
                    for field_type, count in sorted(findings.items()):
                        field, secret_type = field_type.rsplit(":", 1)
                        print(f"point_id={point.id} field={field} type={secret_type} count={count}")
            if offset is None:
                break

        if not flagged:
            print(f"No secret findings in '{settings.QDRANT_COLLECTION}'.")
            return 0
        print(f"Flagged {len(flagged)} point(s) in '{settings.QDRANT_COLLECTION}'.")
        if not args.delete:
            return 0
        confirmation = input("Type DELETE to remove flagged points: ")
        if confirmation != "DELETE":
            print("Deletion cancelled.")
            return 0
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=models.PointIdsList(points=list(flagged)),
            wait=True,
        )
        print(f"Deleted {len(flagged)} flagged point(s).")
        return 0
    except Exception as exc:
        print(f"ERROR: collection audit failed ({type(exc).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
