"""Value-free secret detection and redaction for ingestion boundaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
import re
from typing import Any, TypedDict

from app.core.config import settings


class Finding(TypedDict):
    type: str
    start: int
    end: int


@dataclass(frozen=True)
class FindingSummary:
    source_id: str
    field: str
    type: str
    count: int


class SecretScanError(ValueError):
    """Raised when block mode finds sensitive data before ingestion."""

    def __init__(self, summaries: list[FindingSummary]) -> None:
        self.summaries = summaries
        super().__init__(format_scan_summaries(summaries))


_PATTERNS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), 0),
    ("openai_api_key", re.compile(r"\bsk-(?:proj-|live-)?[A-Za-z0-9_-]{20,}\b"), 0),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"), 0),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), 0),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        0,
    ),
    (
        "pem_private_key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.IGNORECASE,
        ),
        0,
    ),
    (
        "connection_string",
        re.compile(r"\b(?:mysql|postgres(?:ql)?|mongodb(?:\+srv)?|redis)://[^\s]+", re.IGNORECASE),
        0,
    ),
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|token|api[_-]?key|private[_-]?key)\s*[:=]\s*['\"]?([^\s,'\";]+)",
            re.IGNORECASE,
        ),
        1,
    ),
    (
        "bearer_token",
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
        0,
    ),
)
_LONG_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9_-]{32,}(?![A-Za-z0-9])")
_PLACEHOLDER_PATTERN = re.compile(
    r"^(?:your-[a-z0-9_-]+|replace-with-[a-z0-9_-]+|example(?:[-_a-z0-9]+)?|x{4,}|<[^>]+>)$",
    re.IGNORECASE,
)
_REDACTED_PLACEHOLDER_PATTERN = re.compile(r"^\[redacted:[a-z0-9_-]+\]$", re.IGNORECASE)


def _allowlist_patterns() -> list[re.Pattern[str]]:
    raw = settings.SECRET_SCAN_ALLOWLIST.strip()
    if not raw:
        return []
    patterns: list[re.Pattern[str]] = []
    for expression in raw.splitlines():
        expression = expression.strip()
        if not expression:
            continue
        try:
            patterns.append(re.compile(expression, re.IGNORECASE))
        except re.error as exc:
            raise ValueError("SECRET_SCAN_ALLOWLIST contains an invalid regular expression") from exc
    return patterns


def _is_allowlisted(value: str, extra_patterns: list[re.Pattern[str]]) -> bool:
    normalized = value.strip().lower()
    return (
        normalized == "example"
        or bool(_PLACEHOLDER_PATTERN.fullmatch(normalized))
        or bool(_REDACTED_PLACEHOLDER_PATTERN.fullmatch(normalized))
        or any(
        pattern.fullmatch(value) or pattern.search(value) for pattern in extra_patterns
        )
    )


def _entropy(value: str) -> float:
    counts = {character: value.count(character) for character in set(value)}
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _has_connection_credentials(value: str) -> bool:
    authority = value.split("://", 1)[1].split("/", 1)[0]
    return "@" in authority and ":" in authority.split("@", 1)[0]


def scan_text(text: str) -> list[Finding]:
    """Return secret findings using only type and character offsets."""

    if not isinstance(text, str) or not text:
        return []

    allowlist = _allowlist_patterns()
    findings: list[Finding] = []
    occupied: list[tuple[int, int]] = []

    def add_finding(secret_type: str, start: int, end: int, value: str) -> None:
        if _is_allowlisted(value, allowlist) or any(start < other_end and end > other_start for other_start, other_end in occupied):
            return
        findings.append({"type": secret_type, "start": start, "end": end})
        occupied.append((start, end))

    for secret_type, pattern, group in _PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(group)
            if secret_type == "connection_string" and not _has_connection_credentials(value):
                continue
            start, end = match.span(group)
            add_finding(secret_type, start, end, value)

    for match in _LONG_TOKEN_PATTERN.finditer(text):
        value = match.group(0)
        if _entropy(value) >= 4.0:
            add_finding("high_entropy_string", match.start(), match.end(), value)

    return sorted(findings, key=lambda item: (item["start"], item["end"]))


def redact(text: str) -> str:
    """Replace detected values without exposing the original text."""

    findings = scan_text(text)
    for finding in reversed(findings):
        replacement = f"[REDACTED:{finding['type']}]"
        text = text[: finding["start"]] + replacement + text[finding["end"] :]
    return text


def _scan_value(value: Any) -> tuple[str, list[Finding]]:
    text = value if isinstance(value, str) else str(value)
    return text, scan_text(text)


def guard_documents(
    documents: Iterable[tuple[str, str, Mapping[str, Any]]],
) -> tuple[list[tuple[str, str, dict[str, Any]]], list[FindingSummary]]:
    """Scan canonical documents before embedding and optionally redact them."""

    mode = settings.SECRET_SCAN_MODE
    if mode == "off":
        return [(source_id, text, dict(metadata)) for source_id, text, metadata in documents], []

    guarded: list[tuple[str, str, dict[str, Any]]] = []
    grouped: dict[tuple[str, str, str], int] = defaultdict(int)
    for source_id, text, metadata in documents:
        field_values: list[tuple[str, Any]] = [("text", text)]
        field_values.extend((f"metadata.{key}", value) for key, value in metadata.items())
        redacted_text = text
        redacted_metadata = dict(metadata)
        for field, value in field_values:
            original, findings = _scan_value(value)
            for finding in findings:
                grouped[(str(source_id), field, finding["type"])] += 1
            if mode == "redact" and findings:
                replacement = redact(original)
                if field == "text":
                    redacted_text = replacement
                else:
                    redacted_metadata[field.removeprefix("metadata.")] = replacement
        guarded.append((str(source_id), redacted_text, redacted_metadata))

    summaries = [
        FindingSummary(source_id=source_id, field=field, type=secret_type, count=count)
        for (source_id, field, secret_type), count in sorted(grouped.items())
    ]
    if mode == "block" and summaries:
        raise SecretScanError(summaries)
    return guarded, summaries


def format_scan_summaries(summaries: Iterable[FindingSummary]) -> str:
    lines = [
        f"source_id={item.source_id} field={item.field} type={item.type} count={item.count}"
        for item in summaries
    ]
    return "Secret scan findings:\n" + "\n".join(lines) if lines else "Secret scan: no findings."


def print_scan_summary(summaries: Iterable[FindingSummary]) -> None:
    summaries = list(summaries)
    if summaries:
        print(format_scan_summaries(summaries))
    else:
        print("Secret scan: no findings.")
