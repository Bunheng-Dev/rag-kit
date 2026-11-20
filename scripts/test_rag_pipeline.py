"""Small smoke test for the public FastAPI contract.

Run this against a configured, running API. It intentionally does not embed
credentials or assume a particular dataset, provider, or domain.
"""

import json
import os
import urllib.error
import urllib.request

from app.core.config import settings


BASE_URL = os.getenv("RAG_KIT_BASE_URL", f"http://127.0.0.1:{settings.APP_PORT}")
API_KEY = os.getenv("CHAT_INTERNAL_KEY", settings.CHAT_INTERNAL_KEY)


def request_json(path: str, payload: dict | None = None, headers: dict | None = None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main() -> None:
    status, health = request_json("/health")
    assert status == 200 and health["status"] == "online"
    print(f"health: {health}")

    try:
        request_json("/api/chat/stream", {"message": "hello"})
    except urllib.error.HTTPError as exc:
        assert exc.code == 401
    else:
        raise AssertionError("missing chat key was not rejected")

    try:
        request_json(
            "/api/chat/stream",
            {"unexpected": True},
            {"X-Chat-API-Key": API_KEY},
        )
    except urllib.error.HTTPError as exc:
        assert exc.code == 422
    else:
        raise AssertionError("invalid payload was not rejected")

    request = urllib.request.Request(
        f"{BASE_URL}/api/chat/stream",
        data=json.dumps({"message": "What information is available?"}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Chat-API-Key": API_KEY},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        stream = response.read().decode("utf-8")
    assert "data: [DONE]" in stream
    print("chat stream: [DONE] received")


if __name__ == "__main__":
    main()
