"""End-to-end RAG test using data/sample.json.

Steps:
  1. Check /health
  2. (optional) ingest sample data:  --ingest
  3. Ask questions whose answers exist ONLY in the sample data
  4. Check the streamed answer contains the expected keywords

Run (Git Bash):
  export DATASOURCE=file
  PYTHONPATH=. python scripts/test_rag_sample.py --ingest

Env vars:
  RAG_KIT_BASE_URL   default http://127.0.0.1:8000
  CHAT_INTERNAL_KEY  value of X-Chat-API-Key (read from .env if not set)
"""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

from app.core.config import settings

BASE_URL = os.getenv("RAG_KIT_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("CHAT_INTERNAL_KEY", settings.CHAT_INTERNAL_KEY)

# Bypass system proxy / VPN for localhost
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# (question, keywords - ANY one must appear in the answer)
TEST_CASES = [
    ("How much per night is the product with a swimming pool?", ["120"]),
    ("What time does the guided service start and what is included?", ["05:00", "5:00", "guide", "transport"]),
    ("What information can customers check before booking?", ["cancellation", "payment", "opening hours"]),
]


def get_json(path: str) -> dict:
    with OPENER.open(f"{BASE_URL}{path}", timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def ask(question: str) -> str:
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat/stream",
        data=json.dumps({"message": question}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Chat-API-Key": API_KEY},
        method="POST",
    )
    with OPENER.open(req, timeout=120) as r:
        raw = r.read().decode("utf-8")

    # Collect text from SSE "data:" lines (JSON or plain text)
    parts = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        chunk = line[5:]
        if chunk.startswith(" "):
            chunk = chunk[1:]
        if chunk.strip() == "[DONE]":
            continue
        if chunk.lstrip().startswith("{"):
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(obj, dict):
                    chunk = obj.get("text") or obj.get("content") or obj.get("token") or chunk
        parts.append(chunk)
    return "".join(parts) if parts else raw


def main() -> int:
    if "--ingest" in sys.argv:
        print("Ingesting data/sample.json ...")
        env = {**os.environ, "DATASOURCE": "file", "PYTHONPATH": "."}
        subprocess.run([sys.executable, "scripts/ingest_sample.py"], check=True, env=env)

    try:
        health = get_json("/health")
    except Exception as exc:
        print(f"FAIL: cannot reach {BASE_URL}/health -> {exc}")
        print("Is uvicorn running? Is the port correct?")
        return 1
    print(f"health: {health}")
    if health.get("vector_database_status") != "ready":
        print("FAIL: vector database is not ready (Qdrant paused or wrong URL/key?)")
        return 1

    passed = 0
    for question, keywords in TEST_CASES:
        print(f"\nQ: {question}")
        try:
            answer = ask(question)
        except urllib.error.HTTPError as exc:
            print(f"  FAIL: HTTP {exc.code} {exc.reason}")
            continue
        print(f"A: {answer.strip()[:400]}")
        ok = any(k.lower() in answer.lower() for k in keywords)
        print(f"  {'PASS' if ok else 'FAIL'} (expected one of: {keywords})")
        passed += ok

    print(f"\nResult: {passed}/{len(TEST_CASES)} passed")
    return 0 if passed == len(TEST_CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
