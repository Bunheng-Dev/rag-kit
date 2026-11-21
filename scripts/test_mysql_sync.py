"""Opt-in real MySQL/Qdrant/chat integration test.

Start dependencies first:
    docker compose --profile selfhost --profile mysql-test up -d

Then run this script from the repository root. It is intentionally outside
``tests/`` so it is not part of the default unit-test or CI path.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pymysql
from qdrant_client import QdrantClient

from app.core.config import settings


ROOT = Path(__file__).resolve().parents[1]
MYSQL_HOST = os.getenv("MYSQL_TEST_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_TEST_PORT", "3307"))
MYSQL_USER = os.getenv("MYSQL_TEST_USER", "rag_kit")
MYSQL_PASSWORD = os.getenv("MYSQL_TEST_PASSWORD", "rag_kit_pw")
MYSQL_DATABASE = os.getenv("MYSQL_TEST_DATABASE", "rag_kit")
QDRANT_URL = os.getenv("QDRANT_TEST_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_TEST_API_KEY", "")
COLLECTION = "rag_kit_mysql_test"
API_URL = "http://127.0.0.1:8000"
CHAT_KEY = os.getenv("CHAT_INTERNAL_KEY", settings.CHAT_INTERNAL_KEY)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _safe_error(exc: Exception) -> str:
    message = str(exc)
    for secret in (MYSQL_PASSWORD, QDRANT_API_KEY, settings.GEMINI_API_KEY, CHAT_KEY):
        if secret:
            message = message.replace(secret, "[redacted]")
    return message[:240] or type(exc).__name__


def _connect_mysql():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor,
        charset="utf8mb4",
        connect_timeout=3,
    )


def wait_for_mysql() -> None:
    last_error: Exception | None = None
    for _ in range(60):
        try:
            connection = _connect_mysql()
            connection.close()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"MySQL did not become healthy: {_safe_error(last_error or RuntimeError())}")


def load_seed() -> None:
    sql = (ROOT / "data" / "mysql_seed.sql").read_text(encoding="utf-8")
    connection = _connect_mysql()
    try:
        with connection.cursor() as cursor:
            for statement in sql.split(";"):
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)
        connection.commit()
    finally:
        connection.close()


def build_test_environment() -> dict[str, str]:
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    environment.update(
        {
            "DATASOURCE": "mysql",
            "MYSQL_HOST": MYSQL_HOST,
            "MYSQL_PORT": str(MYSQL_PORT),
            "MYSQL_USER": MYSQL_USER,
            "MYSQL_PASSWORD": MYSQL_PASSWORD,
            "MYSQL_DATABASE": MYSQL_DATABASE,
            "MYSQL_DOCUMENT_TABLE": "documents",
            "MYSQL_DOCUMENT_ID_COLUMN": "id",
            "MYSQL_DOCUMENT_TEXT_COLUMN": "text",
            "MYSQL_DOCUMENT_METADATA_COLUMNS": "category,price",
            "QDRANT_URL": QDRANT_URL,
            "QDRANT_API_KEY": QDRANT_API_KEY,
            "QDRANT_COLLECTION": COLLECTION,
            "APP_PORT": "8000",
            "CHAT_INTERNAL_KEY": CHAT_KEY,
        }
    )
    if settings.GEMINI_API_KEY and "GEMINI_API_KEY" not in environment:
        environment["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
    return environment


def run_sync(environment: dict[str, str]) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/sync_mysql_to_qdrant.py"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sync command exited with code {result.returncode}")


def wait_for_qdrant(client: QdrantClient) -> None:
    last_error: Exception | None = None
    for _ in range(30):
        try:
            client.get_collections()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Qdrant did not become reachable: {_safe_error(last_error or RuntimeError())}")


def verify_points(client: QdrantClient) -> set[str]:
    count = client.count(collection_name=COLLECTION, exact=True).count
    if count != 3:
        raise AssertionError(f"expected 3 points, found {count}")

    points, _ = client.scroll(collection_name=COLLECTION, limit=10, with_payload=True, with_vectors=False)
    expected_ids = {str(uuid5(NAMESPACE_URL, f"{COLLECTION}:{document_id}")) for document_id in (1, 2, 3)}
    actual_ids = {str(point.id) for point in points}
    if actual_ids != expected_ids:
        raise AssertionError("point IDs do not match deterministic source IDs")

    payloads = [point.payload or {} for point in points]
    if not any("swimming pool" in str(payload.get("text", "")) and payload.get("price") == 120 for payload in payloads):
        raise AssertionError("expected product text and price metadata were not found")
    if not any(payload.get("category") == "experience" and "05:00" in str(payload.get("text", "")) for payload in payloads):
        raise AssertionError("expected service text and category metadata were not found")
    return actual_ids


def wait_for_api(process: subprocess.Popen[bytes]) -> None:
    last_error: Exception | None = None
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError(f"API process exited with code {process.returncode}")
        try:
            with OPENER.open(f"{API_URL}/health", timeout=3) as response:
                health = json.loads(response.read().decode("utf-8"))
            if health.get("vector_database_status") == "ready":
                return
            last_error = RuntimeError("API health is not ready")
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"API did not become ready: {_safe_error(last_error or RuntimeError())}")


def ask(question: str) -> str:
    request = urllib.request.Request(
        f"{API_URL}/api/v1/chat/stream",
        data=json.dumps({"message": question}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Chat-API-Key": CHAT_KEY},
        method="POST",
    )
    with OPENER.open(request, timeout=120) as response:
        raw = response.read().decode("utf-8")

    parts: list[str] = []
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
                value = json.loads(chunk)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(value, dict):
                    chunk = value.get("text") or value.get("content") or value.get("token") or chunk
        parts.append(chunk)
    return "".join(parts) if parts else raw


def run_step(label: str, function) -> tuple[bool, object | None]:
    try:
        result = function()
        print(f"PASS: {label}")
        return True, result
    except Exception as exc:
        print(f"FAIL: {label} ({_safe_error(exc)})")
        return False, None


def main() -> int:
    environment = build_test_environment()
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=10)
    api_process: subprocess.Popen[bytes] | None = None
    all_passed = True

    try:
        passed, _ = run_step("wait for MySQL health", wait_for_mysql)
        all_passed &= passed
        if not passed:
            return 1

        passed, _ = run_step("load MySQL seed", load_seed)
        all_passed &= passed
        if not passed:
            return 1

        passed, _ = run_step("wait for Qdrant", lambda: wait_for_qdrant(client))
        all_passed &= passed
        if not passed:
            return 1

        passed, _ = run_step("sync MySQL rows into Qdrant", lambda: run_sync(environment))
        all_passed &= passed
        if not passed:
            return 1

        passed, first_ids = run_step("verify 3 points and payload metadata", lambda: verify_points(client))
        all_passed &= passed
        if not passed:
            return 1

        api_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        passed, _ = run_step("start API with MySQL collection", lambda: wait_for_api(api_process))
        all_passed &= passed
        if not passed:
            return 1

        passed, answer = run_step(
            "answer a MySQL-only chat question",
            lambda: ask("What is the nightly price of the product with a swimming pool?"),
        )
        all_passed &= passed
        print(">> DEBUG answer: ", repr(answer)) 
        if not passed or "120" not in str(answer):
            if passed:
                print("FAIL: MySQL-only chat answer did not contain the expected keyword (120)")
            all_passed = False
            return 1

        passed, _ = run_step("sync the same rows a second time", lambda: run_sync(environment))
        all_passed &= passed
        if not passed:
            return 1
        passed, second_ids = run_step("confirm idempotent point count and IDs", lambda: verify_points(client))
        all_passed &= passed and second_ids == first_ids
        if passed and second_ids != first_ids:
            print("FAIL: deterministic point IDs changed between sync runs")
        return 0 if all_passed else 1
    finally:
        if api_process is not None and api_process.poll() is None:
            api_process.terminate()
            try:
                api_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                api_process.kill()
        cleanup_passed, _ = run_step(
            "drop test collection",
            lambda: client.delete_collection(collection_name=COLLECTION)
            if client.collection_exists(COLLECTION)
            else None,
        )
        if not cleanup_passed:
            all_passed = False


if __name__ == "__main__":
    raise SystemExit(main())
