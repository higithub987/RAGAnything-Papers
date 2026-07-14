"""Milvus query-readiness gate for server startup.

Milvus's proxy accepts gRPC on 19530 (so ``describe_collection`` works) *before*
its QueryCoord/QueryNodes finish registering. During that window a
``load_collection`` call blocks indefinitely -- pymilvus polls the load progress
with ``timeout=None`` and never gives up -- and because that call runs
synchronously inside LightRAG's async storage init, it freezes the whole event
loop at "Waiting for application startup".

So before we let storage init touch any collection, we wait here until Milvus is
actually query-ready. The strong signal is Milvus's HTTP health endpoint on the
metrics port (9091) ``/healthz``, which only returns 200/OK once the components
are up. If that port isn't reachable at all (e.g. a remote/managed Milvus that
only exposes the gRPC port), we fall back to a plain gRPC connectivity probe so a
perfectly healthy deployment isn't blocked just because 9091 is closed.
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Milvus exposes its health endpoint on the metrics port. docker-compose.yml sets
# COMMON_METRICSPORT: 9091 and publishes 9091:9091, matching this default.
MILVUS_HEALTH_PORT = 9091


def _health_url(milvus_uri: str) -> str:
    """Derive http://<host>:9091/healthz from the configured Milvus URI."""
    host = urlparse(milvus_uri).hostname or "localhost"
    return f"http://{host}:{MILVUS_HEALTH_PORT}/healthz"


def _probe_healthz(url: str, timeout: float) -> str:
    """One health probe. Returns 'ready', 'not_ready', or 'unreachable'.

    'unreachable' distinguishes "couldn't connect to 9091 at all" (port likely
    not exposed) from "connected but Milvus reports unhealthy" ('not_ready'),
    which decides whether the gRPC fallback is worth trying.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read(512).decode("utf-8", "replace")
            # A 200 is the ready signal; Milvus returns "OK" (some builds return
            # a JSON body containing OK). Treat any 200 as ready.
            if resp.status == 200 and ("OK" in body.upper() or not body.strip()):
                return "ready"
            return "not_ready"
    except urllib.error.HTTPError:
        # Got an HTTP response (e.g. 503) -> server is up but not healthy yet.
        return "not_ready"
    except (urllib.error.URLError, OSError):
        # Connection refused / DNS / timeout -> nothing answered on 9091.
        return "unreachable"


def _probe_grpc(uri: str, timeout: float) -> bool:
    """Fallback probe: can we complete a cheap gRPC round-trip to Milvus?

    Weaker than /healthz (it only proves the proxy answers, not full query
    readiness), so it's used only when the health port never responded.
    """
    try:
        from pymilvus import MilvusClient
    except Exception:
        return False

    client = None
    try:
        client = MilvusClient(uri=uri, timeout=timeout)
        client.get_server_version(timeout=timeout)
        return True
    except Exception:
        return False
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


async def wait_for_milvus_ready(
    milvus_uri: str,
    timeout_s: float = 120.0,
    interval_s: float = 2.0,
) -> None:
    """Block (cooperatively) until Milvus is query-ready, or raise on timeout.

    Probes run in a worker thread via ``asyncio.to_thread`` so this never blocks
    the event loop. Raises ``RuntimeError`` if Milvus is neither healthy (9091)
    nor reachable over gRPC within ``timeout_s`` -- fail loudly instead of letting
    the later ``load_collection`` hang forever.
    """
    url = _health_url(milvus_uri)
    deadline = time.monotonic() + timeout_s
    saw_http_response = False
    attempt = 0

    while True:
        attempt += 1
        probe_timeout = min(interval_s if interval_s > 0 else 2.0, 5.0)
        status = await asyncio.to_thread(_probe_healthz, url, probe_timeout)
        if status == "ready":
            logger.info(
                "Milvus query-ready (%s OK) after %d attempt(s)", url, attempt
            )
            return
        if status == "not_ready":
            saw_http_response = True

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if attempt == 1 or attempt % 5 == 0:
            logger.info(
                "Waiting for Milvus to become query-ready (%s: %s) ...", url, status
            )
        await asyncio.sleep(min(interval_s, remaining))

    # Health endpoint never went green. If it never even answered, 9091 is
    # probably not exposed (remote/managed Milvus) -- accept a gRPC round-trip.
    if not saw_http_response and await asyncio.to_thread(_probe_grpc, milvus_uri, 5.0):
        logger.warning(
            "Milvus health port unreachable (%s); proceeding on successful gRPC "
            "probe to %s. If startup still stalls, expose port %d for a stronger "
            "readiness check.",
            url,
            milvus_uri,
            MILVUS_HEALTH_PORT,
        )
        return

    raise RuntimeError(
        f"Milvus not reachable/ready at {milvus_uri} (checked {url}) after "
        f"{timeout_s:.0f}s. Ensure the Milvus stack is up and healthy "
        f"(docker compose ps) before starting the API."
    )
