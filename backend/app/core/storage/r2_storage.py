"""
r2_storage.py
-------------
Backblaze B2 object storage client for TesseractRAG.

Replaces all local disk I/O in session_manager.py with B2 calls.
B2 is S3-compatible, so we use boto3 pointed at Backblaze's endpoint.

Storage layout in the B2 bucket:
    sessions/{session_id}/metadata.json   ← session identity + messages
    sessions/{session_id}/chunks.json     ← all text chunks
    sessions/{session_id}/faiss.index     ← FAISS binary index (raw bytes)

Why B2 instead of local disk?
    Render free tier has an ephemeral filesystem — every redeploy wipes it.
    B2 lives outside the container and survives restarts, redeployments,
    and container recycling. Free up to 10 GB.
"""

import json
import boto3
from botocore.exceptions import ClientError
from typing import Optional

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class R2Storage:
    """
    Thin wrapper around boto3 S3 client pointed at Backblaze B2.

    All methods operate on raw bytes and string keys.
    JSON serialization/deserialization happens in session_manager.py,
    not here — this class knows nothing about sessions or FAISS.

    Design principle: one responsibility.
        R2Storage      → move bytes in and out of B2
        SessionManager → decide WHAT to move and WHEN
    """

    def __init__(self):
        settings = get_settings()

        # boto3 S3 client configured for Backblaze B2.
        # Endpoint format: https://s3.{region}.backblazeb2.com
        # region_name must be set — boto3 requires it even though B2 ignores it.
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            region_name="auto",
        )
        self._bucket = settings.R2_BUCKET_NAME
        logger.info(f"R2Storage initialized | bucket={self._bucket}")

    # ── Core primitives ────────────────────────────────────────────────────────

    def put(self, key: str, data: bytes) -> None:
        """
        Upload raw bytes to B2 at the given key.

        Args:
            key:  Object key, e.g. "sessions/abc-123/metadata.json"
            data: Raw bytes to store.

        Raises:
            ClientError: If the upload fails (network, auth, bucket missing).
        """
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)
        logger.debug(f"R2 PUT {key} ({len(data)} bytes)")

    def get(self, key: str) -> Optional[bytes]:
        """
        Download raw bytes from B2 at the given key.

        Returns None if the key does not exist rather than raising,
        so callers can treat missing keys as "not yet created" cleanly.

        Args:
            key: Object key to retrieve.

        Returns:
            Raw bytes if the key exists, None if it does not.

        Raises:
            ClientError: On any error other than 404 (NoSuchKey).
        """
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            data = response["Body"].read()
            logger.debug(f"R2 GET {key} ({len(data)} bytes)")
            return data
        except ClientError as e:
            # 404 / NoSuchKey → key doesn't exist yet, return None
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise  # any other error (auth, network) → propagate

    def delete_prefix(self, prefix: str) -> None:
        """
        Delete all objects whose keys start with the given prefix.

        Used to delete all files for a session in one call:
            r2.delete_prefix("sessions/abc-123/")

        Does NOT use Delimiter — scans Contents directly for full
        Backblaze B2 compatibility. Paginates to handle any size.

        Args:
            prefix: Key prefix to match, e.g. "sessions/abc-123/"
        """
        paginator = self._client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self._bucket,
            Prefix=prefix,
            # No Delimiter — scan all keys directly for B2 compatibility
        )

        keys_to_delete = []
        for page in pages:
            for obj in page.get("Contents", []):
                keys_to_delete.append({"Key": obj["Key"]})

        if not keys_to_delete:
            logger.debug(f"R2 delete_prefix: no objects found under {prefix!r}")
            return

        self._client.delete_objects(
            Bucket=self._bucket,
            Delete={"Objects": keys_to_delete},
        )
        logger.info(f"R2 deleted {len(keys_to_delete)} object(s) under {prefix!r}")

    def list_session_ids(self) -> list[str]:
        """
        List all session IDs that exist in B2.

        Scans all objects under "sessions/" prefix and extracts
        unique session IDs from the key paths.

        Does NOT use Delimiter/CommonPrefixes — scans Contents directly
        for full Backblaze B2 compatibility.

        Returns:
            List of session ID strings, e.g. ["abc-123", "def-456"].
        """
        paginator = self._client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self._bucket,
            Prefix="sessions/",
            # No Delimiter — B2 compatibility: scan all keys and extract IDs manually
        )

        session_ids = set()
        for page in pages:
            for obj in page.get("Contents", []):
                # Key looks like: "sessions/abc-123/metadata.json"
                # Split on "/" → ["sessions", "abc-123", "metadata.json"]
                parts = obj["Key"].split("/")
                if len(parts) >= 3 and parts[0] == "sessions":
                    session_ids.add(parts[1])

        ids = list(session_ids)
        logger.info(f"R2 found {len(ids)} session(s)")
        return ids

    # ── Convenience helpers ────────────────────────────────────────────────────
    # These wrap put/get with JSON encoding so session_manager.py
    # doesn't need to think about bytes ↔ dict conversion.

    def put_json(self, key: str, data: dict | list) -> None:
        """Serialize data to JSON bytes and upload to B2."""
        raw = json.dumps(data, indent=2).encode("utf-8")
        self.put(key, raw)

    def get_json(self, key: str) -> Optional[dict | list]:
        """Download from B2 and deserialize JSON. Returns None if key missing."""
        raw = self.get(key)
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))