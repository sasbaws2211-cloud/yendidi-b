"""Hosts the on-device face embedding model.

The model still runs on the phone — the photo never reaches this server. It is
served from here, rather than bundled into the app, only to keep the install
small and to let the model change without an app release. Whatever is served
must match a version in `SUPPORTED_FACE_MODELS` (schools.py): embeddings from
different models are not comparable.

Where the file comes from:
  * `models/<version>.onnx` if it is already on disk and matches the pinned hash;
  * otherwise it is downloaded at startup from `FACE_MODEL_URL` and verified
    against `FACE_MODEL_SHA256` before it is ever served.
A host with an ephemeral disk (Render's free tier) loses the file on every
redeploy or spin-down, so it re-downloads on each cold start; the endpoints
answer 503 until that finishes rather than serving a partial file.

Environment:
  FACE_MODEL_PATH    override the on-disk location
  FACE_MODEL_URL     where to fetch the file from when it is missing
  FACE_MODEL_SHA256  expected SHA-256 (required with FACE_MODEL_URL)
"""

import asyncio
import hashlib
import logging
import os
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter()

FACE_MODEL_VERSION = "virtuoturing-v1"
FACE_MODEL_PATH = Path(
    os.environ.get("FACE_MODEL_PATH")
    or Path(__file__).resolve().parents[2] / "models" / f"{FACE_MODEL_VERSION}.onnx"
)
FACE_MODEL_URL = os.environ.get("FACE_MODEL_URL") or None
FACE_MODEL_SHA256 = (os.environ.get("FACE_MODEL_SHA256") or "").lower() or None

# "missing" | "downloading" | "ready" | "failed"
_state = {"status": "missing", "sha256": None, "detail": ""}


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> str:
    """Streams to a temp file and only moves it into place once verified."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30, read=120)) as res:
        res.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in res.iter_bytes(1 << 20):
                digest.update(chunk)
                fh.write(chunk)
    actual = digest.hexdigest()
    if FACE_MODEL_SHA256 and actual != FACE_MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"face model hash mismatch: expected {FACE_MODEL_SHA256}, got {actual}")
    os.replace(tmp, dest)
    return actual


def _ensure_blocking() -> None:
    if FACE_MODEL_PATH.is_file():
        actual = _sha256_of(FACE_MODEL_PATH)
        if not FACE_MODEL_SHA256 or actual == FACE_MODEL_SHA256:
            _state.update(status="ready", sha256=actual, detail="")
            return
        logger.warning("face model on disk does not match FACE_MODEL_SHA256; replacing it")
    if not FACE_MODEL_URL:
        _state.update(status="missing", detail="Face model is not installed on this server.")
        return
    if not FACE_MODEL_SHA256:
        # Never serve an unverified binary to every phone.
        _state.update(status="failed", detail="FACE_MODEL_SHA256 must be set when FACE_MODEL_URL is used.")
        return
    _state.update(status="downloading", detail="Face model is being prepared; try again shortly.")
    try:
        _state.update(status="ready", sha256=_download(FACE_MODEL_URL, FACE_MODEL_PATH), detail="")
        logger.info("face model downloaded and verified")
    except Exception as exc:  # noqa: BLE001 - surfaced via the endpoints, not raised into startup
        logger.error("face model download failed: %s", exc)
        _state.update(status="failed", detail="Face model could not be prepared on this server.")


async def ensure_face_model() -> None:
    """Called once at startup, as a background task so a slow download never
    holds the API's health check hostage."""
    await asyncio.to_thread(_ensure_blocking)


def _require_ready() -> None:
    if _state["status"] == "ready":
        return
    code = {
        "missing": status.HTTP_404_NOT_FOUND,
        "downloading": status.HTTP_503_SERVICE_UNAVAILABLE,
        "failed": status.HTTP_503_SERVICE_UNAVAILABLE,
    }.get(_state["status"], status.HTTP_503_SERVICE_UNAVAILABLE)
    raise HTTPException(code, detail=_state["detail"] or "Face model is not available.")


@router.get("/face-model")
async def face_model_manifest() -> dict:
    _require_ready()
    return {
        "model_version": FACE_MODEL_VERSION,
        "sha256": _state["sha256"],
        "size_bytes": FACE_MODEL_PATH.stat().st_size,
    }


@router.get("/face-model/file")
async def face_model_file() -> FileResponse:
    _require_ready()
    # "identity" keeps GZipMiddleware off this response: ONNX weights barely
    # compress, and a gzipped stream has no exact Content-Length to check.
    return FileResponse(
        FACE_MODEL_PATH,
        media_type="application/octet-stream",
        filename=f"{FACE_MODEL_VERSION}.onnx",
        headers={"Content-Encoding": "identity"},
    )
