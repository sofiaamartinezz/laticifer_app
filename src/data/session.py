"""Versioned JSON persistence for resumable analysis sessions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from skimage import io as skio


SESSION_FORMAT = "latexlens-analysis-session"
SESSION_VERSION = 1


class SessionError(ValueError):
    """Raised when a session file is invalid or incomplete."""


@dataclass(frozen=True)
class SessionData:
    image_path: str
    mask_path: str = ""
    dataset_root: str = ""
    initialized_from_model: bool = False
    um_per_px: Optional[float] = None
    scale_source: str = "pixels_only"
    scale_reference_pixels: Optional[float] = None
    scale_reference_length_um: Optional[float] = None
    transect_num_lines: int = 10
    transect_direction: str = "horizontal"
    transect_lines: list[list[list[float]]] = field(default_factory=list)
    active_tab: int = 0

    def validated(self) -> "SessionData":
        if not str(self.image_path).strip():
            raise SessionError("The session does not specify an image path.")
        if self.um_per_px is not None:
            value = float(self.um_per_px)
            if not math.isfinite(value) or value <= 0:
                raise SessionError("The session contains an invalid image scale.")
        if self.transect_direction not in ("horizontal", "vertical", "both"):
            raise SessionError("The session contains an invalid transect direction.")
        if int(self.transect_num_lines) < 1:
            raise SessionError("The session contains an invalid transect count.")
        if not 0 <= int(self.active_tab) <= 3:
            raise SessionError("The session contains an invalid active tab.")
        for line in self.transect_lines:
            try:
                array = np.asarray(line, dtype=float)
            except (TypeError, ValueError) as exc:
                raise SessionError("The session contains invalid transect geometry.") from exc
            if array.shape != (2, 2) or not np.all(np.isfinite(array)):
                raise SessionError("The session contains invalid transect geometry.")
        return self


def save_session(
    session_path: Path,
    session: SessionData,
    mask_data: Optional[np.ndarray] = None,
) -> Path:
    """Write a session JSON and, when provided, its editable mask sidecar."""
    session_path = Path(session_path).resolve()
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session = session.validated()
    payload = asdict(session)

    if mask_data is not None:
        mask_path = session_path.with_name(f"{session_path.stem}_mask.tif")
        skio.imsave(mask_path, (np.asarray(mask_data) > 0).astype(np.uint8) * 255)
        payload["mask_path"] = mask_path.name

    payload["image_path"] = _portable_path(session.image_path, session_path.parent)
    if payload.get("mask_path"):
        payload["mask_path"] = _portable_path(payload["mask_path"], session_path.parent)
    if payload.get("dataset_root"):
        payload["dataset_root"] = _portable_path(payload["dataset_root"], session_path.parent)

    document = {
        "format": SESSION_FORMAT,
        "version": SESSION_VERSION,
        "session": payload,
    }
    temporary = session_path.with_suffix(session_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(session_path)
    return session_path


def load_session(session_path: Path) -> SessionData:
    """Read, validate and resolve paths from a session JSON."""
    session_path = Path(session_path).resolve()
    try:
        document = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SessionError(f"Could not read the session file: {exc}") from exc
    if document.get("format") != SESSION_FORMAT:
        raise SessionError("This is not a LatexLens analysis session.")
    if document.get("version") != SESSION_VERSION:
        raise SessionError(
            f"Unsupported session version: {document.get('version')!r}."
        )
    try:
        payload: dict[str, Any] = dict(document["session"])
        for key in ("image_path", "mask_path", "dataset_root"):
            if payload.get(key):
                payload[key] = str(_resolve_path(payload[key], session_path.parent))
        session = SessionData(**payload).validated()
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SessionError):
            raise
        raise SessionError(f"Invalid session data: {exc}") from exc

    if not Path(session.image_path).is_file():
        raise SessionError(f"Image file not found: {session.image_path}")
    if session.mask_path and not Path(session.mask_path).is_file():
        raise SessionError(f"Mask file not found: {session.mask_path}")
    return session


def _portable_path(value: str, base: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()
