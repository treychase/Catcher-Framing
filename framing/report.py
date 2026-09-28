"""Writes the results into the self-contained web page."""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path

PLACEHOLDER = "/*__PAYLOAD__*/null"


def _clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def render(payload: dict, template: str | None = None) -> str:
    if template is None:
        template = (
            resources.files("framing")
            .joinpath("templates/catcher_framing_template.html")
            .read_text()
        )
    if PLACEHOLDER not in template:
        raise ValueError("template has no payload placeholder")
    blob = json.dumps(_clean(payload), separators=(",", ":"), allow_nan=False)
    # Keep a stray "</script>" in a name from closing the tag early.
    blob = blob.replace("</", "<\\/")
    return template.replace(PLACEHOLDER, blob)


def write(payload: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(payload))
    return path
