"""Read Google Docs and search Drive with your gcloud credentials.

Drive access needs `gcloud auth login --enable-gdrive-access`, which `export` and
`search` run when the current credentials fall short.
"""

import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime

EXPORT_URL = (
    "https://www.googleapis.com/drive/v3/files/{}/export?mimeType=text/markdown"
)
LIST_URL = "https://www.googleapis.com/drive/v3/files?{}"
IMAGE_LINE = re.compile(rb"^\[image\d+\]: <data:[^\n]*\n?", re.MULTILINE)
IMAGE_REF = re.compile(rb"!\[\]\[image\d+\]")


class GdocError(Exception):
    """Failure with a user-facing message; the CLI prints it and exits instead of
    unwinding a traceback."""


@dataclass(frozen=True)
class DriveFile:
    id: str
    name: str
    modified: datetime


def _access_token() -> str | None:
    proc = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        stdout=subprocess.PIPE,
        text=True,
    )
    return proc.stdout.strip() or None


def _login() -> str:
    print("Re-authenticating gcloud with Drive access", file=sys.stderr)
    subprocess.run(
        ["gcloud", "auth", "login", "--enable-gdrive-access", "--force"],
        check=True,
    )
    token = _access_token()
    if token is None:
        raise GdocError("gcloud produced no token after login")
    return token


def _read(url: str, token: str, subject: str) -> bytes | None:
    """The response body, or None if these credentials can't read it."""
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        body: bytes = urllib.request.urlopen(request).read()
        return body
    except urllib.error.HTTPError as err:
        detail = err.read().decode(errors="replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            detail = detail.strip()
        if err.code in (401, 403, 404):
            print(f"{subject}: HTTP {err.code} — {detail}", file=sys.stderr)
            return None
        raise GdocError(f"{subject}: HTTP {err.code} {err.reason} — {detail}") from err


def _authorized(url: str, subject: str) -> bytes:
    token = _access_token()
    body = _read(url, token, subject) if token else None
    if body is None:
        body = _read(url, _login(), subject)
    if body is None:
        raise GdocError(
            f"{subject}: not readable after re-auth — check that you are signed in "
            "on an account with Drive access to it"
        )
    return body


def export(doc: str) -> bytes:
    """A document's markdown. Inline base64 images become `[image not exported]`."""
    raw = _authorized(EXPORT_URL.format(doc), doc)
    return IMAGE_REF.sub(b"[image not exported]", IMAGE_LINE.sub(b"", raw))


def search(query: str) -> list[DriveFile]:
    """Files matching a Drive query. Raises past one 1000-result page."""
    params = urllib.parse.urlencode(
        {
            "q": query,
            "fields": "files(id,name,modifiedTime),nextPageToken",
            "pageSize": "1000",
        }
    )
    page = json.loads(_authorized(LIST_URL.format(params), query))
    if page.get("nextPageToken"):
        raise GdocError(f"{query}: more results than one page holds")
    return [
        DriveFile(f["id"], f["name"], datetime.fromisoformat(f["modifiedTime"]))
        for f in page["files"]
    ]
