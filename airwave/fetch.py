"""One-time model downloads, and the reason this is not a two-line urlopen.

Airwave makes no network calls while running (G5). These are explicit setup
steps the user asks for by name - the MediaPipe hand model, and a speech model
for dictation.

The complication is trust stores. Python's ssl module verifies against either
OpenSSL's bundle or the Windows certificate store, and on a machine running a
TLS-intercepting antivirus or corporate proxy it can fail to verify *every*
host: the interceptor's root CA is often malformed by OpenSSL 3.x's standards
("Basic Constraints of CA cert not marked critical") even though the operating
system itself trusts it fine. The result is a Python process that cannot reach
anything while the browser and curl on the same machine work perfectly.

So: try Python first, and on a certificate failure fall back to curl, which
uses the platform's own TLS stack. This is not a verification bypass - curl
still validates the chain, just against the store the machine actually uses.
There is no code path here that skips verification.
"""

from __future__ import annotations

import logging
import shutil
import ssl
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("airwave.fetch")


class DownloadError(RuntimeError):
    """Raised with the manual fallback spelled out."""


def _is_certificate_error(exc: BaseException) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    reason = getattr(exc, "reason", None)
    if isinstance(reason, ssl.SSLError):
        return True
    return "CERTIFICATE_VERIFY_FAILED" in str(exc)


def _curl(url: str, dest: Path, timeout: int) -> bool:
    """Download with the system TLS stack. Returns False if curl is unusable."""
    curl = shutil.which("curl")
    if curl is None:
        return False
    log.info("python could not verify the TLS chain; retrying with curl")
    result = subprocess.run(
        [curl, "-fsSL", "--max-time", str(timeout), "-o", str(dest), url],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        log.debug("curl failed: %s", result.stderr.decode("utf-8", "replace").strip())
        return False
    return dest.exists() and dest.stat().st_size > 0


def download_file(url: str, dest: str | Path, *, timeout: int = 300) -> Path:
    """Fetch one file to ``dest``, atomically.

    Writes to a ``.part`` file and renames on success, so an interrupted
    download can never leave a truncated model that fails later with a far more
    confusing error than "download interrupted".
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as fh:  # noqa: S310
            shutil.copyfileobj(response, fh)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        if not _is_certificate_error(exc) or not _curl(url, tmp, timeout):
            tmp.unlink(missing_ok=True)
            raise DownloadError(
                f"could not download {url}\n"
                f"  {type(exc).__name__}: {exc}\n"
                f"  Download it by hand and put it at: {dest}"
            ) from exc

    tmp.replace(dest)
    return dest
