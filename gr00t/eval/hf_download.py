"""Small retry wrapper for checkpoint downloads; no evaluation/queue logic."""

import argparse
import random
import sys
import time
from email.utils import parsedate_to_datetime

import requests
from huggingface_hub import constants, snapshot_download


class DownloadDeferred(RuntimeError):
    """Retry budget exhausted: keep cached files and let the caller continue."""


def download_checkpoint(repo_id, subfolder, revision, local_dir, attempts=6):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    constants.HF_HUB_DOWNLOAD_TIMEOUT = 60
    constants.HF_HUB_ETAG_TIMEOUT = 60
    for attempt in range(attempts):
        try:
            return snapshot_download(
                repo_id,
                revision=revision,
                allow_patterns=[f"{subfolder}/*"],
                local_dir=local_dir,
                max_workers=1,
                etag_timeout=60,
            )
        except Exception as error:
            # The Hub sometimes wraps the original timeout in a cache error.
            cause, seen, retryable, status, headers = error, set(), False, None, {}
            while cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                response = getattr(cause, "response", None)
                if response is not None:
                    status, headers = response.status_code, response.headers
                    retryable = status in (408, 429) or 500 <= status < 600
                    break
                if isinstance(
                    cause,
                    (
                        requests.Timeout,
                        requests.ConnectionError,
                        requests.exceptions.ChunkedEncodingError,
                        TimeoutError,
                    ),
                ):
                    retryable = True
                    break
                cause = cause.__cause__ or cause.__context__
            if not retryable:
                raise  # Auth, missing checkpoint, disk errors: do not blindly retry.
            delay = max(min(300, 30 * 2**attempt), 300 if status == 429 else 0)
            value = headers.get("Retry-After", "0")
            try:
                server_delay = float(value)
            except ValueError:
                try:
                    server_delay = parsedate_to_datetime(value).timestamp() - time.time()
                except (TypeError, ValueError, OverflowError):
                    server_delay = 0
            delay = max(delay, server_delay) + random.uniform(0, 10)
            print(
                f"HF download failed ({type(error).__name__}, HTTP {status}); "
                f"attempt {attempt + 1}/{attempts}, waiting {delay:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            # Cool down even on the final failure, before the caller tries its next checkpoint.
            time.sleep(delay)
            if attempt + 1 == attempts:
                raise DownloadDeferred("HF download deferred; cached partial files retained") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("repo-id", "subfolder", "revision", "local-dir"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    try:
        download_checkpoint(**vars(args))
    except DownloadDeferred:
        print("Download deferred; record it separately and continue with the next checkpoint.", file=sys.stderr)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
