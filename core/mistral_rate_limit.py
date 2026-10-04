import threading
import time


_MIN_REQUEST_INTERVAL = 1.0
_request_lock = threading.Lock()
_last_request_at = 0.0


def invoke_with_rate_limit(chain, value):
    global _last_request_at

    with _request_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

        _last_request_at = time.monotonic()
        return chain.invoke(value)