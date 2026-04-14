"""Shared bootstrap for Briefcase-bundled Voice Loop.

Must run before ANY import that opens HTTPS connections or loads kokoro_onnx.
"""

import os
import sys


def _log(msg):
    print(f"[bootstrap] {msg}", file=sys.stderr, flush=True)


def bootstrap():
    # ── SSL certificate bootstrap ──
    # Briefcase bundles its own Python without the system certificate store.
    if not os.environ.get("SSL_CERT_FILE"):
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass

    # ── espeak-ng bootstrap ──
    # Force PHONEMIZER_ESPEAK_LIBRARY to the bundled espeakng_loader path
    # so kokoro_onnx never falls back to a brew path that doesn't exist
    # inside the .app bundle.
    try:
        import espeakng_loader
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeakng_loader.get_library_path()
    except ImportError:
        pass

    # ── Disable multiprocessing resource tracker ──
    # In a Briefcase macOS app, sys.executable is the stub binary, which
    # runs runpy._run_module_as_main() and ignores Python's -c flag.
    # When resource_tracker.ensure_running() fork+exec's sys.executable
    # with '-c "from multiprocessing.resource_tracker import main;main(fd)"',
    # the stub re-launches the entire Toga GUI — causing a second Dock icon.
    #
    # The Python framework in the bundle is a dylib (not an executable),
    # so there's no real Python interpreter to point multiprocessing at.
    #
    # This app uses threads (not multiprocessing workers), so the resource
    # tracker serves no purpose.  Disable it entirely — the OS reclaims
    # all resources on process exit.
    try:
        import multiprocessing.resource_tracker as _rt
        tracker = _rt._resource_tracker
        # Prevent the tracker from fork+exec'ing the Briefcase stub
        tracker.ensure_running = lambda: None
        # No-op register/unregister so callers don't error writing to a closed fd
        tracker.register = lambda name, rtype: None
        tracker.unregister = lambda name, rtype: None
        # Also remove __del__ to prevent deadlock at shutdown (cpython#88887)
        if hasattr(_rt.ResourceTracker, "__del__"):
            del _rt.ResourceTracker.__del__
        _log("Disabled multiprocessing resource tracker (Briefcase stub workaround)")
    except Exception as e:
        _log(f"Resource tracker patch error: {e}")
