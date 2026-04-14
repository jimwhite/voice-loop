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
        # Patch the class so ALL instances (current and future) are disabled.
        # Instance-attribute patching on the singleton is insufficient because
        # tqdm → multiprocessing.synchronize calls the module-level register()
        # which resolves through the class method, not the instance attribute.
        _rt.ResourceTracker.ensure_running = lambda self: None
        _rt.ResourceTracker.register = lambda self, name, rtype: None
        _rt.ResourceTracker.unregister = lambda self, name, rtype: None
        _rt.ResourceTracker._send = lambda self, cmd, name, rtype: None
        if hasattr(_rt.ResourceTracker, "__del__"):
            del _rt.ResourceTracker.__del__
        # Also patch the module-level convenience functions
        _rt.ensure_running = lambda: None
        _rt.register = lambda name, rtype: None
        _rt.unregister = lambda name, rtype: None
        _log("Disabled multiprocessing resource tracker (Briefcase stub workaround)")
    except Exception as e:
        _log(f"Resource tracker patch error: {e}")
