"""Shared bootstrap for Briefcase-bundled Voice Loop.

Must run before ANY import that opens HTTPS connections or loads kokoro_onnx.
"""

import os


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
