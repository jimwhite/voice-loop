"""Entry-point for ``python -m voiceloop`` (used by Briefcase launcher)."""

# ---------------------------------------------------------------------------
# SSL certificate bootstrap — must run before ANY import that may open an
# HTTPS connection (transformers, huggingface_hub, urllib, etc.).
#
# Briefcase bundles its own Python framework on macOS, which does not include
# the system certificate store.  Without this, every TLS handshake fails with
# ``ssl.SSLCertVerificationError: certificate verify failed: unable to get
# local issuer certificate``.
#
# We point the standard ``SSL_CERT_FILE`` env-var at the CA bundle shipped by
# the ``certifi`` package (already a transitive dependency).
# ---------------------------------------------------------------------------
import os as _os

if not _os.environ.get("SSL_CERT_FILE"):
    try:
        import certifi as _certifi
        _os.environ["SSL_CERT_FILE"] = _certifi.where()
    except ImportError:
        pass  # certifi unavailable — fall back to system defaults

# ---------------------------------------------------------------------------
# espeak-ng bootstrap — must run before kokoro_onnx is imported.
#
# The bundled espeakng_loader wheel ships libespeak-ng.dylib + data.  Force
# PHONEMIZER_ESPEAK_LIBRARY to point there so kokoro_onnx never falls back to
# a brew path that doesn't exist inside the Briefcase bundle.
# ---------------------------------------------------------------------------
try:
    import espeakng_loader as _espeakng
    _os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = _espeakng.get_library_path()
except ImportError:
    pass

from voiceloop.app import main

if __name__ == "__main__":
    main()
