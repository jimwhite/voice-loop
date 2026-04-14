"""Entry-point for ``python -m voiceloop`` (Briefcase GUI launcher)."""

import sys
import traceback

try:
    from voiceloop._bootstrap import bootstrap
    bootstrap()

    from voiceloop.gui import main  # noqa: E402  (must run after bootstrap)

    if __name__ == "__main__":
        main().main_loop()
except Exception:
    # Ensure fatal startup errors are visible on stderr (system log for .app)
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    raise
