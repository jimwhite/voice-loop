"""Entry-point for ``python -m voiceloop`` (Briefcase GUI launcher)."""

from voiceloop._bootstrap import bootstrap

bootstrap()

from voiceloop.gui import main  # noqa: E402  (must run after bootstrap)

if __name__ == "__main__":
    main().main_loop()
