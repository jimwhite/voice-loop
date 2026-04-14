"""Entry-point for ``python -m voiceloop_cli`` (Briefcase console launcher)."""

from voiceloop._bootstrap import bootstrap

bootstrap()

from voiceloop.app import main  # noqa: E402  (must run after bootstrap)

if __name__ == "__main__":
    main()
