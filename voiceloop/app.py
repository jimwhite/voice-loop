"""Briefcase entry-point for Voice Loop.

Briefcase looks for ``voiceloop.app:main`` when launching the macOS .app.
We delegate to the existing ``voice_loop_mac.main()`` so the rest of the
code-base stays untouched.
"""

from voice_loop_mac import main as _entry


def main():
    _entry()
