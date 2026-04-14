"""Briefcase entry-point for Voice Loop (console app with Typer CLI).

Briefcase looks for ``voiceloop.app:main`` when launching.
This module provides a Typer CLI that mirrors the options in
voice_loop_mac.main() and delegates to it via sys.argv.
"""

import sys
from typing import Optional

import typer

app = typer.Typer(
    name="voice-loop",
    help="Voice Loop — on-device voice agent for Apple Silicon.",
    add_completion=False,
)


@app.command()
def run(
    tts: bool = typer.Option(True, "--tts/--no-tts", help="Kokoro TTS output"),
    smart_turn: bool = typer.Option(
        True, "--smart-turn/--no-smart-turn", help="Smart Turn v3 endpoint detection"
    ),
    aec: bool = typer.Option(True, "--aec/--no-aec", help="WebRTC AEC3 voice interrupt"),
    chime: bool = typer.Option(
        True, "--chime/--no-chime", help="Chime + ticks while generating"
    ),
    memory: bool = typer.Option(False, "--memory", help="Read/write MEMORY.md"),
    audio_mode: bool = typer.Option(
        False, "--audio-mode", help="Send audio directly to Gemma (experimental)"
    ),
    model: str = typer.Option(
        "mlx-community/gemma-4-E4B-it-4bit", help="LLM model name"
    ),
    silence_ms: int = typer.Option(700, help="Silence duration (ms) before end-of-turn"),
    record: Optional[str] = typer.Option(
        None, "--record", help="Record mic to WAV file path"
    ),
    voice: str = typer.Option("af_heart", help="Kokoro TTS voice"),
    backend: str = typer.Option(
        "local", help="Backend: 'local' (Gemma on-device) or 'mcp-server'"
    ),
) -> None:
    """Start the Voice Loop agent."""
    # Build sys.argv for voice_loop_mac.main() which uses argparse internally
    argv = ["voice-loop"]
    if not tts:
        argv.append("--no-tts")
    if not smart_turn:
        argv.append("--no-smart-turn")
    if not aec:
        argv.append("--no-aec")
    if not chime:
        argv.append("--no-chime")
    if memory:
        argv.append("--memory")
    if audio_mode:
        argv.append("--audio-mode")
    argv.extend(["--model", model])
    argv.extend(["--silence-ms", str(silence_ms)])
    argv.extend(["--voice", voice])
    argv.extend(["--backend", backend])
    if record is not None:
        argv.extend(["--record", record])

    sys.argv = argv
    from voice_loop_mac import main as _entry

    _entry()


def main():
    """Entry-point called by ``voiceloop/__main__.py``."""
    app()
