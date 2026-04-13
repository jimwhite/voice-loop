#!/usr/bin/env python3
"""Voice Loop MCP Server — exposes voice I/O as MCP tools.

Coding agents (Copilot, Claude Code, Codex, Hermes) call these tools
to hear the user (voice_listen) and speak back (voice_speak).

Runs over stdio (the standard MCP transport for local tool servers).

Usage:
    uv run mcp_server.py                    # start MCP server (default)
    uv run voice_loop_mac.py --backend=mcp-server  # same, via main entry point

Integration examples:
    # VS Code (.vscode/mcp.json)
    { "servers": { "voice-loop": { "command": "uv", "args": ["run", "mcp_server.py"] } } }

    # Claude Code (~/.config/claude/mcp_servers.json)
    { "voice-loop": { "command": "uv", "args": ["run", "mcp_server.py"] } }
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
from pathlib import Path

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------
# Safety: cap speech length to prevent TTS resource exhaustion and limit
# the surface area for any text-based attacks. 2000 chars ≈ 30s of speech.
MAX_SPEAK_LENGTH = 2000  # characters
# Control chars except common whitespace (tab, newline, carriage return)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Module-level pipeline reference (initialised in run_mcp_server)
# ---------------------------------------------------------------------------
_pipeline = None
_mic_stream = None

# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Voice Loop",
    description=(
        "On-device voice I/O for coding agents. "
        "Captures speech via microphone, transcribes locally with Moonshine, "
        "and speaks responses via Kokoro TTS. No code execution — audio and text only."
    ),
)


def _sanitize_text(text: str) -> str:
    """Strip control characters and enforce length limit."""
    text = _CONTROL_CHAR_RE.sub("", text)
    if len(text) > MAX_SPEAK_LENGTH:
        text = text[:MAX_SPEAK_LENGTH]
    return text


@mcp.tool()
def voice_listen(timeout_secs: float | None = None) -> str:
    """Listen for one complete spoken utterance and return the transcription.

    Arms the microphone, waits for speech (using Silero VAD + Smart Turn),
    then transcribes with Moonshine STT.

    Parameters
    ----------
    timeout_secs : float, optional
        Maximum seconds to wait. Returns partial transcript on timeout,
        or empty string if no speech detected.

    Returns
    -------
    str
        The transcribed text from the user's speech.
    """
    if _pipeline is None:
        return "Error: Voice pipeline not initialised."
    result = _pipeline.listen_sync(timeout_secs=timeout_secs)
    return result if result else ""


@mcp.tool()
def voice_speak(text: str, voice: str | None = None, speed: float = 1.0) -> str:
    """Speak text aloud via Kokoro TTS.

    The text is sanitised (control characters stripped, max 2000 chars)
    before synthesis. No code is executed — only audio playback.

    Parameters
    ----------
    text : str
        The text to speak. Maximum 2000 characters.
    voice : str, optional
        Kokoro voice name (e.g. 'af_heart', 'bf_emma'). Uses server default if omitted.
    speed : float
        Playback speed multiplier (default 1.0).

    Returns
    -------
    str
        "ok" on success, or an error description.
    """
    if _pipeline is None:
        return "Error: Voice pipeline not initialised."
    if not text or not text.strip():
        return "Error: text is empty."

    text = _sanitize_text(text.strip())
    if not text:
        return "Error: text is empty after sanitisation."

    # Temporarily override voice if requested
    original_voice = _pipeline.voice
    if voice:
        _pipeline.voice = _sanitize_text(voice)
    try:
        _pipeline.speak_tts_sync(text)
    finally:
        _pipeline.voice = original_voice
    return "ok"


@mcp.tool()
def voice_listen_and_reply(
    reply_text: str,
    timeout_secs: float | None = None,
) -> str:
    """Listen for speech, then speak a reply — a single voice turn.

    Convenience tool that combines voice_listen + voice_speak.
    The agent provides reply_text which is spoken after the user finishes.

    Parameters
    ----------
    reply_text : str
        Text to speak back after hearing the user. Max 2000 chars.
    timeout_secs : float, optional
        Maximum seconds to wait for speech.

    Returns
    -------
    str
        The user's transcribed speech. The reply is spoken as a side-effect.
    """
    transcript = voice_listen(timeout_secs=timeout_secs)
    if transcript:
        voice_speak(reply_text)
    return transcript


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run_mcp_server(args=None):
    """Start the MCP server with a VoicePipeline running in the background.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Pre-parsed CLI args (from voice_loop_mac.py --backend=mcp-server).
        If None, parses its own CLI args.
    """
    global _pipeline, _mic_stream

    if args is None:
        ap = argparse.ArgumentParser(description="Voice Loop MCP Server")
        B = argparse.BooleanOptionalAction
        ap.add_argument("--tts", action=B, default=True, help="Enable Kokoro TTS")
        ap.add_argument("--smart-turn", action=B, default=True, help="Smart Turn v3")
        ap.add_argument("--aec", action=B, default=True, help="WebRTC AEC3")
        ap.add_argument("--chime", action=B, default=False, help="Chime sounds")
        ap.add_argument("--voice", default="af_heart", help="Kokoro voice")
        ap.add_argument("--silence-ms", type=int, default=700)
        args = ap.parse_args()

    # Redirect pipeline prints to stderr so stdout stays clean for MCP JSON
    from voice_pipeline import VoicePipeline

    _pipeline = VoicePipeline(
        tts=getattr(args, "tts", True),
        smart_turn=getattr(args, "smart_turn", True),
        aec=getattr(args, "aec", True),
        chime=getattr(args, "chime", False),
        voice=getattr(args, "voice", "af_heart"),
        silence_ms=getattr(args, "silence_ms", 700),
        data_dir=_DIR,
    )

    # Start mic stream in background — keeps sounddevice callback alive
    _mic_stream = _pipeline.start_mic_stream()
    _mic_stream.start()

    print("Voice Loop MCP server ready (stdio transport)", file=sys.stderr, flush=True)

    try:
        # Run FastMCP server on stdio
        mcp.run(transport="stdio")
    finally:
        _mic_stream.stop()
        _mic_stream.close()
        _pipeline.shutdown()


if __name__ == "__main__":
    run_mcp_server()
