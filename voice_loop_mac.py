#!/usr/bin/env python3
"""Voice Loop — a minimal on-device voice agent. Mac M4 / Apple Silicon.

Moonshine (CPU) transcribes speech. Gemma 4 E4B (Metal) responds.
Kokoro TTS speaks the response. WebRTC AEC3 enables voice interrupt.

Usage:
    uv run voice_loop_mac.py                        # defaults (TTS + smart turn + AEC)
    uv run voice_loop_mac.py --no-tts               # text out only
    uv run voice_loop_mac.py --no-aec               # keypress interrupt only
    uv run voice_loop_mac.py --chime                 # chime + ticks while generating
    uv run voice_loop_mac.py --backend=mcp-server    # MCP server mode (no local LLM)
"""

import argparse
import os
import sys
import time as _time
from pathlib import Path

import numpy as np

from voice_pipeline import VoicePipeline, SAMPLE_RATE, CHUNK_SAMPLES, save_wav, _vad_prob

MAX_HISTORY = 10
_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Core voice loop — shared by both GUI and CLI entry points
# ---------------------------------------------------------------------------

def run_loop(args, *, log=None, stop_check=None, allow_keypress=True):
    """Run the voice agent loop.

    Args:
        args:  Namespace with tts, smart_turn, aec, chime, voice, silence_ms,
               record, model, backend, memory, audio_mode, data_dir attributes.
        log:   ``log(text)`` callable for output.  Defaults to ``print``.
        stop_check:  Returns ``True`` when the loop should exit.
                     ``None`` → run until ``KeyboardInterrupt``.
        allow_keypress:  Use termios raw mode and keypress-to-interrupt TTS.
                         Set ``False`` for GUI (no stdin).
    """
    if log is None:
        log = lambda t, **kw: print(t, flush=True, **kw)  # noqa: E731

    data_dir = getattr(args, "data_dir", _DIR)

    # --- Build pipeline ---
    log("Loading models…")
    pipeline = VoicePipeline(
        tts=args.tts,
        smart_turn=args.smart_turn,
        aec=args.aec,
        chime=args.chime,
        voice=args.voice,
        silence_ms=args.silence_ms,
        record=getattr(args, "record", None),
        data_dir=data_dir,
    )

    # --- Build backend ---
    from backends.local import LocalBackend
    backend = LocalBackend(model_name=args.model)

    log("Models loaded.")

    _mem_path = data_dir / "MEMORY.md"

    def _read_memory():
        return _mem_path.read_text() if _mem_path.exists() else "# Memory\n"

    def _sys_messages():
        sp = pipeline.load_system_prompt(include_memory=args.memory)
        return [{"role": "system", "content": sp}] if sp else []

    def process_utterance(audio, history):
        log(f" ({len(audio) / SAMPLE_RATE:.1f}s)")
        pipeline.play_chime()
        audio_mode = getattr(args, "audio_mode", False)
        wav_path = save_wav(audio) if audio_mode else None
        try:
            messages = _sys_messages()
            for h in history[-MAX_HISTORY:]:
                messages += [{"role": "user", "content": h["user"]},
                             {"role": "assistant", "content": h["assistant"]}]
            if audio_mode:
                transcribe_future = pipeline.executor.submit(pipeline.transcribe, audio)
                messages.append({"role": "user", "content": [{"type": "audio"}]})
            else:
                heard = pipeline.transcribe(audio)
                log(f"  [{heard}]")
                messages.append({"role": "user", "content": heard})
            response = backend.generate(messages, **({"audio": [wav_path]} if audio_mode else {}))
            if audio_mode:
                heard = transcribe_future.result(timeout=10)
                log(f"  [{heard}]")
            log(f"\n> {response}\n")
            if pipeline.kokoro and response:
                pipeline.play_tts_stream(response, allow_keypress_interrupt=allow_keypress, stop_check=stop_check)
            else:
                pipeline.stop_chime()
            history.append({"user": heard, "assistant": response})
            if len(history) > MAX_HISTORY:
                history.pop(0)
            if args.memory:
                backend.update_memory(heard, response, _mem_path, _read_memory)
                if len(history) % 5 == 0:
                    backend.consolidate_memory(_mem_path, _read_memory)
        except Exception as e:
            print(f"\nError: {e}\n", file=sys.stderr)
        finally:
            if wav_path:
                os.unlink(wav_path)

    history: list[dict] = []
    buf: list[np.ndarray] = []
    speaking, silent_chunks = False, 0

    # --- Terminal raw mode (console only) ---
    old_term = None
    if allow_keypress:
        import termios, tty
        old_term = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    try:
        audio_mode = getattr(args, "audio_mode", False)
        mode = "audio" if audio_mode else "text"
        log(f"\nListening (mode: {mode}, tts: {args.tts}, silence: {args.silence_ms}ms, smart-turn: {args.smart_turn})")
        tts_hint = (" Speak or press any key to interrupt TTS." if args.aec else " Press any key to interrupt TTS.") if args.tts else ""
        log(f"Speak into your microphone. Ctrl+C to quit.{tts_hint}\n")

        greeting = backend.generate(_sys_messages() + [
            {"role": "user", "content": (
                "Greet the user as Voice Loop in one short sentence. "
                "If my name is in memory, use it and ask how you can help. "
                "Otherwise, ask for my name."
            )},
        ], max_tokens=60)
        log(f"> {greeting}\n")
        if pipeline.kokoro:
            pipeline.speak_tts_sync(greeting)

        with pipeline.start_mic_stream():
            while True:
                if stop_check and stop_check():
                    break
                try:
                    chunk = pipeline.audio_q.get(timeout=0.2)
                except Exception:
                    continue
                if len(chunk) < CHUNK_SAMPLES:
                    continue

                speech_prob = _vad_prob(pipeline.vad, chunk)
                if speech_prob > 0.5:
                    if not speaking:
                        speaking = True
                        log("[listening...]", end="")
                    silent_chunks = 0
                    buf.append(chunk)
                elif speaking:
                    silent_chunks += 1
                    buf.append(chunk)
                    if silent_chunks < pipeline.silence_limit:
                        continue
                    if pipeline.smart_turn_fn and buf:
                        prob = pipeline.smart_turn_fn(np.concatenate(buf))
                        log(f" [turn prob: {prob:.2f}]", end="")
                        if prob < 0.5:
                            silent_chunks = 0
                            continue
                    process_utterance(np.concatenate(buf), history)
                    buf.clear()
                    speaking, silent_chunks = False, 0
                    pipeline.vad.reset_states()

    except KeyboardInterrupt:
        log("\nBye!")
    finally:
        pipeline.shutdown()
        pipeline.save_recording()
        if old_term is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Voice Loop — a minimal on-device voice agent (Mac)")
    B = argparse.BooleanOptionalAction
    ap.add_argument("--tts", action=B, default=True, help="Kokoro TTS output")
    ap.add_argument("--smart-turn", action=B, default=True, help="Smart Turn v3 endpoint detection")
    ap.add_argument("--aec", action=B, default=True, help="WebRTC AEC3 voice interrupt")
    ap.add_argument("--chime", action=B, default=True,
                    help="Chime on utterance + soft ticks while generating (default: on)")
    ap.add_argument("--memory", action="store_true",
                    help="Read/write MEMORY.md (auto-update durable facts, consolidate every 5 turns)")
    ap.add_argument("--audio-mode", action="store_true", help="Send audio directly to Gemma (experimental)")
    ap.add_argument("--model", default="mlx-community/gemma-4-E4B-it-4bit")
    ap.add_argument("--silence-ms", type=int, default=700)
    ap.add_argument("--record", nargs="?", const="", metavar="FILE",
                    help="Record mic to WAV for debugging (default: tmp/recording-TIMESTAMP.wav)")
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice")
    ap.add_argument("--backend", choices=["local", "mcp-server"], default="local",
                    help="Backend: 'local' (Gemma on-device, default) or 'mcp-server' (MCP server mode)")
    args = ap.parse_args()

    if args.backend == "mcp-server":
        from mcp_server import run_mcp_server
        run_mcp_server(args)
        return

    if args.record == "":
        tmp_dir = _DIR / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        args.record = str(tmp_dir / f"recording-{_time.strftime('%Y%m%d-%H%M%S')}.wav")

    args.data_dir = _DIR
    run_loop(args)


if __name__ == "__main__":
    main()
