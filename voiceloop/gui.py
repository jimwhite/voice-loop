"""Toga GUI for Voice Loop — on-device voice agent for Apple Silicon.

Provides a native macOS window with settings, Run/Stop controls,
and a scrolling log area.  Model downloads are deferred until the
user presses **Run**.

The voice loop itself runs the same ``run_loop()`` function as the CLI —
only the I/O hooks (logging, stop signal, keypress) differ.
"""

import argparse
import asyncio
import os
import sys
import threading
import traceback
from pathlib import Path

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

MODEL_PRESETS = [
    "mlx-community/gemma-4-E4B-it-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Phi-4-mini-instruct-4bit",
]


class VoiceLoopApp(toga.App):

    # ── UI ────────────────────────────────────────────────────────────

    def startup(self):
        self._stop_event = threading.Event()
        self._loop_thread = None
        self._running = False

        # ── Settings ──────────────────────────────────────────────────
        settings = toga.Box(style=Pack(direction=COLUMN, padding=10))

        # HF Token
        row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        row.add(toga.Label("HF Token:", style=Pack(width=100, padding_top=5)))
        self.hf_token = toga.PasswordInput(
            placeholder="hf_…",
            value=os.environ.get("HF_TOKEN", ""),
            style=Pack(flex=1),
        )
        row.add(self.hf_token)
        settings.add(row)

        # Model
        row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        row.add(toga.Label("Model:", style=Pack(width=100, padding_top=5)))
        self.model_input = toga.TextInput(
            value=MODEL_PRESETS[0], style=Pack(flex=1),
        )
        row.add(self.model_input)
        self.model_select = toga.Selection(
            items=MODEL_PRESETS,
            on_change=self._on_model_preset,
            style=Pack(width=200, padding_left=5),
        )
        row.add(self.model_select)
        settings.add(row)

        # Voice + Silence
        row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        row.add(toga.Label("Voice:", style=Pack(width=100, padding_top=5)))
        self.voice_input = toga.TextInput(value="af_heart", style=Pack(flex=1))
        row.add(self.voice_input)
        row.add(toga.Label("  Silence (ms):", style=Pack(padding_top=5)))
        self.silence_input = toga.TextInput(value="700", style=Pack(width=60, padding_left=5))
        row.add(self.silence_input)
        settings.add(row)

        # Toggles
        row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        self.tts_sw = toga.Switch("TTS", value=True, style=Pack(padding_right=15))
        self.st_sw = toga.Switch("Smart Turn", value=True, style=Pack(padding_right=15))
        self.aec_sw = toga.Switch("AEC", value=True, style=Pack(padding_right=15))
        self.chime_sw = toga.Switch("Chime", value=True, style=Pack(padding_right=15))
        self.mem_sw = toga.Switch("Memory", value=False)
        for w in (self.tts_sw, self.st_sw, self.aec_sw, self.chime_sw, self.mem_sw):
            row.add(w)
        settings.add(row)

        # Backend
        row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        row.add(toga.Label("Backend:", style=Pack(width=100, padding_top=5)))
        self.backend_sel = toga.Selection(items=["local", "mcp-server"], style=Pack(width=150))
        row.add(self.backend_sel)
        settings.add(row)

        # ── Control buttons ───────────────────────────────────────────
        btn_box = toga.Box(style=Pack(direction=ROW, padding=10))
        self.run_btn = toga.Button(
            "Run", on_press=self._on_run, style=Pack(flex=1, padding_right=5),
        )
        self.stop_btn = toga.Button(
            "Stop", on_press=self._on_stop, enabled=False,
            style=Pack(flex=1, padding_left=5),
        )
        btn_box.add(self.run_btn)
        btn_box.add(self.stop_btn)

        # ── Status + log ─────────────────────────────────────────────
        self.status = toga.Label("Ready", style=Pack(padding=(0, 10)))
        self.log = toga.MultilineTextInput(
            readonly=True,
            style=Pack(flex=1, padding=10),
        )

        # ── Main layout ──────────────────────────────────────────────
        outer = toga.Box(style=Pack(direction=COLUMN, flex=1))
        outer.add(settings)
        outer.add(toga.Divider())
        outer.add(btn_box)
        outer.add(self.status)
        outer.add(self.log)

        self.main_window = toga.MainWindow(title="Voice Loop", size=(720, 600))
        self.main_window.content = outer
        self.main_window.show()

    def on_exit(self):
        self._stop_event.set()
        # Immediately stop any sd.play()/sd.wait() playback (e.g. greeting TTS)
        # so the voice-loop thread can proceed to its cleanup path.
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        # Wait for the voice-loop thread to finish its cleanup: closing the
        # sd.OutputStream (portaudio native thread), the mic InputStream, and
        # calling pipeline.shutdown().  Without this, native portaudio threads
        # keep the process alive, the resource_tracker pipe stays open, and
        # the child process lingers.
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)
        return True

    # ── Widget callbacks ──────────────────────────────────────────────

    def _on_model_preset(self, widget):
        self.model_input.value = str(widget.value)

    def _on_run(self, widget):
        if self._running:
            return
        token = self.hf_token.value.strip()
        if token:
            os.environ["HF_TOKEN"] = token

        self._running = True
        self._stop_event.clear()
        self.run_btn.enabled = False
        self.stop_btn.enabled = True
        self.log.value = ""
        self._set_status("Starting\u2026")

        try:
            silence = int(self.silence_input.value)
        except (ValueError, TypeError):
            silence = 700

        cfg = dict(
            model=self.model_input.value.strip(),
            voice=self.voice_input.value.strip(),
            silence_ms=silence,
            tts=self.tts_sw.value,
            smart_turn=self.st_sw.value,
            aec=self.aec_sw.value,
            chime=self.chime_sw.value,
            memory=self.mem_sw.value,
            backend=str(self.backend_sel.value),
        )

        self._loop_thread = threading.Thread(
            target=self._voice_loop, args=(cfg,), daemon=True,
        )
        self._loop_thread.start()

    def _on_stop(self, widget):
        self._stop_event.set()
        # Stop sd.play()/sd.wait() immediately (greeting TTS).
        # play_tts_stream uses its own OutputStream and checks stop_check.
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._set_status("Stopping\u2026")

    # ── Thread-safe helpers ───────────────────────────────────────────

    def _log_msg(self, text, **_kw):
        print(text, file=sys.stderr, flush=True)
        self.loop.call_soon_threadsafe(self._append_log, text)

    def _append_log(self, text):
        self.log.value += text + "\n"

    def _set_status(self, text):
        self.loop.call_soon_threadsafe(self._update_status, text)

    def _update_status(self, text):
        self.status.text = text

    def _show_error_dialog(self, title, message):
        """Show a native error dialog on the main thread."""
        async def _show():
            await self.main_window.error_dialog(title, message)
        self.loop.call_soon_threadsafe(self.loop.create_task, _show())

    def _finish(self):
        self._running = False
        self.run_btn.enabled = True
        self.stop_btn.enabled = False

    # ── Voice loop (daemon thread) ──────────────────────────────────────

    def _voice_loop(self, s):
        """Thread target: builds an argparse Namespace and delegates to run_loop().

        Toga's cocoa backend installs a process-wide Cocoa event-loop policy
        (rubicon-objc).  play_tts_stream() calls asyncio.run() which needs a
        standard selector-based event loop — not a Cocoa one that requires a
        CFRunLoop on the current thread.  Resetting the policy here is safe
        because Toga's main-thread loop is already created and running.
        """
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

        try:
            self._set_status("Loading models\u2026")

            args = argparse.Namespace(
                tts=s["tts"],
                smart_turn=s["smart_turn"],
                aec=s["aec"],
                chime=s["chime"],
                voice=s["voice"],
                silence_ms=s["silence_ms"],
                memory=s["memory"],
                model=s["model"],
                backend=s["backend"],
                audio_mode=False,
                record=None,
                data_dir=Path(__file__).resolve().parent.parent,
            )

            from voice_loop_mac import run_loop

            run_loop(
                args,
                log=self._log_msg,
                stop_check=self._stop_event.is_set,
                allow_keypress=False,
            )
        except Exception as e:
            tb = traceback.format_exc()
            self._log_msg(f"Fatal: {e}")
            self._log_msg(tb)
            self._show_error_dialog("Voice Loop Error", str(e))
        finally:
            self._set_status("Stopped")
            self.loop.call_soon_threadsafe(self._finish)


def main():
    """Return a new VoiceLoopApp.  Briefcase / ``__main__`` calls ``.main_loop()``."""
    return VoiceLoopApp("Voice Loop", "com.voiceloop.app")
