"""Toga GUI for Voice Loop — on-device voice agent for Apple Silicon.

Provides a native macOS window with settings, Run/Stop controls,
and a scrolling log area.  Model downloads are deferred until the
user presses **Run**.
"""

import os
import threading
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
        if self._loop_thread:
            self._loop_thread.join(timeout=3)
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
        self._set_status("Stopping\u2026")

    # ── Thread-safe helpers ───────────────────────────────────────────

    def _log_msg(self, text):
        self.loop.call_soon_threadsafe(self._append_log, text)

    def _append_log(self, text):
        self.log.value += text + "\n"

    def _set_status(self, text):
        self.loop.call_soon_threadsafe(self._update_status, text)

    def _update_status(self, text):
        self.status.text = text

    def _finish(self):
        self._running = False
        self.run_btn.enabled = True
        self.stop_btn.enabled = False

    # ── Voice loop (runs in a daemon thread) ──────────────────────────

    def _voice_loop(self, s):          # noqa: C901  (complexity OK)
        import numpy as np

        try:
            self._log_msg("Loading models\u2026")
            self._set_status("Loading models\u2026")

            from voice_pipeline import (
                VoicePipeline, SAMPLE_RATE, CHUNK_SAMPLES, _vad_prob,
            )

            data_dir = Path(__file__).resolve().parent.parent
            pipeline = VoicePipeline(
                tts=s["tts"], smart_turn=s["smart_turn"], aec=s["aec"],
                chime=s["chime"], voice=s["voice"],
                silence_ms=s["silence_ms"], data_dir=data_dir,
            )

            if s["backend"] == "mcp-server":
                self._log_msg("MCP-server backend is not available in GUI mode.")
                return

            from backends.local import LocalBackend
            backend = LocalBackend(model_name=s["model"])

            self._log_msg("Models loaded.")
            self._set_status("Listening")

            mem_path = data_dir / "MEMORY.md"
            MAX_HISTORY = 10

            def _sys():
                sp = pipeline.load_system_prompt(include_memory=s["memory"])
                return [{"role": "system", "content": sp}] if sp else []

            # ─ greeting ─
            greeting = backend.generate(
                _sys() + [{"role": "user", "content": (
                    "Greet the user as Voice Loop in one short sentence. "
                    "If my name is in memory, use it and ask how you can help. "
                    "Otherwise, ask for my name."
                )}],
                max_tokens=60,
            )
            self._log_msg(f"> {greeting}")
            if pipeline.kokoro:
                pipeline.speak_tts_sync(greeting)

            history: list[dict] = []
            buf: list = []
            speaking, silent_chunks = False, 0

            with pipeline.start_mic_stream():
                while not self._stop_event.is_set():
                    try:
                        chunk = pipeline.audio_q.get(timeout=0.2)
                    except Exception:
                        continue
                    if len(chunk) < CHUNK_SAMPLES:
                        continue

                    prob = _vad_prob(pipeline.vad, chunk)
                    if prob > 0.5:
                        if not speaking:
                            speaking = True
                            self._log_msg("[listening\u2026]")
                        silent_chunks = 0
                        buf.append(chunk)
                    elif speaking:
                        silent_chunks += 1
                        buf.append(chunk)
                        if silent_chunks < pipeline.silence_limit:
                            continue
                        if pipeline.smart_turn_fn and buf:
                            tp = pipeline.smart_turn_fn(np.concatenate(buf))
                            self._log_msg(f"  [turn prob: {tp:.2f}]")
                            if tp < 0.5:
                                silent_chunks = 0
                                continue

                        # ── process utterance ──
                        audio = np.concatenate(buf)
                        self._log_msg(f"  ({len(audio) / SAMPLE_RATE:.1f}s)")
                        pipeline.play_chime()
                        try:
                            heard = pipeline.transcribe(audio)
                            self._log_msg(f"  [{heard}]")
                            msgs = _sys()
                            for h in history[-MAX_HISTORY:]:
                                msgs += [
                                    {"role": "user", "content": h["user"]},
                                    {"role": "assistant", "content": h["assistant"]},
                                ]
                            msgs.append({"role": "user", "content": heard})
                            resp = backend.generate(msgs)
                            self._log_msg(f"\n> {resp}")
                            if pipeline.kokoro and resp:
                                pipeline.play_tts_stream(resp)
                            else:
                                pipeline.stop_chime()
                            history.append({"user": heard, "assistant": resp})
                            if len(history) > MAX_HISTORY:
                                history.pop(0)
                            if s["memory"]:
                                def _rm():
                                    return mem_path.read_text() if mem_path.exists() else "# Memory\n"
                                backend.update_memory(heard, resp, mem_path, _rm)
                                if len(history) % 5 == 0:
                                    backend.consolidate_memory(mem_path, _rm)
                        except Exception as e:
                            self._log_msg(f"Error: {e}")
                        buf.clear()
                        speaking, silent_chunks = False, 0
                        pipeline.vad.reset_states()

            pipeline.shutdown()
            self._log_msg("Stopped.")
        except Exception as e:
            self._log_msg(f"Fatal: {e}")
            import traceback
            self._log_msg(traceback.format_exc())
        finally:
            self._set_status("Stopped")
            self.loop.call_soon_threadsafe(self._finish)


def main():
    """Return a new VoiceLoopApp.  Briefcase / ``__main__`` calls ``.main_loop()``."""
    return VoiceLoopApp("Voice Loop", "com.voiceloop.app")
