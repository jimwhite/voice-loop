"""VoicePipeline — encapsulates mic capture, VAD, Smart Turn, STT (Moonshine),
TTS (Kokoro), and AEC (WebRTC AEC3 via LiveKit APM).

Exposes two primary async methods:
    listen()  → str    — wait for one complete utterance, return transcription
    speak(text) → None — synthesize and play text via Kokoro TTS

Also provides synchronous helpers used by both the standalone loop and MCP server.
"""

import asyncio
import os
import queue
import select
import sys
import tempfile
import time as _time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sounddevice as sd

sd.default.latency = "high"
import torch

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz (required by Silero VAD)
CHIME_SR = 24000


def _fade_tone(freq, dur, amp=0.6):
    """Tone with raised-cosine (Hann) envelope — smooth fade in/out, no clicks."""
    n = int(dur * CHIME_SR)
    t = np.linspace(0, dur, n, dtype=np.float32)
    env = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    return amp * np.sin(2 * np.pi * freq * t) * env


def _silence(dur):
    return np.zeros(int(dur * CHIME_SR), dtype=np.float32)


def make_chime(duration=30.0, tick_every=1.5):
    """Two-tone chime + periodic short ticks. Single buffer → one sd.play()."""
    head = np.concatenate(
        [_fade_tone(880, 0.09), _silence(0.03), _fade_tone(1320, 0.10)]
    )
    tick = _fade_tone(550, 0.04, amp=0.18)
    total = int(duration * CHIME_SR)
    buf = np.zeros(total, dtype=np.float32)
    buf[: len(head)] = head
    step = int(tick_every * CHIME_SR)
    for pos in range(len(head), total, step):
        end = min(pos + len(tick), total)
        buf[pos:end] = tick[: end - pos]
    return buf


def _lang_from_voice(v: str) -> str:
    """Infer Kokoro lang code from voice prefix."""
    prefix = v[:1] if len(v) > 1 and v[1] == "_" else ""
    return {
        "a": "en-us",
        "b": "en-gb",
        "e": "es",
        "f": "fr-fr",
        "h": "hi",
        "i": "it",
        "j": "ja",
        "p": "pt-br",
        "z": "cmn",
    }.get(prefix, "en-us")


def save_wav(audio, sr=SAMPLE_RATE):
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(
            (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        )
    return path


def load_smart_turn():
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor

    model_path = os.path.join(
        tempfile.gettempdir(), "smart_turn_v3", "smart_turn_v3.2_cpu.onnx"
    )
    if not os.path.exists(model_path):
        print("Downloading Smart Turn v3.2 model...", flush=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(
            "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx",
            model_path,
        )
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    def predict(audio_float32: np.ndarray) -> float:
        max_samples = 8 * SAMPLE_RATE
        audio_float32 = audio_float32[-max_samples:]
        features = extractor(
            audio_float32,
            sampling_rate=SAMPLE_RATE,
            max_length=max_samples,
            padding="max_length",
            return_attention_mask=False,
            return_tensors="np",
        )
        return float(
            session.run(
                None,
                {"input_features": features.input_features.astype(np.float32)},
            )[0]
            .flatten()[0]
        )

    return predict


def _vad_prob(vad, chunk):
    p = vad(torch.from_numpy(chunk), SAMPLE_RATE)
    return p.item() if hasattr(p, "item") else p


def _get_ref_segment(tts_concat, pos, length):
    if pos >= len(tts_concat):
        return np.zeros(length, dtype=np.float32)
    seg = tts_concat[pos : pos + length]
    return (
        np.concatenate([seg, np.zeros(length - len(seg), dtype=np.float32)])
        if len(seg) < length
        else seg
    )


class VoicePipeline:
    """Encapsulates mic capture, VAD, Smart Turn, STT, TTS, and AEC.

    Parameters
    ----------
    tts : bool
        Enable Kokoro TTS.
    smart_turn : bool
        Enable Smart Turn v3 endpoint detection.
    aec : bool
        Enable WebRTC AEC3 echo cancellation.
    chime : bool
        Play chime on utterance + soft ticks while generating.
    voice : str
        Kokoro voice name (e.g. 'af_heart').
    silence_ms : int
        Silence timeout in milliseconds.
    record : str | None
        Path to save mic recording, or None.
    data_dir : Path | None
        Directory for SOUL.md/MEMORY.md. Defaults to script directory.
    """

    def __init__(
        self,
        *,
        tts: bool = True,
        smart_turn: bool = True,
        aec: bool = True,
        chime: bool = True,
        voice: str = "af_heart",
        silence_ms: int = 700,
        record: str | None = None,
        data_dir: Path | None = None,
    ):
        self.tts_enabled = tts
        self.aec_enabled = aec
        self.chime_enabled = chime
        self.voice = voice
        self.silence_ms = silence_ms
        self.record_path = record
        self.data_dir = data_dir or Path(__file__).parent
        self.silence_limit = max(
            1, int(silence_ms / (CHUNK_SAMPLES / SAMPLE_RATE * 1000))
        )

        # --- Load models ---
        print("Loading Silero VAD...", flush=True)
        from silero_vad import load_silero_vad

        self.vad = load_silero_vad(onnx=True)

        print("Loading Moonshine (transcription)...", flush=True)
        from moonshine_voice import Transcriber, get_model_for_language

        ms_path, ms_arch = get_model_for_language("en")
        self.moonshine = Transcriber(model_path=str(ms_path), model_arch=ms_arch)

        self.smart_turn_fn = load_smart_turn() if smart_turn else None

        self.kokoro = None
        if tts:
            print("Loading Kokoro TTS...", flush=True)
            import subprocess

            try:
                prefix = subprocess.check_output(
                    ["brew", "--prefix", "espeak-ng"], text=True
                ).strip()
                os.environ.setdefault(
                    "PHONEMIZER_ESPEAK_LIBRARY",
                    f"{prefix}/lib/libespeak-ng.dylib",
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
            from kokoro_onnx import Kokoro

            cache_dir = os.path.join(tempfile.gettempdir(), "kokoro_tts")
            model_file = os.path.join(cache_dir, "kokoro-v1.0.onnx")
            voices_file = os.path.join(cache_dir, "voices-v1.0.bin")
            if not os.path.exists(model_file):
                os.makedirs(cache_dir, exist_ok=True)
                import urllib.request

                base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
                print("  Downloading kokoro model (~300MB)...", flush=True)
                urllib.request.urlretrieve(f"{base}/kokoro-v1.0.onnx", model_file)
                urllib.request.urlretrieve(f"{base}/voices-v1.0.bin", voices_file)
            self.kokoro = Kokoro(model_file, voices_file)

        self._make_aec_processor = None
        if aec:
            from livekit.rtc import AudioFrame
            from livekit.rtc.apm import AudioProcessingModule

            WF = 160  # 10ms @ 16kHz

            def _to_i16(x):
                s = (x * 32767).clip(-32768, 32767).astype(np.int16)
                return np.pad(s, (0, max(0, WF - len(s)))) if len(s) < WF else s

            def _frame(b):
                return AudioFrame(
                    b.tobytes(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=WF,
                )

            def make_aec():
                apm = AudioProcessingModule(
                    echo_cancellation=True, noise_suppression=True
                )

                def process(mic, ref):
                    cleaned = np.zeros_like(mic)
                    for i in range(0, len(mic), WF):
                        mic_f = _frame(_to_i16(mic[i : i + WF]))
                        apm.process_reverse_stream(
                            _frame(_to_i16(ref[i : i + WF]))
                        )
                        apm.process_stream(mic_f)
                        cleaned[i : i + WF] = (
                            np.frombuffer(bytes(mic_f.data), dtype=np.int16).astype(
                                np.float32
                            )
                            / 32767
                        )[: len(mic[i : i + WF])]
                    return cleaned

                return process

            self._make_aec_processor = make_aec
            print("  AEC: WebRTC AEC3 (LiveKit APM)")

        self.chime_sound = make_chime() if chime else None
        self.audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self.record_buf: list[np.ndarray] | None = [] if record else None
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Chime timing
        self._chime_started_at = 0.0

        # Listen result — used by async listen()
        self._listen_result: asyncio.Future | None = None

    # --- Audio callbacks ---

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        chunk = indata[:, 0].copy()
        if self.record_buf is not None:
            self.record_buf.append(chunk)
        self.audio_q.put(chunk)

    def drain_audio_q(self):
        while not self.audio_q.empty():
            self.audio_q.get_nowait()

    # --- Transcription ---

    def transcribe(self, audio_data: np.ndarray) -> str:
        return " ".join(
            l.text
            for l in self.moonshine.transcribe_without_streaming(
                audio_data.tolist(), SAMPLE_RATE
            ).lines
            if l.text
        ).strip()

    # --- TTS ---

    def speak_tts_sync(self, text: str) -> None:
        """Speak text synchronously (blocking). No voice-interrupt support."""
        if not self.kokoro:
            return
        samples, sr = self.kokoro.create(
            text,
            voice=self.voice,
            speed=1.0,
            lang=_lang_from_voice(self.voice),
        )
        sd.play(samples, sr)
        sd.wait()

    def play_tts_stream(self, response: str, allow_keypress_interrupt: bool = True) -> bool:
        """Stream TTS with AEC and voice-interrupt support. Returns True if interrupted."""
        if not self.kokoro:
            return False
        self.drain_audio_q()
        tts_stream = self.kokoro.create_stream(
            response,
            voice=self.voice,
            speed=1.0,
            lang=_lang_from_voice(self.voice),
        )
        out_stream, interrupted = None, False
        tts_16k_buf: list[np.ndarray] = []
        state = {"play_start": None, "consec_speech": 0, "mic_pos": 0}
        aec_process = self._make_aec_processor() if self._make_aec_processor else None

        def check_barge_in():
            if not (aec_process and state["play_start"] and tts_16k_buf):
                return False
            if _time.monotonic() - state["play_start"] < 0.5:
                return False
            tts_concat = np.concatenate(tts_16k_buf)
            while not self.audio_q.empty():
                mic_chunk = self.audio_q.get_nowait()
                if len(mic_chunk) < CHUNK_SAMPLES:
                    continue
                ref = _get_ref_segment(tts_concat, state["mic_pos"], len(mic_chunk))
                state["mic_pos"] += len(mic_chunk)
                cleaned = aec_process(mic_chunk, ref)
                if _vad_prob(self.vad, cleaned.astype(np.float32)) > 0.8:
                    state["consec_speech"] += 1
                    if state["consec_speech"] >= 5:
                        return True
                else:
                    state["consec_speech"] = 0
            return False

        async def _play():
            nonlocal out_stream, interrupted
            async for chunk_samples, sr in tts_stream:
                if out_stream is None:
                    if self.chime_sound is not None:
                        self._wait_for_chime_gap()
                        sd.stop()
                    out_stream = sd.OutputStream(
                        samplerate=sr, channels=1, dtype="float32"
                    )
                    out_stream.start()
                    self.drain_audio_q()
                    self.vad.reset_states()
                    state["play_start"] = _time.monotonic()
                if aec_process is not None:
                    if sr == SAMPLE_RATE:
                        tts_16k_buf.append(chunk_samples.astype(np.float32))
                    else:
                        idx = np.arange(0, len(chunk_samples), sr / SAMPLE_RATE)
                        tts_16k_buf.append(
                            np.interp(
                                idx,
                                np.arange(len(chunk_samples)),
                                chunk_samples,
                            ).astype(np.float32)
                        )
                data = chunk_samples.reshape(-1, 1)
                for i in range(0, len(data), 4096):
                    if allow_keypress_interrupt and select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.read(1)
                        interrupted = True
                    elif check_barge_in():
                        interrupted = True
                        print("  [voice interrupt]", flush=True)
                    if interrupted:
                        break
                    out_stream.write(data[i : i + 4096])
                if interrupted:
                    break
            if out_stream:
                out_stream.stop()
                out_stream.close()

        asyncio.run(_play())
        if interrupted and state["consec_speech"] < 3:
            print("  [interrupted]")
        self.drain_audio_q()
        self.vad.reset_states()
        return interrupted

    # --- Chime support ---

    def play_chime(self):
        if self.chime_sound is not None:
            print("  *chime*", flush=True)
            sd.play(self.chime_sound, CHIME_SR)
            self._chime_started_at = _time.monotonic()

    def stop_chime(self):
        if self.chime_sound is not None:
            self._wait_for_chime_gap()
            sd.stop()

    def _wait_for_chime_gap(self):
        """Wait until we're in a silent gap between ticks."""
        if self.chime_sound is None or self._chime_started_at == 0:
            return
        CHIME_HEAD = 0.22
        TICK_DUR = 0.04
        TICK_EVERY = 1.5
        t = _time.monotonic() - self._chime_started_at
        if t < CHIME_HEAD:
            _time.sleep(CHIME_HEAD - t)
            return
        phase = (t - CHIME_HEAD) % TICK_EVERY
        if phase < TICK_DUR:
            _time.sleep(TICK_DUR - phase + 0.005)

    # --- System prompt / memory ---

    def load_system_prompt(self, include_memory: bool = False) -> str:
        names = ("SOUL.md", "MEMORY.md") if include_memory else ("SOUL.md",)
        parts = [
            (self.data_dir / n).read_text().strip()
            for n in names
            if (self.data_dir / n).exists()
        ]
        return "\n\n".join(p for p in parts if p)

    # --- Blocking listen (used by main loop and MCP) ---

    def listen_sync(self, timeout_secs: float | None = None) -> str | None:
        """Block until one complete utterance is detected and transcribed.

        Returns the transcribed string, or None on timeout.
        """
        buf: list[np.ndarray] = []
        speaking = False
        silent_chunks = 0
        deadline = _time.monotonic() + timeout_secs if timeout_secs else None

        while True:
            if deadline and _time.monotonic() > deadline:
                # If we have partial audio, try to transcribe it
                if buf:
                    audio = np.concatenate(buf)
                    return self.transcribe(audio)
                return None
            try:
                chunk = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if len(chunk) < CHUNK_SAMPLES:
                continue

            speech_prob = _vad_prob(self.vad, chunk)
            if speech_prob > 0.5:
                if not speaking:
                    speaking = True
                    print("[listening...]", end="", flush=True)
                silent_chunks = 0
                buf.append(chunk)
            elif speaking:
                silent_chunks += 1
                buf.append(chunk)
                if silent_chunks < self.silence_limit:
                    continue
                if self.smart_turn_fn and buf:
                    prob = self.smart_turn_fn(np.concatenate(buf))
                    print(f" [turn prob: {prob:.2f}]", end="", flush=True)
                    if prob < 0.5:
                        silent_chunks = 0
                        continue
                audio = np.concatenate(buf)
                print(f" ({len(audio) / SAMPLE_RATE:.1f}s)")
                text = self.transcribe(audio)
                print(f"  [{text}]")
                self.vad.reset_states()
                return text

    # --- Async wrappers ---

    async def listen(self, timeout_secs: float | None = None) -> str | None:
        """Async wrapper around listen_sync — runs STT in a thread."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.listen_sync, timeout_secs
        )

    async def speak(self, text: str) -> None:
        """Async wrapper around speak_tts_sync."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.speak_tts_sync, text)

    # --- Mic stream context ---

    def start_mic_stream(self):
        """Return a sounddevice InputStream (caller should use as context manager)."""
        return sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=self._audio_callback,
        )

    def save_recording(self):
        """Save recorded mic audio to self.record_path if recording was enabled."""
        if self.record_path and self.record_buf:
            full = np.concatenate(self.record_buf)
            with wave.open(self.record_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(
                    (full * 32767)
                    .clip(-32768, 32767)
                    .astype(np.int16)
                    .tobytes()
                )
            print(
                f"Recorded {len(full) / SAMPLE_RATE:.1f}s to {self.record_path}",
                flush=True,
            )

    def shutdown(self):
        self.executor.shutdown(wait=False)
