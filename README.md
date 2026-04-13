# Voice Loop

A minimal on-device voice agent loop. Runs entirely on Mac M4 / Apple Silicon.

Now also works as an **MCP server** — plug it into Copilot, Claude Code, Codex, or any MCP-compatible coding agent.

> Need a custom voice model or production voice agent? See [Trelis Voice AI Services](https://trelis.com/voice-ai-services/).

## Features

- **Smart turn detection** — Silero VAD + pipecat's Smart Turn v3, so the agent waits when you pause mid-sentence
- **Voice interruption** — speak over the agent; WebRTC AEC3 cancels echo from speakers so your voice cuts through
- **Editable persona** — `SOUL.md` controls the agent's style, live-reloaded each turn
- **Optional long-term memory** — enable with `--memory`; the agent learns durable facts about you in `MEMORY.md` and consolidates every 5 turns
- **Fully local** — no API keys, no cloud. Everything runs on-device
- **MCP server mode** — expose voice I/O as tools for coding agents (Copilot, Claude Code, Codex, Hermes)
- **macOS app packaging** — distributable as a `.dmg` with App Sandbox and microphone entitlements

## Stack

- **Moonshine** (CPU) for speech-to-text transcription
- **Gemma 4 E4B** (MLX/Metal) for response generation (standalone mode)
- **Kokoro** (CPU) for TTS (streaming)
- **Silero VAD** + **Smart Turn v3** for turn detection
- **WebRTC AEC3** (via LiveKit APM) for voice interruption
- **FastMCP** for MCP server protocol

## Setup

```bash
brew install portaudio espeak-ng
git clone https://github.com/jimwhite/voice-loop.git
cd voice-loop
uv sync
```

First run downloads Gemma 4 E4B (~3GB), Moonshine (~250MB), Kokoro (~300MB).

## Usage

### Standalone mode (local LLM)

```bash
# Recommended defaults (TTS + smart turn + voice interrupt all on)
uv run voice_loop_mac.py

# + chime on utterance + soft ticks while generating
uv run voice_loop_mac.py --chime

# + persistent memory (reads/writes MEMORY.md)
uv run voice_loop_mac.py --memory

# Text-only mode (no TTS)
uv run voice_loop_mac.py --no-tts

# Disable voice interruption (keypress only)
uv run voice_loop_mac.py --no-aec

# Different voice (see below)
uv run voice_loop_mac.py --voice bf_emma

# Use the smaller E2B model (faster, slightly lower quality)
uv run voice_loop_mac.py --model mlx-community/gemma-4-E2B-it-4bit

# Custom silence timeout
uv run voice_loop_mac.py --silence-ms 500

# Debug: record mic stream to a WAV
uv run voice_loop_mac.py --record
```

### MCP server mode (for coding agents)

Run Voice Loop as an MCP server — coding agents call `voice_listen` and `voice_speak` tools:

```bash
# Start MCP server directly
uv run mcp_server.py

# Or via the main entry point
uv run voice_loop_mac.py --backend=mcp-server
```

The MCP server exposes three tools over stdio:

| Tool | Description |
|------|-------------|
| `voice_listen` | Wait for speech, return transcription. Optional `timeout_secs`. |
| `voice_speak` | Speak text via Kokoro TTS. Max 2000 chars, sanitised. |
| `voice_listen_and_reply` | Listen then speak — a single voice turn. |

### Connecting to coding agents

**VS Code (Copilot)** — add to `.vscode/mcp.json`:
```json
{
  "servers": {
    "voice-loop": {
      "command": "uv",
      "args": ["run", "mcp_server.py"],
      "cwd": "/path/to/voice-loop"
    }
  }
}
```

**Claude Code** — add to `~/.config/claude/mcp_servers.json`:
```json
{
  "voice-loop": {
    "command": "uv",
    "args": ["run", "mcp_server.py"],
    "cwd": "/path/to/voice-loop"
  }
}
```

**Codex CLI** — add to `~/.codex/config.json`:
```json
{
  "mcpServers": {
    "voice-loop": {
      "command": "uv",
      "args": ["run", "mcp_server.py"],
      "cwd": "/path/to/voice-loop"
    }
  }
}
```

### Auto-start at login (LaunchAgent)

```bash
cp com.voiceloop.mcp.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.voiceloop.mcp.plist
```

## macOS App (.dmg)

The GitHub Actions workflow builds a macOS `.app` bundle via Briefcase and packages it as a `.dmg`:

1. Push a tag (`v0.1.0`) to trigger the build
2. Download the `.dmg` from the GitHub Release
3. Drag Voice Loop to Applications
4. On first launch, models download to `~/Library/Application Support/VoiceLoop/`

The app runs in an **App Sandbox** with only these entitlements:
- Microphone access
- Outbound network (model downloads + localhost MCP)
- User-selected file access (SOUL.md, MEMORY.md)

No subprocess execution of agent-provided content. No arbitrary code execution.

## Safety

Voice Loop is designed to be safe to run outside of containers:

- **No code execution** — the app only processes audio and text
- **Input sanitisation** — `voice_speak` strips control characters and enforces a 2000 char limit
- **App Sandbox** — macOS entitlements restrict file and network access
- **No shell access** — agent-provided text is never passed to subprocesses
- **Text-only relay** — the MCP server relays transcriptions and speech, nothing else

## Recommended Kokoro voices

Only the higher-quality voices are listed here:

| Voice | Accent | Gender | Notes |
|-------|--------|--------|-------|
| `af_heart` | US | Female | **Top pick** — Grade A (default) |
| `af_bella` | US | Female | Grade A-, HH training |
| `bf_emma` | UK | Female | Grade B-, HH training |
| `am_fenrir` | US | Male | Grade C+, H training |
| `am_puck` | US | Male | Grade C+, H training |
| `am_michael` | US | Male | Grade C+, H training |
| `bm_fable` | UK | Male | Grade C, MM training |
| `bm_george` | UK | Male | Grade C, MM training |

## Architecture

```
   Mic (16kHz) ──► Silero VAD ──► Smart Turn ──► Moonshine ──► Gemma 4 E4B ──► Kokoro ──► Speakers
                                                                    ▲                         │
                                                        SOUL.md + MEMORY.md                   │
                                                                                              ▼
   Mic during TTS ──► WebRTC AEC3 (LiveKit APM) ──► Silero VAD ──► voice interrupt ◄──────────┘
```

In MCP server mode, the LLM (Gemma) is replaced by the coding agent — it calls `voice_listen` / `voice_speak` tools.

## How it works

1. **Mic capture** via sounddevice (16kHz mono)
2. **Silero VAD** detects speech vs silence
3. **Smart Turn** confirms end-of-turn on silence (default on)
4. **Moonshine** transcribes your audio to text (CPU)
5. **Gemma 4 E4B** responds using SOUL.md (+ MEMORY.md if `--memory`) as system prompt — or in MCP mode, the coding agent provides the response
6. **Kokoro** synthesizes speech, streams audio
7. **WebRTC AEC3** cleans mic during TTS playback → Silero VAD on cleaned audio → voice interrupt

Press any key during TTS to interrupt.

## Project structure

```
voice-loop/
├── voice_loop_mac.py       # Main entry point (standalone + --backend=mcp-server)
├── voice_pipeline.py       # VoicePipeline class (mic, VAD, STT, TTS, AEC)
├── mcp_server.py           # MCP server (FastMCP, stdio transport)
├── backends/
│   ├── local.py            # On-device Gemma backend (MLX)
│   └── passthrough.py      # No-op backend for MCP mode
├── entitlements.plist      # macOS App Sandbox entitlements
├── com.voiceloop.mcp.plist # LaunchAgent for auto-start
├── SOUL.md                 # Persona / style
├── MEMORY.md.example       # Memory template
├── pyproject.toml          # Dependencies + Briefcase config
└── .github/workflows/
    └── build.yml           # macOS build + DMG CI
```

## Persona & Memory

- `SOUL.md` — persona / style (always loaded, live-reloaded each turn)
- `MEMORY.md` — long-term facts. Only read/written when `--memory` is passed. When enabled, the agent extracts new durable facts after each turn and consolidates every 5 turns.

Both files are re-read at the start of every turn, so edits take effect immediately.

## Memory usage

~3.5 GB total. Fits easily in 16GB.

## Credits

Built with:
- [Moonshine](https://github.com/moonshine-ai/moonshine) — STT
- [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) — TTS
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) — end-of-turn detection
- [LiveKit APM](https://github.com/livekit/python-sdks) — WebRTC AEC3
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — MLX multimodal inference
- [Gemma 4](https://huggingface.co/google/gemma-4-E4B-it) — LLM
- [FastMCP](https://github.com/PrefectHQ/fastmcp) — MCP server framework

## License

Apache 2.0.
