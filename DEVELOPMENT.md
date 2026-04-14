# Local Development

Guide for building and testing Voice Loop locally on Apple Silicon (Mac Studio M3, MacBook M-series, etc.) without going through GitHub Actions.

## Prerequisites

```bash
# System dependencies
brew install portaudio espeak-ng

# Optional — only needed for DMG packaging
brew install create-dmg
```

Python 3.11+ is required. If you use [uv](https://docs.astral.sh/uv/) (recommended), it manages Python for you.

## Quick start (no packaging)

The fastest way to run Voice Loop during development — no Briefcase, no .app bundle:

```bash
# Install dependencies and run directly
uv sync
uv run voice_loop_mac.py

# Or with options
uv run voice_loop_mac.py --chime --memory
uv run voice_loop_mac.py --no-tts          # text output only
uv run voice_loop_mac.py --backend=mcp-server
```

This is the recommended workflow for day-to-day development in VS Code or any editor.

## VS Code setup

1. Open the `voice-loop` folder in VS Code
2. Install the **Python** extension
3. Select the uv-managed interpreter: `Cmd+Shift+P` → *Python: Select Interpreter* → choose `.venv/bin/python`
4. To run/debug, add to `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Voice Loop",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/voice_loop_mac.py",
      "args": ["--chime"],
      "console": "integratedTerminal"
    },
    {
      "name": "MCP Server",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/mcp_server.py",
      "console": "integratedTerminal"
    }
  ]
}
```

## Building the macOS .app locally

Use [Briefcase](https://briefcase.readthedocs.io/) to build a native macOS `.app` bundle:

```bash
# Install Briefcase (pin rich to avoid version conflict)
pip install "rich<15.0" briefcase

# One-time: create the app scaffold
briefcase create macOS app

# Build the app
briefcase build macOS app
```

The built app is at: `build/voiceloop/macos/app/Voice Loop.app`

### Run the built app

```bash
# Via Briefcase (opens the .app)
briefcase run macOS app

# Or directly from terminal
open "build/voiceloop/macos/app/Voice Loop.app"

# Or from the command line to see stdout/stderr
"build/voiceloop/macos/app/Voice Loop.app/Contents/MacOS/Voice Loop"
```

### Iterating on code changes

After editing source files, rebuild without re-creating the scaffold:

```bash
# Update sources and rebuild
briefcase update macOS app
briefcase build macOS app
```

Or use **dev mode** which runs your code directly from source (no .app bundle, but uses Briefcase's dependency resolution):

```bash
briefcase dev
```

### Packaging as DMG

```bash
APP_PATH="build/voiceloop/macos/app/Voice Loop.app"

# Sign (ad-hoc for local testing)
codesign --force --deep --sign - \
  --entitlements entitlements.plist \
  "$APP_PATH"

# Create DMG
create-dmg \
  --volname "Voice Loop" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --app-drop-link 400 185 \
  --no-internet-enable \
  "VoiceLoop-local.dmg" \
  "$APP_PATH"
```

## Using a local GitHub Actions runner

If you have a self-hosted runner on your Mac Studio:

1. Make sure the runner has the `macos-15` label (or update the workflow to match your label)
2. Push to a branch or manually trigger the workflow via `workflow_dispatch`
3. The runner will execute the full CI build locally

To trigger manually from the command line:

```bash
gh workflow run build.yml --ref your-branch
```

## Xcode

Voice Loop is a Python project, so Xcode isn't used for the main development workflow. However, if you want to inspect or debug the packaged `.app`:

1. Build with Briefcase first (see above)
2. Open the built `.app` in Xcode via *File → Open* to inspect the bundle structure
3. Use Xcode's **Console.app** or `log stream` to view system logs from the sandboxed app

For actual code editing, VS Code with the Python extension is the recommended IDE.

## Troubleshooting

### `rich` version conflict with Briefcase

Briefcase 0.4.x requires `rich<15.0`. If you see a compatibility error, pin rich:

```bash
pip install "rich<15.0" briefcase
```

### Models download on first run

Gemma 4 E4B (~3 GB), Moonshine (~250 MB), and Kokoro (~300 MB) download on first launch. Ensure you have network access and ~4 GB free disk space.

### Microphone permission

macOS will prompt for microphone access on first run. If running from terminal, grant permission to Terminal.app (or iTerm2). If running the `.app`, it will prompt automatically.

### `portaudio` not found

```bash
brew install portaudio
```

If Python still can't find it, you may need:

```bash
export DYLD_LIBRARY_PATH="$(brew --prefix portaudio)/lib:$DYLD_LIBRARY_PATH"
```
