"""Local Gemma backend — runs the LLM on-device via MLX.

This preserves the original standalone voice-loop behaviour.
"""

import os
from pathlib import Path


class LocalBackend:
    """On-device LLM backend using mlx-vlm (Gemma 4 E4B by default)."""

    def __init__(self, model_name: str = "mlx-community/gemma-4-E4B-it-4bit"):
        print(f"Loading {model_name} (first run downloads ~3GB)...", flush=True)
        from mlx_vlm import load, generate

        self._model, self._processor = load(model_name)
        self._generate = generate

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        r = self._generate(
            self._model,
            self._processor,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=1.2,
            verbose=False,
            **kwargs,
        )
        return r.text if hasattr(r, "text") else str(r)

    def update_memory(
        self,
        heard: str,
        response: str,
        mem_path: Path,
        read_memory_fn,
    ) -> None:
        """Extract new durable facts and append to MEMORY.md."""
        result = self._run_memory(
            f"Current memory:\n{read_memory_fn()}\n\n"
            f"User said: {heard}\n\n"
            "Did the user state a new durable fact about themselves? "
            "If yes, output one short fact per line starting with '- '. "
            "If no, output ONLY: NONE. Do not invent facts.",
            max_tokens=60,
            temperature=0.2,
            label="memory update",
        )
        if result and "NONE" not in result.upper():
            lines = [l for l in result.splitlines() if l.strip().startswith("-")]
            if lines:
                with open(mem_path, "a") as f:
                    f.write("\n" + "\n".join(lines) + "\n")
                print(f"  [memory +{len(lines)}]", flush=True)

    def consolidate_memory(
        self,
        mem_path: Path,
        read_memory_fn,
    ) -> None:
        """Merge duplicates, remove transient items from MEMORY.md."""
        if not mem_path.exists():
            return
        result = self._run_memory(
            f"Here is a memory file about a user:\n\n{read_memory_fn()}\n\n"
            "Rewrite it: merge duplicates, remove transient/session-specific "
            "items (questions asked, topics discussed, tests), keep only "
            "durable facts (identity, preferences, relationships, location, "
            "ongoing projects). Output the cleaned file, starting with '# Memory' "
            "followed by bullets starting with '- '. No explanation.",
            max_tokens=300,
            temperature=0.2,
            label="memory consolidation",
        )
        if result and result.startswith("# Memory"):
            mem_path.write_text(result + "\n")
            print("  [memory consolidated]", flush=True)

    def _run_memory(self, prompt, max_tokens, temperature, label):
        import sys

        try:
            return self.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            ).strip()
        except Exception as e:
            print(f"  [{label} failed: {e}]", file=sys.stderr)
            return None
