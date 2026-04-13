"""Passthrough backend — returns empty responses.

Used in MCP server mode where the coding agent provides the response text
directly via the voice_speak tool rather than generating it locally.
"""


class PassthroughBackend:
    """No-op backend for MCP server mode.

    The MCP client (coding agent) drives the conversation — it calls
    voice_listen to get transcribed speech, then calls voice_speak with
    its own response text.
    """

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        return ""

    def update_memory(self, *args, **kwargs) -> None:
        pass

    def consolidate_memory(self, *args, **kwargs) -> None:
        pass
