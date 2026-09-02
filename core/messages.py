"""Getting plain text out of a model response.

Gemini 3 returns structured content blocks rather than a string, so
`response.content` is a list like

    [{"type": "text", "text": "..."}]

and may carry reasoning blocks beside the answer. Reading `.content` directly
gives a list where a string is expected, which is why this exists in one place
rather than being handled at each of the four call sites.

Reasoning blocks are dropped deliberately. They are the model's working, not
its output; letting them through would put chain-of-thought into the user's
answer and into JSON that is about to be parsed.

Lives in core rather than ai/ because services/chart_config.py needs it too,
and ai/nodes/chart.py already imports that module - putting it in ai/ would
point a dependency back the way it came.
"""

from typing import Any

# Blocks that are the model thinking rather than the model answering. Several
# spellings because the key has changed across provider and library versions.
_REASONING_TYPES = {
    "thinking",
    "thought",
    "reasoning",
    "reasoning_content",
}


def _from_block(block: Any) -> str:
    if isinstance(block, str):
        return block

    if isinstance(block, dict):
        if str(block.get("type", "")).lower() in _REASONING_TYPES:
            return ""

        text = block.get("text")

        return text if isinstance(text, str) else ""

    # Some versions yield block objects rather than dicts.
    text = getattr(block, "text", None)

    return text if isinstance(text, str) else ""


def text_of(message: Any) -> str:
    """
    Accepts a message, a streamed chunk, or a raw content value.

    Tries the library's own text accessor first - newer LangChain exposes
    `.text` on messages and does this flattening itself - and falls back to
    walking the blocks when that is absent or is not a string.
    """
    if message is None:
        return ""

    if isinstance(message, str):
        return message

    accessor = getattr(message, "text", None)

    if isinstance(accessor, str):
        return accessor

    if callable(accessor):
        try:
            value = accessor()

            if isinstance(value, str):
                return value
        except Exception:
            pass

    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(_from_block(block) for block in content)

    return "" if content is None else str(content)
