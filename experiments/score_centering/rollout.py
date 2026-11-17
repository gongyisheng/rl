"""Single-turn math generation through Miles' TITO session server."""

from miles.utils.http_utils import post


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Let the session server retain generated token IDs and sampling probabilities."""
    await post(
        f"{base_url}/v1/chat/completions",
        {**request_kwargs, "model": "default", "messages": prompt, "stream": False},
    )
