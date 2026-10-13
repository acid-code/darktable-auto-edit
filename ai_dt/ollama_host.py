"""Host and reachability for a local Ollama server.

The vision advisor and the text reasoner both use this. ``OLLAMA_HOST``
is the same variable the Ollama client reads.
"""

from __future__ import annotations

import os
import socket


def ollama_base_url() -> str:
    """HTTP origin for ``OLLAMA_HOST``, the same host ``ollama_reachable`` dials."""

    hostport = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    hostport = hostport.removeprefix("http://").removeprefix("https://").split("/")[0]
    return "http://" + hostport


def ollama_reachable(timeout: float = 0.4) -> bool:
    """Return True when a TCP connection to the Ollama host succeeds."""

    hostport = ollama_base_url().removeprefix("http://").removeprefix("https://")
    if hostport.startswith("["):
        host, _, port_text = hostport[1:].partition("]")
        port_text = port_text[1:] if port_text.startswith(":") else "11434"
    elif ":" in hostport:
        host, port_text = hostport.rsplit(":", 1)
    else:
        host, port_text = hostport, "11434"
    try:
        port = int(port_text)
    except ValueError:
        port = 11434
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False
