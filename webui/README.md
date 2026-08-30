# Lightweight browser chat

This profile runs Open WebUI v0.11.1-slim as a CPU-only, single-user-oriented
frontend for the local vLLM endpoint. The image is pinned to the Linux AMD64
manifest digest. The UI and model API remain bound to host loopback; use an SSH
tunnel rather than exposing either port to the LAN or Internet.

## Start

Start one vLLM profile in the first tmux window:

```bash
just serve-webui         # 64K text chat; thinking selectable per request
# or, separately:
just serve-webui-vision  # experimental 32K multimodal profile
```

Start the UI in another window:

```bash
just webui-up
just webui-logs
```

Forward the UI from the computer running the browser:

```bash
ssh -N -L 3030:127.0.0.1:3030 emmy@<server-address>
```

Open `http://127.0.0.1:3030`, create the first local administrator account,
then disable new registrations under Admin settings. Chat data persists in the
Docker volume `qwen38_open_webui_data`. Removing the container does not remove
the volume; deleting that volume deletes the saved chats and settings.

## Model presets

In Workspace > Models, create presets over the `qwen38-w8a8` base model. Do not
attach tools, knowledge, memory, or web search to these general-chat presets.

| Preset | Capability | Custom parameter |
| --- | --- | --- |
| `Qwen Fast Chat` | Vision off; thinking off | `chat_template_kwargs` = `{"enable_thinking":false}` |
| `Qwen Think` | Vision off; thinking on | `chat_template_kwargs` = `{"enable_thinking":true}` |
| `Qwen Vision` | Mark Vision capability on; thinking off by default | `chat_template_kwargs` = `{"enable_thinking":false}` |

Set temperature, `top_p`, seed, and maximum output tokens in each preset or in
the per-chat controls. The vLLM server owns the real input-context ceiling:
Open WebUI can retain or trim less history, but it cannot raise the server
window. Qwen exposes a Boolean thinking switch, not low/medium/high reasoning
effort. A smaller output-token limit is a useful latency/budget control but is
not a distinct reasoning level.

The text profile adds vLLM's `qwen3` reasoning parser while leaving thinking
off by default. The vision profile removes `--language-model-only`, enables the
preserved vision tower, and initially limits the engineering window to 32K with
1.5 GiB of BF16 KV cache per GPU. It is intentionally not covered by the 64K
text throughput claim: validate startup memory, image paste, deterministic text,
context capacity, and throughput before promoting it.

## Operations

```bash
just webui-status
just webui-logs
just webui-down       # preserves chats/settings
```

The Compose deployment has no GPU access and caps the UI at two CPU cores and
2 GiB RAM. The pinned image occupies about 1.54 GB locally and used roughly
850 MiB RAM after the measured first startup. The slim image downloads and
caches `sentence-transformers/all-MiniLM-L6-v2` during its first initialization
even though knowledge/RAG features are disabled; subsequent starts reuse the
persistent-volume cache. Ollama, web search, memory, code execution, the code interpreter,
and the OpenAI catch-all passthrough are disabled. Uploaded images and chat
content are still stored in the persistent local volume; do not paste secrets
or protected records without reviewing that storage boundary.
