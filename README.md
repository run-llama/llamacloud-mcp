# llamacloud-mcp (CLI-first fork)

> Run LlamaCloud indexes and extract agents as Model Context Protocol (MCP) tools directly from Claude Code or any other CLI-friendly client.

This repository is a **personal fork and refactor** of [run-llama/llamacloud-mcp](https://github.com/run-llama/llamacloud-mcp). The original project focused on Claude Desktop demos; this fork prioritizes terminal-first workflows so that MCP tools are easy to run from Claude Code, VS Code, or headless automation.

Desktop guidance is still available in the appendix, but everything else assumes a CLI environment.

## Highlights of this fork

- CLI-first documentation and examples tailored to Claude Code.
- Cached LlamaCloud clients so every tool invocation is faster and kinder to rate limits.
- Explicit host/port/path flags for SSE and streamable HTTP transports.
- Safety checks that ensure at least one tool is registered before the server starts.
- A recommended testing workflow (pytest via `uv`) to keep future changes healthy.

## Requirements

| Requirement | Why |
|-------------|-----|
| Python 3.10+ | matches LlamaIndex + uv support |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | dependency + CLI runner |
| LlamaCloud project/org + API key | authenticates index + extract agent access |
| (Optional) Claude Code or another MCP-compatible CLI | consumes the tools you expose |

Set the following environment variable (via `.env` or your shell):

```bash
export LLAMA_CLOUD_API_KEY=<your key>
```

You can still pass `--api-key` explicitly when launching the CLI.

## Installing / Running locally

Clone this fork and install dependencies once:

```bash
git clone git@github.com:ronamosa/llamacloud-mcp.git
cd llamacloud-mcp
uv venv
uv pip install .
```

### Defining tools

Each `--index` or `--extract-agent` flag accepts `name:description`. The `name` becomes part of the tool ID (`query_<name>` / `extract_<name>`), so keep it alphanumeric and short.

```bash
uv run llamacloud-mcp \
  --index product-docs:"Search curated product documentation" \
  --index eng-rfcs:"Retrieve engineering RFCs" \
  --extract-agent pricing-parser:"Extract key pricing terms from PDFs" \
  --project-id 123e4567-e89b-12d3-a456-426614174000 \
  --org-id 42a1c403-6ac1-40df-9321-5db0a4979a36 \
  --transport stdio
```

When using Claude Code, add an MCP server entry that shells out to the command you prefer (stdio is the simplest for CLI usage). Because clients call the same binary repeatedly, caching the LlamaCloud retrievers/agents inside this refactor significantly reduces cold-start latency.

### Transport options

The CLI now exposes host/port/path settings so you can intentionally choose a transport:

| Flag | Default | Notes |
|------|---------|-------|
| `--transport` | `stdio` | `stdio`, `sse`, or `streamable-http` |
| `--host` | `127.0.0.1` | Binding for SSE + streamable HTTP |
| `--port` | `8000` | SSE port; ignored for stdio |
| `--mount-path` | `/` | Base path for HTTP modes |
| `--sse-path` | `/sse` | SSE endpoint |
| `--message-path` | `/messages/` | SSE message polling path |
| `--streamable-http-path` | `/mcp` | Streamable HTTP endpoint |

#### SSE (Claude Code recommended)

```bash
uv run llamacloud-mcp \
  --transport sse \
  --host 127.0.0.1 \
  --port 8765 \
  --index product-docs:"Docs search over SSE"
```

Point Claude Code’s MCP config at `http://127.0.0.1:8765/sse`.

#### Streamable HTTP

Use this when your client supports the [streamable HTTP transport](https://modelcontextprotocol.io). Adjust `--streamable-http-path` if you need to reverse-proxy the endpoint.

## Claude Code configuration example

Add the following snippet to your Claude Code `claude_code_config.json` (or whichever file the client uses to discover MCP servers). Replace values as needed:

```jsonc
{
  "mcpServers": {
    "llamacloud-cli": {
      "command": "uv",
      "args": [
        "run",
        "llamacloud-mcp",
        "--transport",
        "sse",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
        "--index",
        "product-docs:Search curated product documentation",
        "--extract-agent",
        "pricing-parser:Extract pricing terms from PDFs",
        "--project-id",
        "123e4567-e89b-12d3-a456-426614174000",
        "--org-id",
        "42a1c403-6ac1-40df-9321-5db0a4979a36"
      ],
      "env": {
        "LLAMA_CLOUD_API_KEY": "<your key>"
      }
    }
  }
}
```

Restart Claude Code after editing its config so the MCP server reloads.

## Testing & CI workflow

This fork does not ship a full test suite yet, but the recommended workflow is:

```bash
uv run pytest          # add smoke tests for CLI parsing + tool wiring
uv run ruff check .    # optional linting once ruff is added to dependencies
```

When you add tests, wire them into your preferred CI (GitHub Actions, Buildkite, etc.) so every change exercises the CLI surface before merging. Running `uv run llamacloud-mcp --help` in CI is a quick sanity check that Click options stay valid.

## Troubleshooting

- **“API key not found”** – set `LLAMA_CLOUD_API_KEY` or pass `--api-key`.
- **“Please provide at least one --index or --extract-agent option.”** – the server now fails fast to prevent running without tools.
- **Claude Code cannot connect over SSE** – ensure the host/port is reachable from your editor and matches the config path (default `/sse`).
- **Rate limits / slow responses** – verify you are not recreating indexes per query. This fork caches retrievers, so restarting the server is usually sufficient.

## Appendix A – Claude Desktop (legacy instructions)

Claude Desktop still works, it’s just no longer the primary audience. Follow the CLI instructions above, then configure Desktop via `Claude → Settings → Developer → Edit Config`. Example:

```json
{
  "mcpServers": {
    "llamacloud-cli": {
      "command": "uv",
      "args": [
        "run",
        "llamacloud-mcp",
        "--transport",
        "stdio",
        "--index",
        "docs:Docs search"
      ],
      "env": {
        "LLAMA_CLOUD_API_KEY": "<your key>"
      }
    }
  }
}
```

Restart Claude Desktop after editing the config and you’ll see the MCP server listed under the tool icon. The screenshot from the upstream README (`./claude.png`) still applies if you need a visual cue.
