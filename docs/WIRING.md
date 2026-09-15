# Wiring and running

## 1. Install

Python 3.12. From the repository root:

```bash
pip install -e .            # the CLI and the MCP server
pip install -e ".[dev]"     # plus pytest, ruff, mypy
```

## 2. The model server

Any OpenAI-compatible endpoint. The default and documented one is
[`llama-server`](https://github.com/ggml-org/llama.cpp), which serves one model per process,
so run a chat model and an embedding model:

```bash
llama-server -m qwen2.5-7b-instruct.gguf --port 8081                 # chat
llama-server -m mxbai-embed-large.gguf --embedding --port 8082       # embeddings
```

Run them on the machine with the GPU and reach them over your overlay network, or on
`localhost` for offline work; only the endpoint URL differs. Ollama's `/v1` and vLLM speak the
same protocol.

## 3. Configure

Copy `.env.example` to **`~/.config/morgan/.env`** (`$XDG_CONFIG_HOME/morgan/.env`). That
file is read from every working directory, which is what the CLI needs: it is meant to run
from inside whichever repository you are working in. A `./.env` in the current directory is
read after it and overrides it; real environment variables override both.

```bash
MORGAN_LLM_ENDPOINT=http://localhost:8081/v1   # the chat model
MORGAN_LLM_MODEL=qwen2.5-7b-instruct
MORGAN_EMBEDDING_MODEL=mxbai-embed-large      # the embedding model
MORGAN_EMBEDDING_DIM=1024                     # must match the embedding model
# MORGAN_EMBEDDING_ENDPOINT=http://localhost:8082/v1   # only if it is a separate server
# MORGAN_LLM_API_KEY=                         # only if the server enforces --api-key
# MORGAN_LLM_JSON_MODE=json_schema            # how consolidation asks for JSON
# MORGAN_DATA_DIR=~/.local/share/morgan       # the one database
# MORGAN_API_KEY=                             # required before morgan-mcp binds beyond loopback
```

Two keys point in opposite directions: `MORGAN_LLM_API_KEY` is what Morgan presents *to* the
model server; `MORGAN_API_KEY` is what MCP clients present *to* Morgan over HTTP.

When one server answers both, leave `MORGAN_EMBEDDING_ENDPOINT` empty and everything goes to
`MORGAN_LLM_ENDPOINT`. Set it when they are separate, which is the common case:
`llama-server` loads one model per process, and a chat server started without `--embedding`
answers `/embeddings` with a 501.

## 4. Check: `morgan doctor`

```
database: /home/you/.local/share/morgan/morgan.db
config_file: /home/you/.config/morgan/.env
config_file_present: True
project: my-repo
embedding_backend: provider
llm_endpoint: http://gpu-box:8081/v1
sqlite_vec: v0.1.9
fts5: True
provider: reachable
memory_rows: 0
```

Every probe is independent, so one failure does not hide the rest. The first two lines answer
"why is my brain empty?": a database or config file somewhere other than where you expect.

## 5. The CLI

Every command takes `--project` (default: the current git repository's directory name),
`--json`, and where it makes sense `--all-projects`.

```bash
morgan remember "the Harbor mirror blocked the deploy"   # embedding model only
morgan recall "what blocked the deploy"                   # vector + FTS5, fused
morgan facts                                              # currently-valid facts
morgan ask "what do you know about the deploy"            # chat model: recall, answer, remember
morgan consolidate                                        # chat model: memories → facts
morgan forget                                             # everything under this project
morgan import ~/Downloads/conversations.json              # seed from a ChatGPT export
```

`import` seeds memory from a ChatGPT export so a fresh brain is not an empty box. It writes
to `archive/chatgpt`, not to your working project, and takes no `--project`: a fifth of the
conversations go to `archive/chatgpt-holdout` instead, reserved for evaluating the memory
against conversations nothing has learned from. Expect it to take a while, since every turn
costs an embedding call; progress goes to stderr. Re-running updates in place.

`MORGAN_EMBEDDING_BACKEND=hash` replaces the embedding call with a deterministic stub, so the
memory commands run with no model server at all (keyword and entity search still work; vector
similarity does not mean anything).

When the model server is down, every command that needs it says which endpoint it could not
reach and exits 1; under `--json` the error is the whole of stdout.

## 6. The MCP server

The same five operations for any MCP client, through the same gate. `project` is a tool
argument (the server is a daemon; its own working directory means nothing to a client).

```bash
claude mcp add morgan -- morgan-mcp --transport stdio      # a client on this machine
morgan-mcp --transport http                                 # loopback, MORGAN_MCP_HOST/PORT
MORGAN_API_KEY=… morgan-mcp --transport http --host 100.64.0.7   # other machines, over the overlay
```

The HTTP transport enforces `MORGAN_API_KEY` as a bearer token. With no key set it serves
loopback only, and refuses to start on any other host: these tools include `forget`. See
[`OPERATIONS.md`](OPERATIONS.md) for client configuration.

## 7. Docker

`docker compose up -d` builds the image and runs `morgan-mcp --transport http` on port 8090
with `./data` mounted as the database directory. Put `MORGAN_LLM_ENDPOINT` and a real
`MORGAN_API_KEY` in `.env` next to the compose file; the published port is not loopback, so
the server refuses to start without the key.

## 8. Consolidation on a schedule

Nothing in Morgan runs a model unasked. If you want nightly consolidation, that is one cron
line on the machine that holds the database:

```
0 3 * * * morgan consolidate --all-projects --json >> ~/.local/share/morgan/consolidate.log
```
