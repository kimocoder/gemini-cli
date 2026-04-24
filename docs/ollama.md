# Using Ollama with Gemini CLI

Gemini CLI supports [Ollama](https://ollama.com) as an alternative model
provider, allowing you to chat with locally-hosted or self-hosted LLMs with **no
Google account, no subscription, and no API key required**.

## Prerequisites

1. **Install Ollama** — Download from [ollama.com](https://ollama.com) or via:

   ```bash
   # macOS / Linux
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Start the Ollama server**:

   ```bash
   ollama serve
   ```

3. **Pull a model**:

   ```bash
   ollama pull llama3.1        # Meta Llama 3.1 8B (recommended)
   ollama pull mistral         # Mistral 7B
   ollama pull codellama       # Code-optimised Llama
   ollama pull qwen2.5-coder   # Qwen 2.5 Coder
   ```

   See the full model library at
   [ollama.com/library](https://ollama.com/library).

## Selecting Ollama as Your Provider

### Interactive Setup (Recommended)

1. Start Gemini CLI normally or run `/auth` from within a session.
2. Choose **"Ollama (local / self-hosted, free)"** from the auth dialog.
3. The CLI fetches the list of models currently installed on your Ollama server.
4. Select the model you want to use.
5. Start chatting — no login required!

### Environment Variable

Set `OLLAMA_HOST` (or `OLLAMA_BASE_URL`) to automatically use Ollama:

```bash
export OLLAMA_HOST=http://localhost:11434
gemini
```

When either variable is set, Gemini CLI automatically selects the Ollama
provider.

### CLI Flag

Pass the Ollama host directly when starting the CLI:

```bash
OLLAMA_HOST=http://localhost:11434 gemini
```

## Connecting to a Remote Ollama Server

If you run Ollama on a remote host (or in a container), point the CLI at it:

```bash
export OLLAMA_HOST=https://my-remote-ollama.example.com
gemini
```

Or set `ollamaHost` in your Gemini CLI settings file
(`~/.gemini/settings.json`):

```json
{
  "security": {
    "auth": {
      "selectedType": "ollama",
      "ollamaHost": "https://my-remote-ollama.example.com",
      "ollamaModel": "llama3.1"
    }
  }
}
```

## Listing and Switching Models

To switch models after the initial setup, run `/auth` from within the chat
session. Select **Ollama** again and you will be presented with the updated
model list from the server (so pulling new models with `ollama pull` is
immediately reflected).

## Troubleshooting

| Symptom                         | Solution                                                     |
| ------------------------------- | ------------------------------------------------------------ |
| _"Could not connect to Ollama"_ | Make sure `ollama serve` is running                          |
| No models shown                 | Pull at least one model: `ollama pull llama3.1`              |
| Slow first response             | Models are loaded on first use — subsequent turns are faster |
| Want a different host           | Set `OLLAMA_HOST` env var or `ollamaHost` in settings        |

## Notes

- **Tool calling** (function declarations) is supported for models that
  advertise tool-call capability (e.g. `llama3.1`, `mistral-nemo`,
  `qwen2.5-coder`). Models without tool support simply ignore function
  declarations.
- **Token counting** uses a heuristic (~4 chars per token) since Ollama does not
  expose a dedicated endpoint.
- **Embeddings** are available via the `/api/embed` endpoint for models that
  support them.
- Context compression and other advanced features that depend on the Gemini API
  are not available with Ollama.
