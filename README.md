# Personal AI assistant

The assistant accepts tasks from one Telegram account, calls Llama 3.1 8B through Ollama, and processes an open-source backlog every two hours when idle. The Python application, outbound gateway, and snippet runner run in Docker.

## Run locally

1. Start Ollama on the host and install the model with `ollama pull llama3.1`. Check that `http://localhost:11434/api/tags` responds. The Docker gateway must be able to reach the host Ollama service on port `11434` through `host.docker.internal`. [Ollama installation](https://ollama.com/download)
2. In PowerShell, run `cd assistant` and `Copy-Item .env.example .env`. Fill in `TELEGRAM_TOKEN` and your numeric private `TELEGRAM_CHAT_ID`. Open the bot's private chat and send `/start`. The bot uses polling, so use a bot with no webhook configured.
3. Run `docker compose up --build -d` and `docker compose logs -f assistant`. Send the bot a task, such as `Create /workspace/hello.txt containing Hello`. If a task needs to overwrite an existing file, the bot sends a one-time `/approve` command that expires after 15 minutes.

Ollama has 120 seconds to answer each model request, including the first load of Llama 3.1. Set `OLLAMA_TIMEOUT_SECONDS` in `assistant/.env` to change that budget, then run `docker compose up -d` from `assistant` to apply the setting to both services. The other tools use their own 30-second limits.

For definitions, web search looks up the term in Wiktionary. For general searches, it tries DuckDuckGo first and uses Mwmbl if DuckDuckGo fails or has no usable results. Set `WEB_SEARCH_FALLBACK=0` in `.env` for DuckDuckGo-only searches. The previous `BING_SEARCH_FALLBACK=0` setting still disables alternate sources unless `WEB_SEARCH_FALLBACK` is set.

Docker Desktop uses Linux containers. On native Linux, the bind-mounted `assistant/workspace` must be writable by UID `10001`. The internal networks use Docker's isolated bridge gateway mode. [Docker bridge gateway modes](https://docs.docker.com/engine/network/port-publishing/#gateway-modes)

## Enable GitHub and backlog work

1. In `.env`, set `GITHUB_REPO` to `owner/name` for a repository or fork you control. Set `GITHUB_TOKEN` to a token with Contents write access to that repository.
2. Clone that repository over HTTPS into `assistant/workspace/project` and check out `main`. Use the same GitHub repository for `origin` and `GITHUB_REPO`.
3. Copy `autonomous/backlog.md` to `workspace/project/backlog.md`. Add tasks as `- [ ] Task description`. The first check runs two hours after startup. Scheduled work starts with a clean checkout, apart from edits to the backlog. A task that needs to overwrite a file pauses for owner approval and stays unchecked.

The assistant marks a successful entry `[x]`, writes a daily Markdown report under `workspace/reports`, and publishes a copy under `reports/` in the project checkout. It commits the task's changed files, the backlog, and the report together and pushes to `origin main`. A failed push is retried before any new task. Failed task attempts are recorded in the local daily report. Report dates default to `Asia/Tomsk` through the supplied `.env.example`.

The assistant has one writable host mount at `/workspace`. The gateway permits GitHub, Telegram, DuckDuckGo, Mwmbl, Wiktionary, and the local Ollama chat endpoint. Python snippets run in a separate read-only container with temporary space under its own `/workspace`. The runner receives no host workspace or service credentials. Stop the stack with `docker compose down`.
