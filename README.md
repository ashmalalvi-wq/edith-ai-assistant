# E.D.I.T.H.

**E**ven **D**ead, **I**'m **T**he **H**ero — a local-first, personal AI assistant inspired by Tony Stark's EDITH system from *Spider-Man: Far From Home*.

> **Status: Currently Improving.** This project is under active, ongoing development. What's published here is a curated view of the core architecture, not the full application — see "What's in this repository" below.

> **Disclaimer:** E.D.I.T.H. is a personal, non-commercial project. The name is a fan-made reference to the fictional "E.D.I.T.H." system from Marvel's *Spider-Man: Far From Home*. This project has no official affiliation with, endorsement from, or connection to Marvel, Disney, or any related rights holders.

---

## What it is

E.D.I.T.H. is a desktop AI assistant that runs entirely on your own machine: your own local LLM (via [Ollama](https://ollama.com)), your own database, your own files. It's built around five ideas:

1. **It remembers.** Conversations, projects, and facts persist across restarts and are recalled by meaning (semantic search), not just keyword.
2. **It can act.** A permissioned tool system lets it touch the filesystem, run terminal commands, open your editor, and drive external integrations — never without approval where it matters.
3. **It can delegate real engineering work.** An "Expert Mode" hands off complex coding/analysis tasks to [Claude Code](https://claude.com/claude-code) running as a subprocess, with the same approval gates as everything else.
4. **It's context-aware.** "Profiles" group projects, workspaces, and preferences, so asking it to "resume my engineering work" reopens the right apps and pulls in the right memory.
5. **It automates.** A rule engine triggers capabilities on a schedule, a file event, or another capability completing — always through the same permission system a human would go through.

It ships as both a web app (FastAPI + React) and a packaged Windows desktop app (Electron).

## What's in this repository

This is a **curated subset** of the full project, published to show the core architecture and design patterns, not a drop-in-runnable full application. Included:

```
backend/
├── ai/            AI provider abstraction, prompt building, the capability-calling
│                  conversation loop, and the Expert Reasoning Agent (Claude Code delegation)
├── memory/        Structured memory (Postgres), semantic memory (Qdrant), profiles,
│                  session state, notifications, materials/research stores
├── capabilities/  Core capability framework only — the tool interface, permission
│                  system, sandboxing, and execution/logging engine
├── models/        Pydantic/SQLAlchemy schemas shared across the systems above
├── config/        Centralized settings
└── main.py
```

**Intentionally not included here** (they live in a private, actively-evolving copy of this codebase): the ~75 built-in tool implementations (filesystem, terminal, Git, tasks, etc.), external service integrations (Gmail/Calendar, Bambu, Fusion 360, weather), the automation engine, the MCP client, the FastAPI route layer, the test suite, the React frontend, and the Electron desktop shell. Because of that, this repository won't boot into a running app as-is — it's meant to be read, not deployed.

## Architecture highlights

### AI (`backend/ai/`)

- `AIProvider` interface with one fully-implemented backend (**Ollama**, local) and two structured placeholders (**OpenAI**, **Anthropic**) that raise a clear "not implemented" error rather than pretending to work.
- `ConversationOrchestrator` runs the tool-calling loop: the model can request a capability, get the result fed back, and continue — capped at 5 iterations, and any capability requiring approval pauses the turn instead of ever auto-confirming.
- A deterministic keyword pre-filter (`capability_keywords.py`) only attaches relevant tool schemas per message, instead of always sending every registered tool to the model — this measurably improved response quality and latency on a small local model.
- **Expert Reasoning Agent** (`expert_reasoning/`): shells out to the `claude` CLI for tasks that need a stronger model or real filesystem edits, gated by the same permission system as everything else.

### Memory (`backend/memory/`)

- **Structured memory** (PostgreSQL) — projects, conversations, messages, tasks, notifications, research/material stores.
- **Semantic memory** (Qdrant) — one vector per *extracted memory* (not raw chat logs), so a differently-worded question still surfaces the right fact.
- **Profiles** — first-class contexts (Engineering, School, Personal, custom) that carry their own workspaces and preferences.
- A deterministic, rule-based classifier (`classifier.py`) decides what's worth remembering — no extra model call, no hallucinated "memories."

### Capabilities (`backend/capabilities/`, core only)

The framework every action plugs into: a `Tool` interface, a permission model (`always_allow` / `ask_every_time` / `never_allow`), a sandboxed host-execution boundary, and a manager that discovers, permissions, executes, and logs every call. The actual tool implementations and integrations that plug into this framework are not published here.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, SQLAlchemy (async), Pydantic |
| Structured storage | PostgreSQL |
| Vector storage | Qdrant |
| Local LLM | Ollama (`llama3.1:8b` chat, `nomic-embed-text` embeddings) |
| Frontend (private) | React (Vite) |
| Desktop shell (private) | Electron |
| Automation (private) | APScheduler, `watchdog` |
| External integrations (private) | Google APIs, Open-Meteo, `paho-mqtt`, MCP |
| Expert reasoning | Claude Code CLI (subprocess) |

## Status & honest limitations

Being upfront about maturity, since this is a real personal project rather than a polished product:

**Fully real and live-tested (in the private full codebase):** core chat, memory, the tool-calling loop, filesystem/terminal/VS Code/Git tools, weather, tasks/notifications/briefing, the automation engine, Expert Mode (including session continuity and image attachments), and the Electron desktop shell packaged into a real Windows installer.

**Coded correctly by inspection, never exercised against real external hardware/services:** Gmail/Google Calendar (no OAuth app has ever been registered against this codebase), Bambu Lab printer control (no physical printer has ever been on the network to confirm the protocol handling), and Fusion 360 (an MCP client with no server to connect to yet — no official Fusion 360 MCP server exists).

**Explicitly out of scope so far:** voice, speech recognition, live camera/screen vision (static one-shot image attachments are supported; general vision is not), real auto-update wiring for the desktop app, and a true drag-and-drop workflow builder.

**Known model-behavior limitation:** on the local 8B model, a *second* question about the same topic in one conversation is less reliable than the first — the model tends to anchor on its own prior reply rather than fresh injected context. This is an open problem, not something a prompt tweak fixed.

## License

No license has been chosen yet — all rights reserved by default. If you're interested in the project, reach out.
