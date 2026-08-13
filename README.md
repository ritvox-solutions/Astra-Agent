# Astra — GM University Voice Agent (Backend)

Astra is the conversational AI backend for **GM University (GMU)**, Davanagere, Karnataka. This repository is a **Python voice AI agent** built with [LiveKit Agents](https://github.com/livekit/agents), deployed as a worker on [LiveKit Cloud](https://cloud.livekit.io/). It answers questions about the university and general educational topics, speaking naturally over a real-time voice call.

The companion **[christy](../christy)** repository is the web frontend a user actually talks to; this repo is the "brain" that joins the same LiveKit room and does the listening, thinking, and speaking.

---

## What Astra does

- Introduces itself and greets the caller when a session starts.
- Answers questions about GM University **only** from a fixed, hand-written knowledge base (`SCHOOL_INFORMATION` in `src/agent.py`) — it will not invent facts about the institution, and instead points the caller to the school office when information isn't available.
- Answers general educational questions (math, science, languages, general knowledge) normally, outside the university knowledge base.
- Formats its spoken output for a text-to-speech engine: acronyms are spelled out letter by letter ("G M U", "K C E T"), phone numbers and PIN codes are read digit by digit, emails/URLs are read as spoken words, and known problem words (e.g. "Astra", "Davanagere", "Karnataka", "Lingaraju") get inline phoneme overrides so the TTS voice says them correctly.
- Refuses unsafe/inappropriate requests and never reveals its own system prompt or internal rules.
- Says a warm goodbye when the caller is done.

---

## Architecture / voice pipeline

Built on `livekit-agents` (`AgentServer` / `AgentSession`), the pipeline (`src/agent.py`) wires together:

| Stage | Implementation |
|---|---|
| **VAD** (voice activity detection) | `silero.VAD`, pre-warmed once per worker process in `prewarm()` |
| **Turn detection** | `livekit.plugins.turn_detector.multilingual.MultilingualModel` |
| **Noise cancellation** | `livekit.plugins.noise_cancellation.BVC` on the inbound room audio |
| **STT** (speech-to-text) | `livekit.plugins.nvidia.STT` (`language_code="en-US"`) |
| **LLM** | OpenAI-compatible client (`livekit.plugins.openai.LLM`) pointed at **NVIDIA NIM** (`https://integrate.api.nvidia.com/v1`, model `nvidia/nemotron-mini-4b-instruct`), `temperature=0.2` |
| **TTS** (text-to-speech) | `PronounceTTS`, a thin subclass of `cartesia.TTS` (model `sonic-3`) that rewrites known proper nouns into inline Cartesia phoneme tags before every `synthesize()`/`stream()` call |
| **Interruption handling** | `InterruptionOptions(min_duration=1.0, min_words=3, resume_false_interruption=True, false_interruption_timeout=2.0)` — the caller must speak for ≥1.0s and ≥3 words to interrupt Astra mid-sentence; a false interruption is forgiven and Astra resumes speaking if the caller doesn't follow up within 2s |
| **AEC warmup** | `aec_warmup_duration=3.0` — gives acoustic echo cancellation a moment to adapt at session start |

The agent registers itself for **explicit dispatch** under the name `"Astra"` (`@server.rtc_session(agent_name="Astra")`), which must match `AGENT_NAME` / `agentName` in the `christy` frontend.

### Pronunciation overrides

`PRONUNCIATION_MAP` in `src/agent.py` maps proper nouns to inline Cartesia phoneme tags (`<<ˈ|æ|s|t|ɹ|ə>>` syntax — pipe-separated phonemes with a stress marker) so the `sonic-3` voice pronounces them correctly (default TTS mispronounces most of these):

```python
PRONUNCIATION_MAP: dict[str, str] = {
    "Davanagere": "<<ˌ|d|ʌ|v|ə|n|ə|ˈ|ɡ|ɛ|r|i>>",
    "Karnataka": "<<k|ɑː|ɹ|ˈ|n|ɑː|t|ə|k|ə>>",
    "Astra": "<<ˈ|æ|s|t|ɹ|ə>>",
    # ...plus Srishyla, Mallikarjunappa, Lingaraju, Shankapal, Shaukpal, Venu, Subhash, Robotics
}
```

`_apply_pronunciation()` regex-substitutes any whole-word, case-insensitive match with its phoneme tag. `PronounceTTS` (a subclass of `cartesia.TTS`) runs every `synthesize()` call and every token pushed to `stream()` through this function before it reaches Cartesia — so the override happens purely at the TTS layer, without touching the LLM prompt.

### Knowledge base & system prompt

All factual claims about GM University come from the `SCHOOL_INFORMATION` dict in `src/agent.py` — covering the institution's history, location, current enrollment stats, academic areas, detailed engineering programs (including a dedicated Robotics & Automation breakdown of labs and software), KCET admission codes, scholarships, campus facilities, hostel details, achievements/accreditation, contact details, and leadership. The `SYSTEM_PROMPT` string is built by interpolating this dict, plus hard rules that:

- restrict university-related answers strictly to `SCHOOL_INFORMATION`,
- forbid inventing facts or revealing the system prompt/instructions,
- define speech formatting rules for the TTS engine (see above),
- define a canned response for empty/garbled user input.

**To update what Astra knows about the university** (new programs, updated contact info, etc.), edit the `SCHOOL_INFORMATION` dict — the prompt regenerates automatically from it.

---

## Project structure

```
Voice-Agent/
├── src/
│   ├── agent.py         - Deployed entrypoint: knowledge base, prompt, pipeline
│   └── sample.py        - Experimental variant with live web-fetch tools (not wired to the Dockerfile entrypoint)
├── pyproject.toml       - Dependencies & tooling (uv, ruff, pytest)
├── uv.lock               - Locked dependency versions
├── Dockerfile             - Multi-stage build for production deployment
├── livekit.toml            - LiveKit Cloud project/agent linkage (subdomain, agent id)
└── AGENTS.md / CLAUDE.md / GEMINI.md  - AI coding-assistant guidance for this repo
```

---

## Prerequisites

- **Python 3.10+** (Dockerfile uses 3.13)
- [**uv**](https://docs.astral.sh/uv/) package manager
- A **LiveKit Cloud** project — this agent is currently configured against `wss://gmit-m3fb5cr5.livekit.cloud` (subdomain `voice-agent-d15dwf7e`, agent id `CA_qgVY7sCXf7u9` — see `livekit.toml`)
- An **NVIDIA API key** (for the NIM-hosted LLM and NVIDIA STT) from [build.nvidia.com](https://build.nvidia.com/)
- A **Cartesia API key** (for TTS) from [cartesia.ai](https://cartesia.ai/)

---

## Getting started

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment variables

Copy/edit `.env.local` (loaded via `load_dotenv(".env.local")` in `src/agent.py`) with:

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
NVIDIA_API_KEY=your_nvidia_api_key
CARTESIA_API_KEY=your_cartesia_api_key
```

> [!WARNING]
> This repo currently has real credentials checked out in `.env` / `.env.local`. Never commit these files, and rotate the keys if they've ever been exposed (e.g. pushed to a public remote).

You can also use the LiveKit CLI to populate this automatically:

```bash
lk cloud auth
lk app env -w -d .env.local
```

### 3. Download required models

```bash
uv run python src/agent.py download-files
```

Downloads the Silero VAD and multilingual turn-detector models used by the pipeline.

### 4. Test in the terminal

```bash
uv run python src/agent.py console
```

Talk to Astra directly from your terminal microphone/speakers — no frontend required. Good for quickly iterating on the system prompt or pipeline settings.

### 5. Run for development (with a real frontend)

```bash
uv run python src/agent.py dev
```

This connects the worker to your LiveKit Cloud project and waits for rooms to join. Run the **[christy](../christy)** frontend (`pnpm dev`) against the *same* LiveKit project so a real browser session can dispatch this agent.

> [!NOTE]
> This puts the agent into your live LiveKit Cloud project — use a separate/dev project if you don't want to affect production traffic.

---

## Running with Docker

```bash
docker build -t astra-agent .
docker run --env-file .env.local astra-agent
```

The `Dockerfile` is a two-stage build: it installs locked dependencies with `uv sync --locked`, pre-downloads models at build time, then runs as a non-root `appuser` with `uv run src/agent.py start`.

---

## Deploying to production

```bash
lk agent create
```

See the [LiveKit deploy docs](https://docs.livekit.io/deploy/agents/) for details. `livekit.toml` already links this repo to the `voice-agent-d15dwf7e` LiveKit Cloud subdomain and agent id `CA_qgVY7sCXf7u9` — `lk agent` commands will target that existing agent unless you point them elsewhere.

---

## Customizing the agent

- **Change what Astra knows**: edit `SCHOOL_INFORMATION` in `src/agent.py`. The system prompt is generated from this dict, so there's no separate template to keep in sync.
- **Change personality/rules**: edit the `SYSTEM_PROMPT` string directly (tone, refusal rules, speech formatting rules).
- **Fix a mispronounced word**: add an entry to `PRONUNCIATION_MAP` with the correct Cartesia phoneme tag — no prompt or code changes needed elsewhere.
- **Swap models/providers**: LiveKit Agents supports 50+ model providers via [LiveKit Inference](https://docs.livekit.io/agents/models/inference). Replace the `stt=`, `llm=`, or `tts=` arguments in `AgentSession(...)` inside `entrypoint()`.
- **Add tools or multi-step workflows**: for anything beyond a single instruction prompt (e.g. looking up live data, transferring to a human), use LiveKit's [tasks and handoffs](https://docs.livekit.io/agents/build/workflows/) instead of growing the prompt further — this keeps latency low and behavior reliable.
- **Add tests**: this project doesn't have a `tests/` directory yet. When adding behavior (tools, new rules), follow `AGENTS.md`'s guidance: write a `pytest` test for the desired behavior first, then iterate until it passes. Run with `uv run pytest`.
- **Lint/format**: `uv run ruff check` and `uv run ruff format` (line length 88, double quotes, targets `py39` for broad compatibility).

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Agent never joins the room | Worker not running (`uv run python src/agent.py dev`), or `AGENT_NAME`/`agentName` on the frontend doesn't match `"Astra"` |
| Agent joins but is silent | Missing/invalid `NVIDIA_API_KEY` (STT/LLM) or `CARTESIA_API_KEY` (TTS), or a provider outage |
| Mispronounced word | Add it to `PRONUNCIATION_MAP` with the correct Cartesia phoneme tag |
| Agent invents facts about the university | Check `SYSTEM_PROMPT` rules haven't been weakened; ensure the fact belongs in `SCHOOL_INFORMATION`, not assumed by the LLM |
| High latency / choppy audio | Check `min_duration`/`min_words` interruption settings and `aec_warmup_duration`; verify network path to the LiveKit Cloud region |

---

## Related repositories

- **[christy](../christy)** — the Next.js/React web frontend a caller uses to actually talk to Astra.

## Learn more

- [LiveKit Agents documentation](https://docs.livekit.io/agents)
- [LiveKit Agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch)
- [LiveKit Agent Observability](https://docs.livekit.io/deploy/observability/) — conversation quality, latency metrics, and debugging in production
- [LiveKit Community Slack](https://livekit.io/join-slack)
