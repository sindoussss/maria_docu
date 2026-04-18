# Maria AI Assistant — Project Documentation

**Project:** Maria AI Desktop Assistant
**Language:** Python 3.10+
**UI Framework:** PyQt6
**AI Backend:** Ollama (local inference, no cloud API)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Architecture Overview](#architecture-overview)
5. [Core Modules Explanation](#core-modules-explanation)
6. [Key Functions Reference](#key-functions-reference)
7. [Class-Level Documentation](#class-level-documentation)
8. [UI / UX Flow](#ui--ux-flow)
9. [API / Data Flow](#api--data-flow)

---

## Project Overview

**Maria** is a locally-deployed AI assistant desktop application built entirely in Python. It does not rely on any cloud API — all language model inference runs through **Ollama**, a local model server, which the app communicates with over localhost.

**Core purpose:** Provide a Filipina Taglish-speaking AI assistant that can hold natural conversations, solve math symbolically, execute code safely, retrieve live web data, and send emergency alerts — all from a single PyQt6 desktop window.

**Main features:**

| Category | Feature |
| --- | --- |
| Conversation | Multi-session persistent chat with sidebar navigation |
| Language | Bilingual Filipino/English routing with 14 intent categories |
| Context | Active-task tracking across multi-turn conversations |
| Math | SymPy-backed symbolic computation (equations, calculus, plots) |
| Code | Isolated Python sandbox execution with security blocklist |
| Web | Live DuckDuckGo search with cosine-similarity RAG reranking |
| Safety | Bilingual emergency alert system with SMTP email dispatch |
| Files | PDF text extraction and chat transcript PDF export |
| UI | Real-time code syntax highlighting, LaTeX rendering, math plots |
| AI | Adaptive context compression via LLM summarization |
| Training | LoRA/SFT training data recording for offline fine-tuning |

---

## Tech Stack

### Core Language
- **Python 3.10+** — uses walrus operator (`:=`), union types (`X | Y`), `ast.unparse`

### UI Framework
- **PyQt6** — main window, all widgets, QThread workers for background LLM calls, optional QOpenGLWidget for GPU skeleton animation, QSyntaxHighlighter for code coloring, QPropertyAnimation for transitions

### AI / Machine Learning
- **ollama** — Python client that wraps Ollama's REST API. Every LLM call is throttled and GPU-injected before being sent.
- **scikit-learn** — `TfidfVectorizer`, `HashingVectorizer`, and `cosine_similarity` used in the web RAG pipeline to rank retrieved web passages by relevance.
- **langdetect** — fallback language detection for ambiguous multi-word queries when the rule-based system is not confident.
- **pyttsx3** *(optional)* — text-to-speech for reading responses aloud. Gracefully disabled if unavailable.

### Math / Computation
- **sympy** — symbolic math: `solve()`, `diff()`, `integrate()`, `factor()`, `sympify()`. Runs in a separate timed thread with a hard 3-second wall-clock timeout.
- **numpy** — array operations for plotting and similarity calculations.
- **matplotlib** (`Agg` backend only) — renders math function plots to PNG off-screen with no Qt conflict. Images are saved to a temp file and loaded into chat.

### Networking / Web
- **requests** — all HTTP calls have enforced connect+read timeouts via a custom adapter.
- **DuckDuckGo HTML** — primary search engine. The app scrapes `html.duckduckgo.com` directly with no API key required.
- **beautifulsoup4** — parses raw HTML from web search results.
- **html2text** — converts scraped HTML to clean readable text for RAG injection.
- **smtplib / email.mime** — sends emergency alert emails via Gmail SMTP TLS.

### Data Storage
- **json** — primary storage format for sessions and LoRA training data.
- **sqlite3** — secondary persistence path.
- **pickle** — caches TF-IDF vectorizer state between runs.
- **PyMuPDF (`fitz`)** — extracts text from user-uploaded PDF files.
- **reportlab** — generates styled PDF exports of chat transcripts.

### System / OS Utilities
- **threading / concurrent.futures** — background task pool, concurrency control, debounced writes.
- **subprocess** — sandboxed Python execution in an isolated child process.
- **hashlib (SHA-256)** — content-addressed deduplication for uploaded attachments.
- **pytz** — timezone-aware datetime for scheduling features.

### Runtime / Environment Requirements
- Python 3.10 or newer (tested on 3.10–3.12)
- **Ollama** running locally on `localhost:11434`
- At least one Ollama model pulled (e.g. `llama3`, `mistral`, `phi3`)
- Windows, macOS, or Linux (Qt6 and all pip dependencies available on all three)
- Optional: NVIDIA GPU with CUDA for accelerated inference (auto-detected at startup)

---

## Project Structure

```
Project_Maria/
├── main.py                     # Main application
├── maria_sessions.json         # Chat session persistence (created at runtime)
├── maria_training_data.json    # LoRA/SFT training examples (created at runtime)
├── maria_attachments/          # Content-hashed permanent attachment store
│   └── <sha256_prefix>.<ext>  # e.g. a3f84bc2.pdf, 9e1c4f12.png
├── config.py                   # External config (model names, feature flags)
├── maria_core.py               # Core inference helpers (imported by main)
├── maria_knowledge.py          # Knowledge base definitions
├── maria_training.py           # SFT/LoRA training pipeline
├── maria_utils.py              # GPU probe and utility functions
├── scripts/
│   ├── check_regression.py     # Routing regression test runner
│   ├── maria_dashboard.py      # Analytics dashboard
│   ├── run_ablation.py         # Model ablation experiments
│   ├── run_eval.py             # Evaluation harness
│   ├── train_dpo.py            # DPO training script
│   ├── train_sft.py            # SFT training script
│   ├── downloader/
│   │   ├── gutenberg.py        # Project Gutenberg corpus downloader
│   │   ├── pinoy_datasets.py   # Filipino dataset downloader
│   │   ├── process_simplewiki.py  # Simple Wikipedia preprocessor
│   │   └── wikipedia.py        # Wikipedia corpus downloader
│   └── table/
│       └── make_splits.py      # Train/val/test split generator
├── test_intent_trace.py        # Intent classifier trace-level unit tests
└── test_routing_regressions.py # End-to-end routing regression tests
```

**How files connect:**
- `main.py` is the main file for the AI. It optionally imports from `config.py` and `maria_utils.py` for GPU probing.
- `maria_sessions.json` is read on startup and written on every session change via the async debounced writer.
- `maria_training_data.json` is appended by the background training recorder when the user approves a response.
- `maria_attachments/` is created automatically. Files in it persist across restarts because they are content-hashed.
- Scripts in `scripts/` are standalone tools — they do not import from `main.py`.

---

## Architecture Overview

### System Design

Maria follows a **single-process, multi-threaded architecture**. The main thread runs the Qt event loop (UI). All LLM inference, web search, SymPy computation, and file I/O happen in worker threads or the shared background pool. Results are passed back to the UI via Qt signals.

```
User types message
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                  Pre-processing Pipeline                   │
│  1. detect_language()          → language code             │
│  2. is_trivial_social_message() → fast-path casual?       │
│  3. detect_math_query()        → math pre-processor?      │
│  4. _build_active_task_context() → ongoing task state     │
│  5. _classify_continuation()   → same task or new topic?  │
│  6. _classify_query_intent()   → 14-way intent router     │
└───────────────────────────────────────────────────────────┘
        │
        ▼ (intent label + active task context)
┌───────────────────────────────────────────────────────────┐
│                  Tool / Data Selection                     │
│  math?      → _sympy_precompute()                         │
│  live_info? → web_search() + RAG reranker                 │
│  nav?       → web_search() + slot context injection       │
│  code?      → _safe_execute_python() (if code block)      │
│  exact_fact?→ Wikipedia + web search                      │
│  attachment?→ PyMuPDF text extraction                     │
└───────────────────────────────────────────────────────────┘
        │
        ▼ (context-enriched prompt)
┌───────────────────────────────────────────────────────────┐
│                  LLM Inference                             │
│  _ollama_call(model, messages, options)                   │
│  ← throttled by concurrency semaphore (max 2 concurrent)  │
│  ← GPU layers injected automatically                      │
│  ← history compressed if len > keep_recent               │
└───────────────────────────────────────────────────────────┘
        │
        ▼ (raw LLM response)
┌───────────────────────────────────────────────────────────┐
│                  Post-processing Pipeline                  │
│  clean_output()           → normalise whitespace, filter  │
│  enforce_opinion_style()  → remove bounce-back questions  │
│  _strip_casual_fillers()  → remove interjection noise     │
│  _block_contradictions_against_evidence() → hedge errors  │
│  convert_latex_to_readable() → render LaTeX → Unicode     │
│  markdown_table_to_html()    → pipe tables → HTML         │
└───────────────────────────────────────────────────────────┘
        │
        ▼ (rendered HTML/text)
   PyQt6 Chat View (QLabel / QTextEdit)
```

### Data Flow — Step by Step

1. **User submits message.** Qt emits signal with text + optional attachment path.
2. **Language detection.** `detect_language()` runs against the text. Result routes to Filipino (Taglish) or English system prompt.
3. **Trivial social fast-path.** `is_trivial_social_message()` checks if the entire message is a greeting/reaction. If yes, goes straight to LLM with a short prompt — no search, no math, no history compression.
4. **Math intercept.** `detect_math_query()` scans for mathematical patterns. If detected, `_sympy_precompute()` runs in a separate timed thread and produces verified computation hints.
5. **Active task context derivation.** `_build_active_task_context(history)` scans the last 20 messages to find any in-progress task (navigation, planning, code debugging) and extract its slot values.
6. **Continuation vs. topic shift.** `_classify_continuation(query, ctx)` decides if the current message extends the active task or starts a new one.
7. **Intent classification.** `_classify_query_intent(query)` routes to one of 14 intent labels.
8. **Tool execution.** Based on intent: web search (DuckDuckGo scraping + TF-IDF reranking), Wikipedia lookup, PDF extraction, or Python sandbox execution.
9. **Prompt assembly.** Math hints, web context, slot values, and persona instructions are injected into the message list.
10. **Context compression.** If history exceeds the recent-turn window, the older turns are replaced with an LLM-generated bullet-point summary.
11. **LLM call.** The assembled prompt is sent to Ollama. A semaphore ensures a maximum of 2 concurrent calls at any time.
12. **Post-processing.** Response is cleaned, profanity filtered, LaTeX rendered, contradiction-hedged.
13. **UI update.** Rendered HTML is pushed to the chat bubble via Qt signal.
14. **Persistence.** Session JSON is scheduled for atomic background write.

### Component Interaction

```
main.py
│
├── Concurrency Layer
│   ├── Ollama semaphore          (max 2 concurrent LLM calls)
│   ├── Background thread pool    (4 workers)
│   └── Debounced disk writer     (atomic, 400ms delay)
│
├── Intent & Routing Layer
│   ├── detect_language()
│   ├── detect_math_query()
│   ├── is_trivial_social_message() + has_real_request_signal()
│   ├── _classify_query_intent()        ← 14-way router
│   ├── _build_active_task_context()    ← task state machine
│   ├── _classify_continuation()        ← continuation/shift
│   └── _infer_active_task_intent()     ← history inheritance
│
├── Math Engine
│   ├── _sympy_precompute()             ← timed thread wrapper
│   ├── _sympy_precompute_inner()       ← actual SymPy logic
│   └── _generate_math_plot()           ← matplotlib PNG
│
├── Sandbox
│   └── _safe_execute_python()          ← subprocess + blocklist
│
├── Web RAG
│   ├── DuckDuckGo scraper
│   ├── BeautifulSoup parser
│   └── HashingVectorizer + cosine reranker (MMR)
│
├── Emergency System
│   ├── keyword scanner (Filipino + English)
│   └── SMTP email dispatcher
│
├── UI Layer (PyQt6)
│   ├── ChatSession             ← per-session widget
│   ├── QThread workers         ← LLM streaming
│   ├── QSyntaxHighlighter      ← code coloring
│   └── QOpenGLWidget (opt.)    ← skeleton animation
│
└── Persistence
    ├── save_json()             ← debounced async write
    ├── save_json_immediate()   ← blocking atomic write
    └── save_attachment()       ← content-hash copy
```

### Architecture Notes

There is no traditional backend/frontend split. Everything runs in-process in a single Python application. The "backend" logic (LLM calls, math, web search, sandbox) runs in QThread workers. The "frontend" (UI widgets) runs in the Qt main thread. Communication between them uses Qt signals and slots — the only cross-thread synchronization mechanism used.

---

## Core Modules Explanation

### 1. Intent Classification System

The front-door router for every user message. It classifies the query into one of 14 intent labels without making any LLM calls — it is entirely rule-based using regex patterns and keyword lookups, running in under 1 millisecond.

**14 intent labels:**

| Label | Meaning | Example |
| --- | --- | --- |
| `emotional` | User is expressing distress | "I'm so stressed" |
| `casual` | Greeting or social filler | "hi", "haha", "thanks" |
| `reaction` | Confusion or conversational repair | "what do you mean?" |
| `code` | Programming or debugging | "debug my Python script" |
| `translation` | Language translation request | "translate this to Spanish" |
| `text_task` | Text transformation (summarize, rephrase) | "summarize this paragraph" |
| `creative` | Creative writing | "write me a poem" |
| `planning` | Schedule or study plan creation | "make me a study plan" |
| `navigation` | Transit directions | "how do I get to BGC?" |
| `live_info` | Current events or real-time data | "what's the weather today?" |
| `exact_fact` | Factual lookup | "who wrote Noli Me Tangere?" |
| `explainer` | Concept explanation | "how does DNS work?" |
| `general` | Catch-all fallback | anything else |

**Evaluation order matters.** Each check is tried top-to-bottom; the first match wins. `emotional` is checked first to avoid mis-routing distress as a factual question. `planning` is checked before `live_info` so that "I study 2 hours today" doesn't trigger web search.

---

### 2. Active Task Context System

Tracks an "active task" across multiple conversation turns. Without this, every follow-up message would be routed independently and lose context — e.g. "which station?" after a navigation request would get no useful answer.

**How it works:**

1. Scans the last 20 messages backward to find the most recent task-type message (navigation, planning, code debugging, etc.).
2. Reads the stored routing decision from each message rather than reclassifying — this preserves task continuity even when the raw label shifted.
3. Extracts entity slots from each turn using regex: origin/destination for navigation, exam date/hours for planning, error type/file for code debugging.
4. Slots are merged using per-field policies: `preserve` (never overwrite established facts like exam date), `accumulate` (union of transit modes), `update` (latest value wins for evolving state).
5. Returns an `ActiveTaskContext` dataclass with `intent`, `domain`, `slots`, `turn_count`, and `is_search_backed`.

**Continuation classification** runs before standalone intent classification and decides:
- `"continuation"` — clearly same task; inherit intent
- `"correction"` — user challenges prior answer; stay in task
- `"topic_shift"` — explicit reset phrase; start fresh
- `"fresh"` — no active task or no clear signal; fall back to standalone classification

Trivial social messages always return `"fresh"` — they can never inherit a search-backed task (preventing "thanks" after a navigation turn from re-triggering a web search).

**Active task injection example:**
```
📌 ACTIVE TASK [TRANSIT / ROUTE PLANNING] — follow-up turn 2:
  • Origin: Cubao
  • Destination: Makati
  • Transit modes: lrt, bus
→ This message is a follow-up to the task above. Answer in that context.
```

---

### 3. Math Engine (SymPy Pre-Computer)

Runs before the LLM call. If the query is math-related, SymPy attempts to compute a verified answer and injects it into the prompt as a "hint". This prevents the LLM from hallucinating arithmetic.

**Gate logic — `detect_math_query()`** uses 8 ordered checks:
1. Rejects date/time/version strings (e.g. "3.12" or "12:00 PM")
2. Rejects code-only queries unless they explicitly say "solve" or "calculate"
3. Checks for strong mathematical intent keywords
4. Checks for symbolic math characters (`²`, `∫`, `√`, `±`, `x^2`)
5. Checks for strong non-math context (search/grep/file) and suppresses math routing
6. Checks for weak non-math context (code/debug terms) and suppresses unless algebraic structure is also present
7. Checks 12 multi-pattern word-problem regex patterns (purchase arithmetic, distance-rate, age comparison, area/perimeter, discount, average, combinatorics, remaining quantity)
8. Falls back to word-equation patterns and spelled-number arithmetic

**SymPy operations supported:**

| Operation | How triggered |
| --- | --- |
| Equation solving | Regex for "solve … = …" or bare algebraic equations like `15x+8=99` |
| Arithmetic evaluation | Regex for "what is / calculate … [expression]" |
| Percentage computation | Regex for "N% of X" or "N% off X" |
| Derivatives | Regex for "derivative of … with respect to …" |
| Integrals | Regex for "integral of … with respect to …" |
| Factorisation | Regex for "factor / factorize …" |
| Bare expression fallback | Tries to parse the query directly as a math expression |

**Plot generation:** Scans for `y = f(x)` patterns in the user message and LLM response. Plots up to 4 curves on a dark-themed matplotlib figure. Saves to a temp PNG and displays inline in the chat bubble.

---

### 4. Sandbox Execution System

Executes user-provided Python code in an isolated child process with multiple security restrictions.

**Security layers (defense-in-depth):**

1. **Regex blocklist** — compiled before execution. Blocks OS/process calls, code injection (`exec`, `eval`), filesystem writes/deletes, network access, and debugger escape hatches.

2. **Subprocess isolation** — runs Python with the `-I` (isolated) flag, which ignores `PYTHONPATH`, `PYTHONSTARTUP`, and user site packages. Working directory is set to the system temp folder, not the project directory. A minimal environment is passed (only `PATH`, `SYSTEMROOT`, `TEMP`, `TMP`).

3. **Wall-clock timeout** — raises `TimeoutExpired` after a configurable number of seconds (default 5s). Guards against infinite loops.

4. **Output capping** — stdout and stderr are both capped at 1200 characters with a truncation notice. Prevents flooding the chat with large outputs.

5. **Temp file cleanup** — the `.py` temp file is always deleted in a `finally` block regardless of success or failure.

**Returns:** `(success: bool, stdout: str, stderr: str)`

---

### 5. Web Search and RAG System

When the intent is `live_info`, `exact_fact`, or `navigation`, the app scrapes live web results and injects the most relevant passages into the LLM prompt.

**Pipeline:**
1. **Query rewriting** — the raw user query is cleaned and simplified for web search.
2. **DuckDuckGo scraping** — HTTP GET to `html.duckduckgo.com` with timeout protection. BeautifulSoup parses the result HTML.
3. **Result extraction** — titles, snippets, and URLs are extracted from DDG result links.
4. **Content fetching** *(optional)* — top result URLs are followed and converted from HTML to readable text.
5. **Chunking** — long fetched pages are chunked into ~400-token passages.
6. **Reranking** — a `HashingVectorizer` vectorizes the query and all passages. Cosine similarity scores each passage. An MMR (Maximal Marginal Relevance) pass adds diversity by penalising redundant passages.
7. **Context injection** — the top-ranked passages are formatted into a `[WEB CONTEXT]` block prepended to the user message.
8. **Contradiction checking** — post-processes the LLM response against the injected evidence, hedging any sentence whose year, authorship, or numeric claim differs from the retrieved data by more than 20%.

**Wikipedia fallback** — for `exact_fact` queries, the app also tries the Wikipedia REST API before or alongside web search.

---

### 6. Language Detection System

A multi-tier rule-based language detector with a `langdetect` tiebreaker. Results are cached with a thread-safe LRU cache (max 300 entries).

**Detection priority:**

1. **Script detection (100% reliable):** Checks Unicode ranges for Hiragana/Katakana (→ Japanese), Hangul (→ Korean), Arabic, Devanagari (→ Hindi), Thai, Cyrillic (→ Russian), Greek, CJK (→ Chinese). Japanese is checked before CJK because Japanese uses both Kana and Kanji.

2. **Filipino hard signals:** Checks against a list of 400+ unambiguous Tagalog/Taglish words (`kumusta`, `kasi`, `naman`, `yung`, `nako`, etc.) and common but ambiguous particles (`po`, `ba`, `na`, `lang`).

3. **English safety net:** If 35% or more of word tokens are English function words with zero Filipino hits, returns English. If both English and Filipino hits are present, returns Taglish (`tl`).

4. **Per-language scoring:** Spanish, French, German, Italian, Portuguese, Indonesian, Vietnamese each have hard and soft word sets. The highest-scoring language wins if its score meets the threshold.

5. **Korean/Japanese romanized:** Checks for romanized markers (`annyeong`, `konnichiwa`, etc.).

6. **langdetect tiebreaker:** Only for messages of 6 or more words with low English ratio. Handles edge cases the rule sets miss.

7. **Default:** English.

---

### 7. Emergency Alert System

Detects life-threatening phrases in the user's message and immediately sends email alerts to configurable responders.

**Detection:** Keyword matching across two languages and four categories:
- Filipino: `FIRE` (sunog!, nag-aapoy), `THIEF` (magnanakaw!, holdap!), `MEDICAL` (tulong!, saklolo!), `POLICE` (pulis!, krimen!)
- English: `FIRE` (fire!, burning!), `THIEF` (thief!, robbery!), `MEDICAL` (help!, emergency!), `POLICE` (police!, crime!)

**Dispatch:** Email sent via Gmail SMTP over TLS (port 587). Rate-limited to prevent repeated alerts on false positives.

**Configuration — environment variables:**
- `MARIA_EMAIL_SENDER` — Gmail address used as "from"
- `MARIA_EMAIL_PASSWORD` — Gmail app password (not the account password)
- `MARIA_RESPONDER_POLICE`, `MARIA_RESPONDER_FIRE`, `MARIA_RESPONDER_MEDICAL`, `MARIA_RESPONDER_FAMILY` — recipient addresses per category
- `MARIA_EMAIL_COOLDOWN` — rate-limit cooldown in seconds

The app responds to the user immediately with a bilingual alert message while the email send happens in the background.

---

### 8. Adaptive Context Compression

When the conversation history exceeds the configured recent-turn window (default: last 6 messages), older turns would push the model's context window limit and cause Ollama to truncate or error.

**How it works:**
1. The older slice (everything before the recent window) is formatted into a compact transcript (capped at 400 characters per message to control token count).
2. An LLM summarization call is made: "Summarize this conversation as 4–8 bullet points. Copy EXACT values verbatim…". Temperature is set to near-zero (0.1) for deterministic output.
3. A mandatory preserve block is injected if an active task context with slots is present — this prevents critical values (exam date, error type, origin/destination) from being paraphrased into uselessness.
4. The summary replaces the older turns. The recent messages are appended after it.
5. Falls back to just the raw recent slice if summarization returns an insufficient result.

**Task-aware extension:** If an active task is in progress, the recent-turn window is extended by 6 to reduce the chance of anchor turns being compressed too early.

---

### 9. Async JSON Writer

A production-grade persistence mechanism that prevents blocking the UI thread on disk I/O and avoids corrupt files on crash.

**Design:**
- **Debouncing:** Calling the write function 50 times in 400ms fires exactly one write, 400ms after the last call. This handles rapid message sequences without excessive disk writes.
- **Atomic writes:** Uses `tempfile.mkstemp()` + `os.replace()`. A crash between write and replace leaves the old file intact.
- **Per-path locking:** Each file path has its own lock. This allows concurrent writes to different paths without contention.
- **Immediate write** is the blocking equivalent — used only at app shutdown to guarantee the final state is flushed to disk before the process exits.

---

### 10. Opinion Style Enforcement

Maria is designed to give opinions directly and not end responses with "but what do you think?" bounce-back questions — a common LLM habit.

The enforcement function:
1. Strips hedging openers like "Hmm, not sure about that one—".
2. If the response ends with a bounce-back question, strips it.
3. Checks inline patterns like "Answer — but what do you think?" and removes the bounce-back clause.
4. Leaves interactive mode intact when the user explicitly asks for a back-and-forth debate or quiz.

---

### 11. UI Reasoning Preview Card

Each user message triggers a "reasoning preview card" — a 2–4 line UI widget showing what Maria is thinking about doing before the response arrives. It is intent-aware and generates different text for every mode.

**Example (navigation intent):**
```
Route question — sorting out destination from the travel details.
Pinning down the route details before giving directions.
[active] Putting this into clear, step-by-step directions.
```

Each preview line has a visible label, a tooltip description, a stage tag (`understand`, `route`, `tool`, `compose`, `review`), and an active flag indicating the current step. Exactly one line is marked active at a time.

---

### 12. Text Formatting Layer

All LLM responses go through a text rendering pipeline before display in PyQt6:

| Function | Purpose |
| --- | --- |
| `clean_output()` | Collapses excessive newlines, normalises heading/bold spacing, filters profanity |
| `convert_latex_to_readable()` | Converts LaTeX math to Unicode and HTML with math font stack |
| `markdown_table_to_html()` | Converts pipe-separated Markdown tables to inline-styled HTML for Qt |
| `clean_ascii_text()` | Fixes Mojibake (e.g. `â€œ` → `"`, `Ã©` → `é`) from incorrectly decoded UTF-8 |
| `_process_text_blocks()` | Splits mixed Markdown text into typed blocks: paragraph, heading, bullet, code, etc. |
| `SmartTextWrapper` | Splits paragraphs longer than 1500 characters at sentence boundaries |

A static HTML+CSS header string is built once at module import and reused for every chat bubble, defining the font stack, link styles, inline code styling, and math rendering styles.

---

### 13. AdvancedIntelligenceSystem

A supplementary reasoning orchestrator that runs alongside the primary intent router. It classifies queries into reasoning strategies and generates structured reasoning plans used as internal context for the LLM prompt.

**Reasoning strategies:**
- `DEDUCTIVE` — logic-based reasoning (if A and B then C)
- `INDUCTIVE` — pattern-based reasoning (observations → general principle)
- `ABDUCTIVE` — best-explanation reasoning (observations → most likely cause)
- `ANALOGICAL` — comparison-based reasoning (known domain maps to unknown)
- `CAUSAL` — cause-effect reasoning

**Key operations:**
- Topic shift detection via Jaccard similarity between current query concepts and prior message concepts
- Semantic analysis to extract concept categories and question type (`what`=definition, `how`=process, `why`=explanation)
- Knowledge linking via an inverted index built at startup — O(1) lookup per concept

---

## Key Functions Reference

### `_TimeoutAdapter`

**Purpose:** Custom HTTP adapter that enforces default (connect, read) timeouts on every HTTP request, preventing any accidentally timeout-less call from hanging a worker thread indefinitely.

- Connect timeout: 5 seconds (default)
- Read timeout: 10 seconds (default)
- If the caller already sets a timeout, it is respected and not overridden.

---

### `_ollama_call(**kwargs)`

**Purpose:** Centralised, throttled wrapper around `ollama.chat()`. Every LLM inference call in the application goes through this function.

**Behaviour:**
1. Injects the probed GPU layer count into the call options (without overriding if already set).
2. Acquires a concurrency semaphore (max 2 concurrent holders). Any third concurrent caller blocks until one of the first two finishes.
3. Calls `ollama.chat()` while holding the semaphore.
4. Releases the semaphore on return or exception.

Must always be called from a worker thread, never from the Qt main thread.

---

### `_safe_execute_python(code, timeout=5)`

**Purpose:** Run arbitrary Python code from the user in a safe, isolated environment.

**Parameters:**
- `code (str)` — Python source code to execute
- `timeout (int)` — wall-clock seconds before the child process is killed (default 5)

**Returns:** `(success: bool, stdout: str, stderr: str)`

**Flow:**
1. Check against the security blocklist — if blocked, return an error immediately without spawning a subprocess.
2. Write code to a named temp file.
3. Run in an isolated subprocess with minimal environment.
4. Truncate stdout and stderr if they exceed 1200 characters.
5. Always delete the temp file in a `finally` block.

---

### `_generate_math_plot(user_text, response_text)`

**Purpose:** Render a dark-themed PNG plot for any `y = f(x)` expressions found in the user query or LLM response.

**Returns:** Absolute path to a temp PNG file, or `None` if nothing plottable was found.

**Notes:**
- Plots up to 4 curves per figure
- Y-axis is clipped to the 5th–95th percentile of the data to prevent extreme asymptotes from collapsing the interesting part of the graph
- Non-finite values (discontinuities) render as gaps, not spikes

---

### `detect_math_query(text)`

**Purpose:** Gate function — determines whether to run the expensive SymPy computation pipeline before the LLM call.

**Returns:** `bool` — True if the query should be treated as math.

Uses 8 ordered checks covering date/time rejection, keyword detection, symbolic character detection, non-math context suppression, and 12 word-problem pattern regexes.

---

### `_compress_history(history, model, keep_recent=6, active_ctx=None)`

**Purpose:** Replace old conversation turns with an LLM-generated bullet-point summary to stay within the model's context window.

**Parameters:**
- `history (list)` — full conversation history
- `model (str)` — Ollama model name for summarization
- `keep_recent (int)` — number of most recent messages to keep verbatim (default 6)
- `active_ctx` — if an active task is running, keep_recent is extended by 6

**Returns:** New history list (compressed older turns + verbatim recent turns).

---

### `save_attachment(src_path)`

**Purpose:** Copy an uploaded file into the permanent `maria_attachments/` directory with a content-hash filename to prevent duplicates.

**Returns:** Path to the permanent copy in the attachments directory.

**Logic:**
1. If the file is already in the attachments folder, return it as-is.
2. Compute SHA-256 of file content in 64 KB chunks.
3. Use the first 16 hex characters of the hash as the filename prefix.
4. If the destination already exists (same content), skip the copy — idempotent.
5. On any error, returns the original path (caller falls back to in-place use).

---

### `save_json(path, data)`

**Purpose:** Schedule an atomic, debounced background write. Returns immediately without blocking the UI thread.

**Logic:**
1. Deep-copy the data via `json.dumps` + `json.loads`.
2. Register or update the path in the writer registry.
3. Cancel any pending timer for this path.
4. Start a new timer — the file is written 400ms after the last call for this path.
5. The timer is daemonized so it will not block app shutdown.

---

### `is_connected(timeout=3)`

**Purpose:** Cached connectivity check used before web search.

**Returns:** `bool` — True if internet is reachable.

**Logic:** TCP socket connect to `1.1.1.1:53` (Cloudflare DNS). Result is cached for 30 seconds. Thread-safe via a lock.

---

### `detect_language(text)`

**Purpose:** Determine the language of a user message for routing decisions.

**Returns:** BCP-47 language code (`"en"`, `"tl"`, `"ja"`, `"ko"`, `"ar"`, `"hi"`, `"th"`, `"ru"`, `"el"`, `"zh-cn"`, `"es"`, `"fr"`, `"de"`, `"it"`, `"pt"`, `"id"`, `"vi"`).

**Edge cases:** Empty string → `"en"`. CJK + Kana in same text → Japanese wins. Taglish (English + Filipino mixed) → `"tl"`.

---

### `is_trivial_social_message(text)`

**Purpose:** Fast-path gate that prevents simple social messages from triggering web search, math computation, or context inheritance.

**Returns:** `bool` — True only if the entire message is a social input with no embedded information request.

**Conservative design:** False negatives (missing a trivial message, routing it normally) are harmless. False positives (suppressing a real request) are harmful. Designed to err on the side of False.

---

### `_classify_query_intent(query, is_filipino=False, trace_cb=None)`

**Purpose:** 14-way front-door intent router. Returns one intent label. No LLM calls. Runs in under 1ms.

**Evaluation order (must not be reordered — earlier checks are more specific):**
1. Emotional signals → `emotional`
2. Trivial social check → `casual`
3. Reaction patterns → `reaction`
4. Casual signals + short messages → `casual`
5. Code signals + backticks + def/import → `code`
6. Translation request patterns → `translation`
7. Local text task check → `text_task`
8. Creative signals + "write a/an" → `creative`
9. Planning patterns + combined-signal fallback → `planning`
10. Navigation patterns → `navigation`
11. Local file search gate → `general`
12. Live-info keywords → `live_info`
13. Exact fact starters + short queries → `exact_fact`
14. Explainer signals + how/why verbs → `explainer`
15. Short bare lookup → `exact_fact`
16. Fallback → `general`

---

### `_extract_task_slots(query, intent)`

**Purpose:** Extract structured entity values from a single user message for a given task type. Results accumulate across turns.

**Slots by intent:**
- `navigation`: origin, destination, transport_modes, station
- `planning`: exam_date, exam_time, hours_available, subject
- `code`: error_type, file, function, bug_target
- `exact_fact`: target
- `translation`: target_language, awaiting_source_text
- `text_task`: text_action, awaiting_source_text

---

### `_merge_task_slots(base, update)`

**Purpose:** Merge two slot dicts using per-field merge policies to prevent follow-up messages from corrupting primary task facts.

**Policies:**
- `preserve` — keep first non-empty value (exam_date, error_type, file, subject). These are anchor facts.
- `accumulate` — deduplicated list union (transport_modes). New modes are added without removing old ones.
- `update` — newer value overwrites (origin, destination, station, hours_available). Evolving state.

---

### `_block_contradictions_against_evidence(draft, evidence)`

**Purpose:** Hedge sentences in the LLM's draft where the claimed facts contradict the retrieved web evidence.

**Three contradiction types checked:**
1. **Year claims** — if the draft year differs from the evidence year for the same entity
2. **Authorship claims** — if the claimed author differs from the evidence author
3. **Numeric claims** — if the draft number differs from the evidence number by more than 20%

Maximum 3 hedged sentences per response to avoid cluttering the output.

---

### `validate_python_syntax(code)`

**Purpose:** Check whether a code string is syntactically valid Python before attempting sandbox execution.

**Returns:** `(bool, str)` — (True, "Valid Python syntax") or (False, "Syntax error: …").

---

### `detect_code_issues(code, language)`

**Purpose:** Static analysis for common bad practices in generated code.

**Python checks:** Python 2 print syntax, bare `except:`, `eval()` usage, raw `input()` without type conversion.

**Language-agnostic checks:** TODO/FIXME markers, hardcoded `password` or `secret` strings.

---

## Class-Level Documentation

### `ActiveTaskContext` (dataclass)

Lightweight per-session active-task state. Never persisted to disk — derived fresh from conversation history at the start of every turn.

**Attributes:**
- `intent: str` — active task intent (e.g. `"navigation"`)
- `domain: str` — human-readable domain label for prompt injection (e.g. `"transit / route planning"`)
- `is_search_backed: bool` — True if web search was used for this task in a prior turn
- `is_resolved: bool` — reserved for future use
- `turn_count: int` — number of user turns after the anchor turn
- `anchor_turn_idx: int` — index in recent history of the anchoring message
- `slots: dict` — extracted structured entities

**Methods:**
- `is_active() → bool` — True when intent is non-empty and is_resolved is False
- `should_inherit(standalone_intent: str) → bool` — True when a weak standalone classification should defer to this task

**Lifecycle:** Created at the start of each turn. Consumed by continuation classification, slot formatting, reasoning preview, and context compression. Discarded after the turn completes.

---

### `ReasoningPreviewLine` (dataclass)

One line in the pre-response reasoning preview card shown in the UI while Maria is processing.

**Attributes:**
- `text: str` — short visible label shown in the UI card
- `detail: str` — longer tooltip or expanded description
- `stage: str` — `"understand"` | `"route"` | `"tool"` | `"compose"` | `"review"`
- `active: bool` — whether this line is the current step (spinner shown)

---

### `ReasoningPreviewData` (dataclass)

Container for a complete preview card — the user text that triggered it plus the ordered list of reasoning lines.

**Attributes:**
- `user_text: str`
- `lines: list[ReasoningPreviewLine]`

Supports JSON serialization for session persistence.

---

### `SmartTextWrapper`

Static utility class for text layout. Splits long paragraphs at sentence boundaries to keep chat bubbles readable.

**Key method:** `split_long_paragraphs(text, max_paragraph_length=800) → List[str]`
- If the paragraph is short enough, returns it as-is.
- Splits on sentence-ending punctuation.
- Merges sentences greedily until the next sentence would exceed the limit.
- If no sentence boundaries exist, splits at the word midpoint.

---

### `AdvancedIntelligenceSystem`

Supplementary reasoning orchestrator. Provides domain detection, strategy selection, and structured reasoning plans as context supplements for the LLM.

**Key attributes:**
- `knowledge_base: dict` — static concept/relationship database covering programming, mathematics, science, and philosophy
- `_concept_index: dict` — inverted index built at startup for O(1) concept category lookup
- `reasoning_patterns: dict` — dispatch table mapping strategy names to their handler methods

**Lifecycle:** One instance is created per session and reused across all queries in that session.

---

### `ReasoningStrategy` (Enum)

**Values:** `DEDUCTIVE`, `INDUCTIVE`, `ABDUCTIVE`, `ANALOGICAL`, `CAUSAL`

Tags which reasoning approach the `AdvancedIntelligenceSystem` selects for a given query, used in prompt construction and UI display.

---

### `ReasoningPlan` (dataclass)

**Attributes:**
- `strategy: ReasoningStrategy`
- `steps: List[str]` — ordered reasoning steps as human-readable strings
- `verification_points: List[str]` — checkpoints for self-verification
- `confidence_threshold: float` — threshold below which the system should hedge its answer

---

## UI / UX Flow

### Main Window Layout

```
┌──────────────────────────────────────────────────────────────┐
│  [≡ Sidebar]  [Session Tabs]                 [Settings ⚙]   │
├──────────────┬───────────────────────────────────────────────┤
│              │  Chat View (scrollable)                        │
│  Session     │  ┌──────────────────────────────────────────┐ │
│  Sidebar     │  │  [User message bubble]                   │ │
│              │  │  [Reasoning preview card]                 │ │
│  • Session 1 │  │  [Assistant response bubble]             │ │
│  • Session 2 │  │    [code block with copy button]          │ │
│  • Session 3 │  │    [math plot image]                      │ │
│  + New       │  │  [Action bar: copy | speak | export]      │ │
│              │  └──────────────────────────────────────────┘ │
│              ├───────────────────────────────────────────────┤
│              │  [Input field]  [📎 Attach]  [▶ Send]        │
└──────────────┴───────────────────────────────────────────────┘
```

### User Actions and Event Handling

| User action | Handler | Thread |
| --- | --- | --- |
| Type and submit message | Send button or Enter key | Main thread → launches QThread worker |
| Attach file | File dialog → save_attachment() | Main thread |
| Click Copy on message | Copies rendered text to clipboard | Main thread |
| Click Speak on message | pyttsx3 TTS (if available) | Main thread (platform TTS is async) |
| Click Export as PDF | ReportLab → file save dialog | Main thread |
| Switch session tab | Load session JSON, render history | Main thread (chunked via QTimer) |
| New session button | Create entry in sessions dict, switch tab | Main thread |
| Scroll up in chat | Normal Qt scroll — no lazy loading | Qt native |

### Streaming Response Display

The QThread worker receives token-by-token streaming output from Ollama. It emits a signal with the accumulated partial text after each token. The main thread's slot appends the new token to the chat bubble, giving the appearance of real-time typing. A typing indicator (animated dots or OpenGL skeleton loader) is shown while waiting for the first token.

### Code Block Handling

When the response contains triple-backtick code blocks, they are:
1. Extracted by regex
2. Displayed in a styled `QPlainTextEdit` with monospace font and syntax highlighting
3. A "Copy" button overlays the top-right corner
4. An "Artifact" viewer panel on the right side shows the most recent code block in full for easy editing

### Math Plot Display

After the LLM response is complete, the math plot generator is called. If it returns a PNG path, the image is loaded and inserted into the chat bubble below the text response.

---

## API / Data Flow

### Session JSON Schema (`maria_sessions.json`)

```json
{
  "sessions": {
    "<session_id>": {
      "name": "Session Name",
      "created": "2025-01-01T00:00:00",
      "messages": [
        {
          "role": "user",
          "content": "message text",
          "_routed_intent": "navigation",
          "timestamp": "2025-01-01T00:01:00",
          "attachments": ["<hash>.pdf"],
          "reasoning_preview": { "user_text": "...", "lines": [] }
        }
      ],
      "model": "llama3",
      "settings": {}
    }
  },
  "active_session": "<session_id>"
}
```

The `_routed_intent` field on each user message is how the active task context system knows the final routing outcome of prior turns — even for turns where the standalone classification label differed from the actual task (e.g. a correction classified as `reaction` but routed as `navigation`).

### LoRA Training Data Schema (`maria_training_data.json`)

```json
{
  "training_examples": [
    {
      "input": "user message text",
      "output": "approved assistant response",
      "timestamp": "2025-01-01T00:00:00",
      "intent": "navigation",
      "language": "tl"
    }
  ]
}
```

### Ollama API Contract

Every LLM call passes through the centralised call wrapper:

```python
ollama.chat(
    model="llama3",
    messages=[
        {"role": "system", "content": "<persona + context>"},
        {"role": "user",   "content": "<history or summary>"},
        {"role": "user",   "content": "<current query + math hints + web context>"}
    ],
    options={
        "temperature": 0.7,
        "num_predict": 2048,
        "num_ctx": 8192,
        "num_gpu": "<auto-detected GPU layers>"
    },
    stream=False
)
```

### Emergency Email Structure

```
Subject: 🚨 EMERGENCY ALERT - FIRE DETECTED [MARIA AI ASSISTANT]
From: sender@gmail.com
To: responder@example.com

--- EMERGENCY ALERT ---
Type: FIRE
Language: ENGLISH
User Message: [original message]
Time: 2025-01-01 12:00:00
Location: Unknown (user reported)

This is an automated emergency alert from the Maria AI Assistant.
Please respond immediately to this situation.
```
