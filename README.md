# claude_converter

A zero-dependency Python module for inspecting and converting [Claude Code](https://www.anthropic.com/claude-code) session files (`.jsonl`) into the `messages` format expected by Hugging Face Transformers.

Claude Code stores every session as a JSONL file on disk under `~/.claude/projects/<encoded-project-path>/<session-uuid>.jsonl`. Each line is a JSON record containing the full message history — user prompts, assistant responses, tool calls, tool results, and extended thinking blocks. This module parses that format and flattens it into the simple `[{"role": ..., "content": ...}]` list that `tokenizer.apply_chat_template()` consumes directly.

## Why this exists

The conversion pipeline has two stages that are often conflated:

```
Claude Code JSONL  →  [claude_converter]  →  messages[]  →  apply_chat_template()  →  tokens
                         (this module)                        (transformers handles this)
```

`apply_chat_template()` handles stage 2 (turning a `messages` list into model-specific token sequences), but it knows nothing about Claude Code's JSONL format. This module handles stage 1.

## Installation

```bash
uv pip install claude-converter
```

## Quick start

```python
from claude_converter import session_to_messages

messages = session_to_messages("path/to/session.jsonl")
```

`messages` is ready for `apply_chat_template()`.

## API reference

### `load_session(path)`

Loads a Claude Code `.jsonl` file and returns the raw list of record dicts — one per line.

```python
from claude_converter import load_session

records = load_session("session.jsonl")
# [{"type": "user", "uuid": "...", "message": {...}, ...}, ...]
```

Raises `FileNotFoundError` if the file doesn't exist, `ValueError` if the extension is not `.jsonl` or `.json`, or if the file contains no valid records.

### `session_to_messages(path, output=None)`

Loads a session and converts it to a `messages` list in one call. Optionally saves the result to a JSON file.

```python
from claude_converter import session_to_messages

# In-memory only
messages = session_to_messages("session.jsonl")

# Save to disk too
messages = session_to_messages("session.jsonl", output="messages.json")
```

Returns `list[{"role": str, "content": str}]`. Content blocks (`tool_use`, `tool_result`, `thinking`, `text`) are flattened to plain text using XML-style tags so the conversation structure is preserved.

### `records_to_messages(records)`

Converts an already-loaded list of records into the messages format. Useful when you want to load once and run multiple transformations without re-reading the file.

```python
from claude_converter import load_session, records_to_messages

records  = load_session("session.jsonl")
messages = records_to_messages(records)
```

### `inspect_session(path, show_flow=False, show_blocks=False, show_raw=False)`

Prints a color-coded report of a session: record type counts, content block breakdown by role, and token usage totals. The conversation flow is optional and off by default.

```python
from claude_converter import inspect_session

# Stats only (default)
inspect_session("session.jsonl")

# Stats + timestamped conversation flow
inspect_session("session.jsonl", show_flow=True)

# Flow + content of every block inline
inspect_session("session.jsonl", show_flow=True, show_blocks=True)

# Full report: flow, blocks, and raw record structure examples
inspect_session("session.jsonl", show_flow=True, show_blocks=True, show_raw=True)
```

Example output (default):

```
══════════════════════════════════════════════════════════════════════
  CLAUDE CODE SESSION INSPECTOR
══════════════════════════════════════════════════════════════════════
  File    : session.jsonl
  Size    : 3.6 KB
  Lines   : 9
  Session : abc-123
  CWD     : /home/user/myproject
  Branch  : main
══════════════════════════════════════════════════════════════════════

  RECORD TYPES
  user         4  ████
  assistant    4  ████
  system       1  █

  CONTENT BLOCKS (by record type)
  assistant
    · text                 3
    · tool_use             2
    · thinking             1
  user
    · text                 2
    · tool_result          2

  TOKEN USAGE
  Input       : 2,617
  Output      : 147
  Cache read  : 1,700
  Cache write :   482
  Total       : 2,764
```

## Using the output with Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from claude_converter import session_to_messages

MODEL_ID = "your-model-id"

messages  = session_to_messages("session.jsonl", output="messages.json")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

tokens = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
```

## Fine-tuning local models

Claude Code sessions are a natural source of training data: they capture
real coding conversations — tool calls, reasoning traces, multi-turn edits —
in a format that maps directly to the `messages` list expected by every major
fine-tuning framework.

### Preprocessing recommendations

Before training, filter the messages list for your target model:

- **Strip tool blocks** — most local models don't understand `<tool_use>` /
  `<tool_result>` tags. Remove or replace them unless you are training a
  tool-calling model.
- **Strip thinking blocks** — `<thinking>` blocks leak chain-of-thought that
  may not generalize. Keep them only if you are distilling reasoning behavior.
- **Drop short or empty turns** — single-word assistant replies and empty
  user turns add noise.

```python
def clean_messages(messages, keep_tool_calls=False, keep_thinking=False):
    import re
    cleaned = []
    for msg in messages:
        content = msg["content"]
        if not keep_thinking:
            content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL)
        if not keep_tool_calls:
            content = re.sub(r"<tool_use.*?>.*?</tool_use>", "", content, flags=re.DOTALL)
            content = re.sub(r"<tool_result>.*?</tool_result>", "", content, flags=re.DOTALL)
        content = content.strip()
        if content:
            cleaned.append({"role": msg["role"], "content": content})
    return cleaned
```

### TRL / SFTTrainer example

```python
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from claude_converter import session_to_messages
import glob

MODEL_ID = "your-model-id"

# Load one or more sessions
all_messages = []
for path in glob.glob("~/.claude/projects/**/*.jsonl", recursive=True):
    msgs = session_to_messages(path)
    msgs = clean_messages(msgs)
    if len(msgs) >= 2:           # skip degenerate sessions
        all_messages.append({"messages": msgs})

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForCausalLM.from_pretrained(MODEL_ID)
dataset   = Dataset.from_list(all_messages)

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="./output", max_seq_length=4096),
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### Axolotl / LLaMA-Factory

Both frameworks accept a `sharegpt` format, which is structurally identical to
the `messages` list this module produces. Save the cleaned messages list with
`output="messages.json"` and point your framework config at that file.

## Content block mapping

Claude Code sessions contain several block types that have no direct equivalent in the Transformers `messages` format. This module flattens them as follows:

| Claude Code block | Flattened as |
|---|---|
| `text` | plain text, as-is |
| `thinking` | `<thinking>...</thinking>` |
| `tool_use` | `<tool_use name='...'>{input JSON}</tool_use>` |
| `tool_result` | `<tool_result>...</tool_result>` |
| `system` records | skipped (not included in messages output) |

Records with an empty content string after flattening are also skipped.

## Where Claude Code stores sessions

```
~/.claude/projects/
└── <url-encoded-project-path>/
    └── <session-uuid>.jsonl
```

The project path is URL-encoded: `/home/user/myapp` becomes `-home-user-myapp`. Each session is a separate file, append-only, one JSON object per line.

## Limitations

- **Graph structure**: Claude Code sessions are directed acyclic graphs linked by `parentUuid`. This module reads lines in file order (linear), which is correct for the vast majority of sessions. Branched or multi-agent sessions may need custom traversal.
- **Tool call fidelity**: Most local models don't natively understand `<tool_use>` or `<tool_result>` tags. For inference-only use cases, consider stripping those blocks before passing to `apply_chat_template()`.
- **No streaming**: The module loads the full file into memory. For very large sessions, use `load_session()` and process records in batches.
- **Data quality for fine-tuning**: Sessions include failed attempts, retries, and exploratory tool calls. Blindly training on raw sessions will teach the model bad habits. Filter aggressively: keep only sessions where the final assistant turn solves the stated problem, and prefer sessions with a high ratio of `text` blocks to `tool_use` blocks unless tool-calling is your training target.