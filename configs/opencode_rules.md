# OpenCode Optimization Rules for DEVIOS Turbo

Place this file at `.opencode/rules.md` in your project root or `~/.opencode/rules.md` in your user directory.

```markdown
# OpenCode Execution Rules

CRITICAL: Do NOT generate greeting summaries, work states, or conversational filler. Immediately invoke the requested tool or output the requested code directly. Never output markdown meta-analysis unless explicitly requested.
```

## Why this is critical:
When local LLMs run inside autonomous agent loops (like OpenCode, Cline, Claude Code), conversational fluff ("Sure, I can help you with that! Here is the plan...") generates hundreds of uninformative tokens.
By disabling conversational preamble, token generation stays tightly focused on tool JSON invocations and code blocks, maximizing the speculative decoding acceptance rate (70-85%) and eliminating token waste.
