# mimOE local AI agent

A small tool-using agent that runs against the local mimOE OpenAI-compatible
endpoint (model: `qwen3-1.7b`), with no cloud calls.

## Run

```bash
pip install -r requirements.txt
# defaults to http://localhost:8083/mimik-ai/openai/v1; override for another host:
export MIMOE_BASE_URL=http://<host>:8083/mimik-ai/openai/v1
python agent.py "What is 15% of 240, and what's today's date?"   # one-shot
python agent.py                                                   # interactive
```

Env vars: `MIMOE_BASE_URL`, `MIMOE_API_KEY` (default `1234`), `MIMOE_MODEL`.

## How it works

1. The question goes to the local model along with two tool specs
   (`calculator`, `current_time`) via the OpenAI `tools` parameter.
2. The model decides: answer directly (greetings, general knowledge) or
   return a tool call.
3. The agent runs the tool, sends the result back as a `tool` message, and
   loops (max 5 steps) until the model gives a final answer.

## Choices

- **BYO framework = none.** The endpoint is OpenAI-compatible, so the plain
  `openai` SDK with a custom `base_url` is the thinnest path.
- **Model: qwen3-1.7b.** I first tried `smollm-360m`. It could not emit tool
  calls or follow a text protocol reliably (it even answered "hi" with
  unrelated text), so I had to route tools with regexes. `qwen3-1.7b` supports
  native tool calling, so the model now picks tools itself.
- **Tools:** a safe AST-based calculator (no `eval`) and a clock.
- **`<think>` stripping:** qwen3 emits its reasoning in `<think>` tags; the
  agent removes them before showing the answer.
- **Tool errors** are fed back to the model so it can recover.

## Model experiments: smollm-360m vs qwen3-1.7b

I iterated on the model rather than settling for the first one. Both models
were served by the same mimOE endpoint; only the model name changed.

| | smollm-360m (first attempt) | qwen3-1.7b (final) |
|---|---|---|
| Size | ~360M params | ~1.7B params (~1.5 GB) |
| Native `tool_calls` | Not usable | Works: returns structured calls |
| Text protocol (`TOOL:` / `ANSWER:`) | Ignored the format, echoed prompts | Not needed |
| Casual input ("hi") | Off-topic output (a random math statement) | "Hello! How can I assist you today?" |
| Tool answers | Needed regex routing in code; phrasing often rambled or garbled the result | Chooses tools itself, including two in one question, and answers cleanly |
| Output quirks | Rambling after the correct answer | Emits `<think>` reasoning, stripped in code |

**What I tried with smollm-360m before switching:**
1. A ReAct-style text protocol with few-shot examples, then format-correction nudges. It still ignored the format.
2. A hybrid design: regex routes the question to a tool and the model only phrases the result. This produced correct tool results but poor wording, so I added a guard that appends the tool result when the model's text omits it.

**Outcome:** a quick test of qwen3-1.7b showed native tool calling worked
out of the box, so I rewrote the agent as a proper tool-calling loop and
dropped the regex router and the guard. The agent code got simpler and the
behavior got better, so the main lever was model capability, not prompt hacks.

## Limitations

Only two toy tools, no conversation memory between questions, and a 1.7B
model can still occasionally repeat a tool call (harmless here).
