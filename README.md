# mimOE local AI agent

A small tool-using agent that runs against the local mimOE OpenAI-compatible
endpoint (model: `smollm-360m`), with no cloud calls.

## Run

```bash
pip install -r requirements.txt
# defaults to http://localhost:8083/mimik-ai/openai/v1; override for another host:
export MIMOE_BASE_URL=http://<host>:8083/mimik-ai/openai/v1
python agent.py "What is 123*45?"     # one-shot
python agent.py                       # interactive
```

Env vars: `MIMOE_BASE_URL`, `MIMOE_API_KEY` (default `1234`), `MIMOE_MODEL`.

## Choices

- **BYO framework = none.** The endpoint is OpenAI-compatible, so the plain
  `openai` SDK with a custom `base_url` is the thinnest path.
- **Hybrid tool use.** I first tried native `tool_calls` and a ReAct-style
  `TOOL:`/`ANSWER:` text protocol; a 360M model did not follow either
  reliably. So code routes the question (regex) to a tool, runs it
  deterministically, and the local model phrases the answer from the result.
- **Tools:** a safe AST-based calculator (no `eval`) and a clock.
- **Grounding guard:** if the model's text omits the tool result, the agent
  appends it, so the answer is always correct even when phrasing is poor.

## Limitations

SmolLM-360M often rambles after the correct answer; the router is keyword
based. A larger model (e.g. via mimOE's model view) would allow real
model-driven tool selection.
