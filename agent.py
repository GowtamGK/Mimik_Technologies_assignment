"""Tool-using agent running against the local mimOE OpenAI-compatible endpoint.

Design: raw OpenAI-SDK calls (no agent framework) with native tool calling.
The model (qwen3-1.7b) decides whether to answer directly or call a tool;
the loop executes tool calls and feeds results back until it gives an answer.
"""
import ast
import json
import operator
import os
import re
import sys
from datetime import datetime

from openai import OpenAI

BASE_URL = os.getenv("MIMOE_BASE_URL", "http://localhost:8083/mimik-ai/openai/v1")
API_KEY = os.getenv("MIMOE_API_KEY", "1234")
MODEL = os.getenv("MIMOE_MODEL", "qwen3-1.7b")
MAX_STEPS = 5

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ---------- tools ----------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Safe arithmetic evaluator (AST-based, no eval())."""
    return str(_eval(ast.parse(expression.strip(), mode="eval").body))


def current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = {"calculator": calculator, "current_time": current_time}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression such as 12*(3+4). Use for any math.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "current_time",
        "description": "Get the current local date and time.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM = (
    "You are a friendly, concise assistant. Answer greetings and general "
    "questions directly in plain language. Only call a tool when it is needed: "
    "use calculator for arithmetic and current_time for the date or time. "
    "After a tool returns, state the result in a short sentence."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def clean(text: str) -> str:
    """Drop qwen3's <think> reasoning block (also an unterminated one)."""
    text = THINK_RE.sub("", text or "")
    return text.split("<think>")[0].strip()


def run_agent(question: str, verbose: bool = True) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    for _ in range(MAX_STEPS):
        msg = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SPECS,
            temperature=0.3, max_tokens=800,
        ).choices[0].message
        if not msg.tool_calls:
            return clean(msg.content) or "(no answer)"
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                obs = TOOLS[name](**args) if name in TOOLS else f"unknown tool '{name}'"
            except Exception as e:  # feed errors back so the model can recover
                obs = f"error: {e}"
            if verbose:
                print(f"[tool] {name}({tc.function.arguments}) -> {obs}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(obs)})
    return "Stopped: max steps reached without a final answer."


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("\nFINAL:", run_agent(" ".join(sys.argv[1:])))
    else:
        print(f"mimOE agent ({MODEL} @ {BASE_URL}). Ctrl+C to quit.")
        while True:
            try:
                q = input("\nyou> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if q:
                print("FINAL:", run_agent(q))
