"""Minimal tool-using agent running against the local mimOE OpenAI-compatible endpoint.

Design: raw OpenAI-SDK calls (no agent framework) with a hybrid loop.
SmolLM-360M is too small to follow a tool-calling protocol reliably (tested:
native tool_calls and a TOOL:/ANSWER: text protocol both failed), so
  1. plain code routes the question to a tool (calculator / clock),
  2. the tool runs deterministically,
  3. the local model turns the observation into the final natural-language
     answer, or answers directly if no tool applies.
"""
import ast
import operator
import os
import re
import sys
from datetime import datetime

from openai import OpenAI

BASE_URL = os.getenv("MIMOE_BASE_URL", "http://localhost:8083/mimik-ai/openai/v1")
API_KEY = os.getenv("MIMOE_API_KEY", "1234")
MODEL = os.getenv("MIMOE_MODEL", "smollm-360m")

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


def calculator(expr: str) -> str:
    """Safe arithmetic evaluator (AST-based, no eval())."""
    return str(_eval(ast.parse(expr.strip(), mode="eval").body))


def current_time(_: str = "") -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = {"calculator": calculator, "current_time": current_time}

WORDS = {"times": "*", "multiplied by": "*", "plus": "+", "minus": "-",
         "divided by": "/", "^": "**"}
TIME_RE = re.compile(r"\b(time|date|today)\b", re.I)
MATH_RE = re.compile(r"\d\s*([-+*/^%]|times|plus|minus|divided by|multiplied by)\s*\d", re.I)


def extract_expr(q: str) -> str:
    for word, sym in WORDS.items():
        q = q.replace(word, sym)
    return re.sub(r"[^0-9+\-*/().% ]", " ", q).strip()


ROUTES = [(TIME_RE, "current_time", lambda q: ""),
          (MATH_RE, "calculator", extract_expr)]

SYSTEM = "You are a concise assistant. Answer in one or two sentences."


def ask(prompt: str) -> str:
    return client.chat.completions.create(
        model=MODEL, temperature=0.1, max_tokens=80, stop=["\n\n"],
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
    ).choices[0].message.content.strip()


def run_agent(question: str, verbose: bool = True) -> str:
    for pattern, name, to_input in ROUTES:
        if pattern.search(question):
            arg = to_input(question)
            try:
                obs = TOOLS[name](arg)
            except Exception as e:
                obs = f"error: {e}"
            if verbose:
                print(f"[tool] {name}({arg!r}) -> {obs}")
            fact = f"{arg} = {obs}" if arg else f"The current date and time is {obs}"
            text = ask(f"Fact: {fact}.\nQuestion: {question}\n"
                       "Answer the question in one short sentence using the fact.")
            # a 360M model may garble the fact; always surface the grounded result
            return text if obs in text else f"{text}\n(tool result: {fact})"
    if verbose:
        print("[tool] none -> answering directly")
    return ask(question)


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
