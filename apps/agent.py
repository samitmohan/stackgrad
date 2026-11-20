import argparse
import ast
import functools
import json
import operator
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.gsm8k import final_number, gold_number, matches

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results"
TOOLS = [{"type": "function", "function": {
    "name": "calculator", "description": "Evaluate an arithmetic expression with + - * / ** and parentheses, e.g. 48 * (7 - 2)",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
SYSTEM = "You cannot do arithmetic yourself. For every calculation, call the calculator tool and use the result it returns."
# with "Solve it step by step" the model called the tool 0 times in 250 problems (bench/results/gsm8k_agent_step_by_step.json)
PROMPT = "{question}\nUse the calculator tool for each calculation, then give the final answer on its own line as: #### <number>"
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression):
  """Model text: parsed against an allowlist, never eval'd, and exponents capped so nothing runs for minutes."""
  def value(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
      return float(node.value)  # floats overflow to an error where ints grow without bound: ((9**64)**64)**64
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
      return OPS[type(node.op)](value(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
      left, right = value(node.left), value(node.right)
      if isinstance(node.op, ast.Pow) and abs(right) > 64:
        raise ValueError("exponent too large")
      return OPS[type(node.op)](left, right)
    raise ValueError("only numbers and + - * / ** ( ) are allowed")

  try:
    # GSM8K writes $1,250, 20% and 20% of 40
    # shortcut: a bare percent is a number, so 80 + 150% is 81.5, not 200; the model wrote that once in 250 problems
    text = expression.replace(",", "").replace("$", "").replace("%", "/100").replace("of", "*")
    result = value(ast.parse(text, mode="eval").body)
    return str(round(result, 6)).removesuffix(".0")
  except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError, TypeError) as e:  # TypeError: (-8) ** 0.5
    return f"error: {e}"


def chat(server, messages):
  body = json.dumps({"messages": messages, "tools": TOOLS, "temperature": 0, "max_tokens": 320}).encode()
  request = urllib.request.Request(f"{server}/v1/chat/completions", body, {"content-type": "application/json"})
  start = time.perf_counter()
  with urllib.request.urlopen(request, timeout=600) as r:
    out = json.load(r)
  usage = out["usage"]
  return out["choices"][0]["message"]["content"], usage["prompt_tokens"], usage["completion_tokens"], time.perf_counter() - start


def solve(ask, question, max_turns=8):
  messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT.format(question=question)}]
  trace = []
  for _ in range(max_turns):
    reply, prompt_tokens, reply_tokens, seconds = ask(messages)
    trace.append({"kind": "model", "text": reply, "prompt_tokens": prompt_tokens, "reply_tokens": reply_tokens,
                  "seconds": round(seconds, 3)})
    messages.append({"role": "assistant", "content": reply})
    calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", reply, re.DOTALL)
    if not calls:
      return reply, trace
    for call in calls:
      start = time.perf_counter()
      try:
        result = calculate(str(json.loads(call)["arguments"]["expression"]))
      except (json.JSONDecodeError, KeyError, TypeError):
        result = 'error: write {"name": "calculator", "arguments": {"expression": "..."}}'
      trace.append({"kind": "tool", "expression": call, "result": result, "seconds": round(time.perf_counter() - start, 6)})
      messages.append({"role": "tool", "content": result})
  return "", trace


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--server", default="http://localhost:8000")
  p.add_argument("--n", type=int, default=250)
  args = p.parse_args()

  with open("data/gsm8k/test.jsonl") as f:
    problems = [json.loads(line) for line in f][:args.n]
  rows, start = [], time.time()
  for i, q in enumerate(problems):
    reply, trace = solve(functools.partial(chat, args.server), q["question"])
    gold, got = gold_number(q["answer"]), final_number(reply)
    rows.append({"i": i, "gold": gold, "got": got, "correct": matches(got, gold), "reply": reply, "trace": trace})
    if (i + 1) % 25 == 0:
      print(f"{i + 1:4d}/{len(problems)}  accuracy so far {sum(r['correct'] for r in rows) / (i + 1):.3f}", flush=True)

  calls = [t for r in rows for t in r["trace"]]
  model = [t for t in calls if t["kind"] == "model"]
  summary = {"name": "agent", "n": len(rows), "correct": sum(r["correct"] for r in rows),
             "accuracy": sum(r["correct"] for r in rows) / len(rows),
             "tool_calls": sum(t["kind"] == "tool" for t in calls),
             "problems_using_tool": sum(any(t["kind"] == "tool" for t in r["trace"]) for r in rows),
             "model_calls": len(model), "model_seconds": round(sum(t["seconds"] for t in model), 1),
             "tool_seconds": round(sum(t["seconds"] for t in calls if t["kind"] == "tool"), 4),
             "prompt_tokens": sum(t["prompt_tokens"] for t in model), "reply_tokens": sum(t["reply_tokens"] for t in model),
             "seconds": round(time.time() - start, 1)}
  (RESULTS / "gsm8k_agent.json").write_text(json.dumps(summary, indent=1))
  with open(RESULTS / "gsm8k_agent_replies.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
  print(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
