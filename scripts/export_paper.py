import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURVE_POINTS = 700  # RunCurve draws at most this many points; sending more to a static page only adds bytes
RUNS = ROOT / "runs"


def _read_jsonl(path):
  # a diverged run logs NaN, which python's json writes but strict JSON (the site's JSON.parse) rejects: it becomes null
  return [json.loads(line, parse_constant=lambda _: None) for line in path.open()] if path.exists() else []


def runs():
  out = []
  for d in sorted(RUNS.glob("*/metrics.jsonl")):
    rows = _read_jsonl(d)
    out.append({"name": d.parent.name, "config": json.loads((d.parent / "config.json").read_text()),
                "steps": len(rows), "last": rows[-1] if rows else None, "has_checkpoint": (d.parent / "ckpt.npz").exists()})
  return out


def bench():
  return {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "bench" / "results").glob("*.json"))}


def downsample(rows, n):
  """Picks the rows the site's src/paper/minitorch/lib.ts downsample picks."""
  if len(rows) <= n:
    return rows
  k = -(-len(rows) // n)
  out = rows[::k]
  return out if out[-1] is rows[-1] else out + [rows[-1]]


def slim(rows, fields):
  return [{k: r[k] for k in fields if r.get(k) is not None} for r in rows]


def story_run():
  """Cut to the curve the post draws."""
  rows = _read_jsonl(RUNS / "tinystories" / "metrics.jsonl")
  evals = [r for r in rows if r.get("val_loss") is not None]
  train = downsample([r for r in rows if r.get("val_loss") is None], CURVE_POINTS - len(evals))
  kept = sorted({id(r): r for r in [rows[0], *train, *evals, rows[-1]]}.values(), key=lambda r: r["step"])
  return slim(kept, ["step", "loss", "val_loss", "time"])


def attention(patterns):
  """Lower triangle in integer thousandths: the server rounds weights to 3 decimals."""
  return [{"tokens": p["tokens"], "layers": [[[[round(w * 1000) for w in row[: i + 1]] for i, row in enumerate(head)]
                                               for head in layer] for layer in p["layers"]]} for p in patterns]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("site", type=Path, help="the site's root, holding src/ and public/")
  site = p.parse_args().site
  data, public = site / "src" / "paper" / "minitorch" / "data", site / "public" / "paper" / "minitorch"
  data.mkdir(parents=True, exist_ok=True)
  public.mkdir(parents=True, exist_ok=True)

  results = {
      "bench": bench(), "runs": runs(), "ts": story_run(),
      "lora": slim(_read_jsonl(RUNS / "lora_gsm8k" / "metrics.jsonl"), ["step", "loss", "val_loss"]),
      "grpo": _read_jsonl(RUNS / "grpo_gsm8k" / "metrics.jsonl"),
      "grpoNoStop": _read_jsonl(RUNS / "grpo_no_stop" / "metrics.jsonl"),
  }
  (data / "results.json").write_text(json.dumps(results, separators=(",", ":")) + "\n")

  replays = {f.stem: json.loads(f.read_text()) for f in sorted((ROOT / "bench" / "replays").glob("*.json"))}
  (public / "attention.json").write_text(json.dumps(attention(replays.pop("attention")), separators=(",", ":")) + "\n")
  (data / "replays.json").write_text(json.dumps(replays, ensure_ascii=False, separators=(",", ":")) + "\n")
  for name in ["stable_diffusion_stackgrad.png", "stable_diffusion_diffusers.png"]:
    shutil.copy(ROOT / "bench" / "results" / name, public / name)

  for f in [data / "results.json", data / "replays.json", public / "attention.json"]:
    print(f"{f.relative_to(site)}: {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
  main()
