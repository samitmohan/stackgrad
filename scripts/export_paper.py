import argparse
import difflib
import json
import re
import shutil
import sys
from pathlib import Path

from stackgrad.serve.api import RUNS, _read_jsonl, bench, runs

ROOT = Path(__file__).resolve().parent.parent
CURVE_POINTS = 700  # RunCurve draws at most this many points; sending more to a static page only adds bytes
PARAGRAPH = re.compile(r"<p(?:\s[^>]*)?>(.*?)</p>", re.DOTALL)
CHROME = {"Head.tsx", "kit.tsx"}  # the port's header and data loading are its own


def downsample(rows, n):
    """Picks the rows ui/src/lib.ts's downsample picks."""
    if len(rows) <= n:
        return rows
    k = -(-len(rows) // n)
    out = rows[::k]
    return out if out[-1] is rows[-1] else out + [rows[-1]]


def median(xs):
    return sorted(xs)[len(xs) // 2]  # kit.tsx's median: the upper middle of an even count


def slim(rows, fields):
    return [{k: r[k] for k in fields if r.get(k) is not None} for r in rows]


def story_run():
    """Cut to the curve the post draws; the medians the prose quotes cover every step."""
    rows = _read_jsonl(RUNS / "tinystories" / "metrics.jsonl")
    evals = [r for r in rows if r.get("val_loss") is not None]
    train = downsample([r for r in rows if r.get("val_loss") is None], CURVE_POINTS - len(evals))
    kept = sorted({id(r): r for r in [rows[0], *train, *evals, rows[-1]]}.values(), key=lambda r: r["step"])
    stats = {"tokS": median([r["tokens_per_s"] for r in rows]), "tflops": median([r["tflops"] for r in rows])}
    if rows[0].get("mfu") is not None:
        stats["mfu"] = median([r.get("mfu") or 0 for r in rows])
    return slim(kept, ["step", "loss", "val_loss", "time"]), stats


def attention(patterns):
    """Lower triangle in integer thousandths: the server rounds weights to 3 decimals."""
    return [{"tokens": p["tokens"], "layers": [[[[round(w * 1000) for w in row[: i + 1]] for i, row in enumerate(head)]
                                                 for head in layer] for layer in p["layers"]]} for p in patterns]


def prose_drift(port):
    def paragraphs(f):
        return [" ".join(p.split()) for p in PARAGRAPH.findall(f.read_text())]
    drift = []
    for f in sorted((ROOT / "ui" / "src" / "post").glob("*.tsx")):
        if f.name not in CHROME and (port / f.name).exists():  # Post.tsx is the live page's shell, Paper.tsx the port's
            drift += difflib.unified_diff(paragraphs(port / f.name), paragraphs(f), f"port {f.name}", f"ui {f.name}", n=0, lineterm="")
    return drift


def main():
    p = argparse.ArgumentParser()
    p.add_argument("site", type=Path, help="the site's root, holding src/ and public/")
    site = p.parse_args().site
    data, public = site / "src" / "paper" / "minitorch" / "data", site / "public" / "paper" / "minitorch"
    data.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)

    ts, stats = story_run()
    samples = _read_jsonl(RUNS / "tinystories" / "samples.jsonl")
    results = {
        "bench": bench(), "runs": runs(), "ts": ts, "tsStats": stats,
        "lora": slim(_read_jsonl(RUNS / "lora_gsm8k" / "metrics.jsonl"), ["step", "loss", "val_loss"]),
        "grpo": _read_jsonl(RUNS / "grpo_gsm8k" / "metrics.jsonl"),
        "grpoNoStop": _read_jsonl(RUNS / "grpo_no_stop" / "metrics.jsonl"),
        "sample": samples[-1]["text"] if samples else None,
    }
    (data / "results.json").write_text(json.dumps(results, separators=(",", ":")) + "\n")

    replays = {f.stem: json.loads(f.read_text()) for f in sorted((ROOT / "bench" / "replays").glob("*.json"))}
    (public / "attention.json").write_text(json.dumps(attention(replays.pop("attention")), separators=(",", ":")) + "\n")
    (data / "replays.json").write_text(json.dumps(replays, ensure_ascii=False, separators=(",", ":")) + "\n")
    for name in ["stable_diffusion_stackgrad.png", "stable_diffusion_diffusers.png"]:
        shutil.copy(ROOT / "bench" / "results" / name, public / name)

    for f in [data / "results.json", data / "replays.json", public / "attention.json"]:
        print(f"{f.relative_to(site)}: {f.stat().st_size / 1024:.0f} KB")

    port = site / "src" / "paper" / "minitorch" / "post"
    if not port.is_dir():  # without it the comparison below finds nothing to compare and passes
        sys.exit(f"no port of the post at {port}")
    drift = prose_drift(port)
    if drift:
        sys.exit("The post's paragraphs and the port's differ; make each edit in both:\n" + "\n".join(drift))
    print("prose: every section's paragraphs match the port's")


if __name__ == "__main__":
    main()
