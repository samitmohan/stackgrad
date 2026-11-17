"""Check post.md against the published post's text, then build stackgrad-karpathy-v2.html from it.

    uv run --with markdown python drafts/karpathy/assemble.py
"""
import re
from pathlib import Path

import markdown

HERE = Path(__file__).parent
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?(?:e-?\d+)?")
MATH = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]{1,200}?)\$", re.S)
FENCE = re.compile(r"```.*?```", re.S)
# each heading's part of the published post, for the "original" links
LIVE = {"Autograd": "autograd", "Tokenization": "tokens", "The transformer": "model", "Training it": "training",
        "GPUs": "gpu", "Writing an attention kernel": "attention", "Running someone else's model": "qwen",
        "Serving more than one person": "serving", "Making decoding faster": "quant", "Fine-tuning with LoRA": "lora",
        "Learning from its own answers": "grpo", "Retrieval and a calculator": "tools",
        "Bonus: scaling laws": "appendix", "Bonus: GPT-2 and Stable Diffusion": "appendix"}


def numbers(text):
    text = text.replace("−", "-")
    return {n.replace(",", "").rstrip(".") for n in NUM.findall(text)}


def prose(text):
    """Words a reader reads as text: no code blocks, no table rows."""
    text = FENCE.sub(" ", text)
    return [line for line in text.split("\n") if not line.strip().startswith("|")]


def problems(draft, original):
    """Numbers the published post never states, dashes, emoji, and links from outside REFS.md."""
    out = []
    text = re.sub(r"\]\([^)]*\)", "]", FENCE.sub("", draft))  # link URLs hold arXiv ids, not claims
    # 10^{14} and 10{,}000 are LaTeX spellings of numbers the post writes as 3x10^14 and 10,000
    text = re.sub(r"10\^\{(\d+)\}", r"10\1", text).replace("{,}", ",")
    for n in sorted(numbers(text) - numbers(original)):
        line = next((l for l in draft.split("\n") if n in l.replace(",", "")), "")
        out.append(f"number not in the published post: {n}   ({line.strip()[:90]})")
    if re.search("[–—]", draft):
        out.append("en or em dash")
    if re.search("[\U0001F300-\U0001FAFF☀-➿]", draft):
        out.append("emoji")
    allowed = set(re.findall(r"<(https?://[^>]+)>", (HERE / "REFS.md").read_text()))
    out += [f"link not in REFS.md: {u}" for u in re.findall(r"\]\((https?://[^)\s]+)\)", draft)
            if u not in allowed and "samitmohan" not in u]
    return out


def render(md):
    """Markdown to HTML with math kept away from the Markdown parser, marked for KaTeX as \\( \\) and \\[ \\]."""
    fences, maths = [], []
    md = FENCE.sub(lambda m: fences.append(m.group(0)) or f"@@F{len(fences) - 1}@@", md)

    def stash(m):
        display, inline = m.group(1), m.group(2)
        body = display if display is not None else inline
        # a lone $ before a price ("$7 ticket and $7 popcorn") is money, not math: math has TeX in it
        if display is None and not re.search(r"[\\^_{}=]|^[A-Za-z]$|^[A-Za-z]\s*\(", body.strip()):
            return m.group(0)
        maths.append((display is not None, body.strip()))
        return f"@@M{len(maths) - 1}@@"

    md = MATH.sub(stash, md)
    md = re.sub(r"@@F(\d+)@@", lambda m: fences[int(m.group(1))], md)
    html = markdown.markdown(md, extensions=["fenced_code", "tables"])
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
    return re.sub(r"@@M(\d+)@@", lambda m: (lambda d, b: f"\\[{esc(b)}\\]" if d else f"\\({esc(b)}\\)")(*maths[int(m.group(1))]), html)


draft, original = (HERE / "post.md").read_text(), (HERE / "original.txt").read_text()
html = render(draft)
toc = ['<li><a href="#intro">Intro</a></li>']


def heading(m):
    title = re.sub(r"<[^>]+>", "", m.group(1))
    anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    toc.append(f'<li><a href="#{anchor}">{title}</a></li>')
    live = LIVE.get(title)
    link = f'<a class="orig" href="https://samitmohan.in/blog/minitorch#{live}">original</a>' if live else ""
    return f'<h2 id="{anchor}">{m.group(1)}{link}</h2>'


html = '<p class="origline"><a class="orig" href="https://samitmohan.in/blog/minitorch#intro">original</a></p><section id="intro"></section>' + re.sub(r"<h2>(.*?)</h2>", heading, html)
words, was = sum(len(l.split()) for l in prose(draft)), sum(len(l.split()) for l in prose(original))
page = (HERE / "template.html").read_text().replace("{{TOC}}", "\n".join(toc)).replace("{{SECTIONS}}", html)
page = page.replace("{{TOTAL}}", f"{words:,} words, against {was:,} in the post's text")
(HERE / "stackgrad-karpathy-v2.html").write_text(page)
issues = problems(draft, original)
print("\n".join(issues) if issues else "ok: every number is in the published post, no dashes or emoji, every link from REFS.md")
print(f"{words:,} words of prose, against {was:,}")
