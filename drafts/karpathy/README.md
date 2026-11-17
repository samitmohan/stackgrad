# The post in Karpathy's voice (draft)

The whole post is one file, `post.md`. Edit it, then rebuild and refresh the page:

    uv run --with markdown python drafts/karpathy/assemble.py
    open drafts/karpathy/stackgrad-karpathy-v2.html

The build prints any number the published post does not contain (with the line it is on), any en or em dash or
emoji, and any link that is not in `REFS.md`. `original.txt` is the published post's text it checks against.
Math is LaTeX: `$...$` inline, `$$...$$` on their own lines. A figure is one line, `> **Figure:** caption`.
Once the draft is approved it moves into `ui/src/post/` and the portfolio's copy of the post.
