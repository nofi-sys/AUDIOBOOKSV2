
from typing import List, Dict, Any
import json, re, unicodedata

def front_matter(meta: Dict[str, Any]) -> str:
    fm = ['---']
    for k, v in meta.items():
        fm.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    fm.append('---\n')
    return "\n".join(fm)

def md_heading(level: str, title: str) -> str:
    hashes = {'h1': '#', 'h2': '##', 'h3': '###'}.get(level, '##')
    return f"{hashes} {title.strip()}\n"

def slugify(title: str) -> str:
    normalized = unicodedata.normalize('NFKD', title)
    ascii_only = normalized.encode('ascii', 'ignore').decode('ascii')
    ascii_only = ascii_only.lower()
    ascii_only = re.sub(r'[^a-z0-9\s-]', '', ascii_only)
    ascii_only = re.sub(r'\s+', '-', ascii_only).strip('-')
    return ascii_only or 'section'

def render(document: List[Dict[str, Any]], meta: Dict[str, Any], add_toc: bool = False) -> str:
    out = []
    out.append(front_matter(meta))

    if add_toc:
        out.append("## Contenidos\n")
        for blk in document:
            level = blk["type"]
            if level in ("h1", "h2", "h3"):
                title = blk["text"].strip()
                anchor = slugify(title)
                if level == "h1":
                    indent = ""
                elif level == "h2":
                    indent = "  "
                else:
                    indent = "    "
                out.append(f"{indent}- [{title}](#{anchor})")
        out.append("\n")

    for blk in document:
        if blk["type"] in ("h1", "h2", "h3"):
            out.append(md_heading(blk["type"], blk["text"]))
        elif blk["type"] == "subtitle":
            out.append(f"_{blk['text'].strip()}_\n\n")
        elif blk["type"] == "blockquote":
            # Para blockquotes, prefijar cada linea con "> "
            lines = blk['text'].strip().split('\n')
            quoted_lines = [f"> {line}" for line in lines]
            out.append("\n".join(quoted_lines) + "\n\n")
        elif blk["type"] == "poem":
            poem_lines = blk["text"].splitlines()
            out.append("\n".join(line.rstrip() for line in poem_lines) + "\n\n")
        elif blk["type"] == "letter":
            out.append(blk["text"].rstrip() + "\n\n")
        else:
            out.append(blk["text"].rstrip() + "\n\n")
    return "\n".join(out)
