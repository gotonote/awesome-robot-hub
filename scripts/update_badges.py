#!/usr/bin/env python3
"""Update the chapter/doc-count badges in README.md with real numbers.

Counts:
- chapters: directories matching `NN_*` (e.g. 01_入门指引 ... 15_实战教程)
- docs: every .md file inside those chapter directories (README.md included)

Rewrites the two shields.io badges in README.md:
  [![收录章节](https://img.shields.io/badge/📚-15%20章节-22c55e?style=for-the-badge)]()
  [![收录文档](https://img.shields.io/badge/📄-63%20文档-3b82f6?style=for-the-badge)]()

Idempotent: leaves the file untouched when numbers already match.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

CHAPTER_RE = re.compile(r"^\d{2}_.+$")


def count_chapters() -> int:
    return sum(1 for p in ROOT.iterdir() if p.is_dir() and CHAPTER_RE.match(p.name))


def count_docs() -> int:
    return sum(1 for p in ROOT.glob("[0-9][0-9]_*/**/*.md") if p.is_file())


def main() -> int:
    if not README.is_file():
        print(f"README.md not found at {README}", file=sys.stderr)
        return 1

    chapters, docs = count_chapters(), count_docs()
    text = README.read_text(encoding="utf-8")

    new = re.sub(r"📚-\d+%20章节", f"📚-{chapters}%20章节", text)
    new = re.sub(r"📄-\d+%2B?%20文档", f"📄-{docs}%20文档", new)

    if new == text:
        print(f"no change (chapters={chapters}, docs={docs})")
        return 0

    README.write_text(new, encoding="utf-8")
    print(f"updated badges: {chapters} chapters, {docs} docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
