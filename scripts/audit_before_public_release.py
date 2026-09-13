#!/usr/bin/env python
from pathlib import Path
import re, sys

ROOT = Path(__file__).resolve().parents[1]
BAD_SUFFIXES = {".pt", ".pth", ".ckpt", ".dcm", ".edf", ".mat"}
PATTERNS = {
    "credential-like token": re.compile(r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*[^\s]+"),
    "private key": re.compile(r"BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY"),
}

problems=[]
for p in ROOT.rglob('*'):
    if not p.is_file() or '.git' in p.parts:
        continue
    if p.suffix.lower() in BAD_SUFFIXES:
        problems.append(f"restricted binary suffix: {p.relative_to(ROOT)}")
        continue
    if p.stat().st_size > 5_000_000:
        problems.append(f"large file >5 MB: {p.relative_to(ROOT)}")
    try:
        txt=p.read_text(errors='ignore')
    except Exception:
        continue
    for name, pat in PATTERNS.items():
        if pat.search(txt):
            problems.append(f"{name}: {p.relative_to(ROOT)}")

if problems:
    print('Repository audit FAILED:')
    for x in sorted(set(problems)):
        print(' -', x)
    sys.exit(1)
print('Repository audit passed. Manual PHI review is still required before public release.')
