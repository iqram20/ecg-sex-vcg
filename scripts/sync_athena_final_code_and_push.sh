#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "Run this script from inside a clone of iqram20/ecg-sex-vcg." >&2
  exit 1
fi
cd "$REPO_ROOT"

bash scripts/collect_final_code_from_athena.sh
python scripts/audit_before_public_release.py

git add scripts slurm docs configs README.md .gitignore .gitattributes environment.yml requirements.txt CITATION.cff LICENSE results || true

if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

git status --short

git commit -m "Add exact final ECG-sex analysis code from Athena"
git push origin main

echo "Uploaded final Athena code to GitHub."
