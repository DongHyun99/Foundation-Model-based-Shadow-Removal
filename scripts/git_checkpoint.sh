#!/usr/bin/env bash
# Guarded checkpoint: stage, refuse heavy/binary artifacts, commit (always new), push to origin main.
# Never amends, never force-pushes.
set -euo pipefail
cd "$(dirname "$0")/.."

msg="${1:?usage: git_checkpoint.sh <message>}"
max_bytes=$((5 * 1024 * 1024))

git add -A

bad=0
while IFS= read -r path; do
  [ -z "$path" ] && continue
  if [[ "$path" =~ \.(png|jpe?g|safetensors|bin|ckpt|pt)$ ]] || [[ "$path" == venv/* ]]; then
    echo "checkpoint: refusing forbidden path: $path" >&2
    bad=1
  elif [ -f "$path" ] && [ "$(stat -c %s "$path")" -gt "$max_bytes" ]; then
    echo "checkpoint: refusing file > 5 MB: $path" >&2
    bad=1
  fi
done < <(git diff --cached --name-only --diff-filter=ACMR)

if [ "$bad" -ne 0 ]; then
  git reset -q
  exit 1
fi

if git diff --cached --quiet; then
  echo "checkpoint: nothing to commit"
else
  git commit -q -m "$msg"
fi
git push origin main
