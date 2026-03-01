#!/usr/bin/env bash
# Sync local audio repo to GitHub
set -e

cd "$(dirname "$0")"

if [ -z "$(git status --porcelain)" ]; then
    echo "Nothing to sync — working tree is clean."
    exit 0
fi

echo "Changes:"
git status --short
echo ""

# Stage all tracked + new files (respects .gitignore)
git add -A

# Commit with timestamp
msg="sync $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$msg"

# Push to origin
git push origin main

echo ""
echo "Synced to GitHub."
