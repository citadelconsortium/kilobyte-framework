#!/usr/bin/env bash
set -euo pipefail

# One-line installer entry point:
# curl -fsSL https://raw.githubusercontent.com/citadelconsortium/kilobyte-framework/main/scripts/install-online.sh | bash
REPO_URL="${KILOBYTE_REPO_URL:-https://github.com/citadelconsortium/kilobyte-framework}"
BRANCH="${KILOBYTE_BRANCH:-main}"
# Must match install.sh and install-model.sh: the service account, not the person
# running the installer. Using the login user here left the service running as one
# account with its data owned by another.
OWNER="${KILOBYTE_USER:-kilobyte}"
WORK="$(mktemp -d -t kilobyte-install.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v tar >/dev/null || { echo "tar is required" >&2; exit 1; }
curl --fail --location --retry 5 --output "$WORK/source.tar.gz" \
  "$REPO_URL/archive/refs/heads/$BRANCH.tar.gz"
tar -xzf "$WORK/source.tar.gz" -C "$WORK"
ROOT="$(find "$WORK" -mindepth 1 -maxdepth 1 -type d -name '*-'"$BRANCH" -print -quit)"
[[ -n "$ROOT" ]] || { echo "downloaded repository has no source directory" >&2; exit 1; }
sudo KILOBYTE_USER="$OWNER" "$ROOT/scripts/install.sh"
sudo systemctl restart kilobyte.service
echo "Kilobyte Framework installed. Run: kilo (then /cloud or /gguf)"
