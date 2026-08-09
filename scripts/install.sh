#!/usr/bin/env bash
set -euo pipefail

if [[ "$EUID" -ne 0 ]]; then
    echo "Run with sudo: sudo ./scripts/install.sh" >&2
    exit 1
fi

# Self-bootstrap: when run via  curl ... | sudo bash  there is no checkout on disk,
# so clone the repo to /opt and re-exec this script from there. A normal ./scripts/
# run already has the tree and skips this.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd || true)"
if [[ -z "$ROOT" || ! -f "$ROOT/src/kilobyte/__init__.py" ]]; then
    if ! command -v git >/dev/null; then
        command -v pacman >/dev/null && pacman -Syu --needed --noconfirm git             || { command -v apt-get >/dev/null && apt-get update && apt-get install -y git; }
    fi
    DEST="/opt/kilobyte-framework"
    rm -rf "$DEST"
    git clone --depth 1 https://github.com/citadelconsortium/kilobyte-framework "$DEST"
    exec bash "$DEST/scripts/install.sh" "$@"
fi
# Default to a dedicated service account. Kilobyte runs unattended from boot, so it
# should not depend on a login user existing, and the model should not be reachable
# through a human account's permissions.
KILO_USER="${KILOBYTE_USER:-kilobyte}"
KILO_GROUP="$(id -gn "$KILO_USER" 2>/dev/null || echo "$KILO_USER")"

if command -v pacman >/dev/null; then
    # Arch only supports full upgrades. Keep llama-cpp and ggml on matching
    # versions instead of risking unresolved runtime symbols.
    pacman -Syu --needed --noconfirm llama-cpp python python-prompt_toolkit curl sqlite ripgrep
fi
command -v python >/dev/null
# The framework is deliberately brain-free: cloud mode works without a local
# runtime, while /gguf uses llama-server when the operator supplies one.
if ! command -v llama-server >/dev/null; then
    echo "note: llama-server not found; install llama-cpp before using a local GGUF (cloud mode remains available)."
fi
# The TUI needs prompt_toolkit. Prefer the distro package (installed above); fall back to
# pip so a non-Arch host still gets a working interface.
if ! python -c "import prompt_toolkit" 2>/dev/null; then
    python -m pip install --break-system-packages prompt_toolkit 2>/dev/null \
        || python -m pip install prompt_toolkit \
        || echo "warning: prompt_toolkit missing; the TUI will use the simple fallback UI" >&2
fi

if ! id "$KILO_USER" >/dev/null 2>&1; then
    if [[ "$KILO_USER" == "kilobyte" ]]; then
        echo "Creating service user: $KILO_USER"
        useradd --system --create-home --shell /usr/bin/nologin "$KILO_USER"
    else
        echo "User does not exist: $KILO_USER" >&2
        exit 1
    fi
fi
KILO_GROUP="$(id -gn "$KILO_USER")"

echo "Installing Kilobyte application..."
install -d -m 0755 /opt/kilobyte/app /etc/kilobyte
install -d -m 0750 -o "$KILO_USER" -g "$KILO_GROUP" /var/lib/kilobyte /var/lib/kilobyte/models /var/log/kilobyte
cp -a "$ROOT/src" "$ROOT/pyproject.toml" /opt/kilobyte/app/
chown -R root:root /opt/kilobyte/app
find /opt/kilobyte/app -type d -exec chmod 0755 {} +
find /opt/kilobyte/app -type f -exec chmod 0644 {} +
install -m 0755 "$ROOT/scripts/kilo-wrapper" /usr/local/bin/kilo
# The unit ships with the default account baked in; substitute the account actually
# being installed for, or the service would run as one user while its data directories
# belong to another and every write would fail.
sed -e "s/^User=.*/User=$KILO_USER/" -e "s/^Group=.*/Group=$KILO_GROUP/" \
    "$ROOT/systemd/kilobyte.service" > /etc/systemd/system/kilobyte.service
chmod 0644 /etc/systemd/system/kilobyte.service
if [[ ! -f /etc/kilobyte/policy.json ]]; then
    install -m 0600 -o "$KILO_USER" -g "$KILO_GROUP" "$ROOT/config/policy.json" /etc/kilobyte/policy.json
fi
systemctl daemon-reload
systemctl enable kilobyte.service
echo "Framework installed. Run: kilo (then /cloud or /gguf)"
