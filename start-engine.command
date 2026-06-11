#!/usr/bin/env bash
# macOS double-click launcher. Opens in Terminal and runs start-engine.sh.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/start-engine.sh"
