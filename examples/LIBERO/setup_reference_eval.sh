#!/usr/bin/env bash
set -euo pipefail
echo "setup_reference_eval.sh is deprecated; using the canonical native LIBERO setup." >&2
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../gr00t/eval/sim/LIBERO/setup_libero.sh" "$@"
