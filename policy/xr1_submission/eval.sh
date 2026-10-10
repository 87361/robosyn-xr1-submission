#!/usr/bin/env bash
set -Eeuo pipefail
policy_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
package=$(cd "$policy_dir/../.." && pwd)
exec "${LAUNCHER_PYTHON:-python3}" "$package/tools/launch.py" "$@"
