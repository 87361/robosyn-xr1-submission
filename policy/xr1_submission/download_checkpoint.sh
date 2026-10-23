#!/usr/bin/env bash
# Download and verify the complete, pinned XR1 ten-task checkpoint repository.
# Usage: bash download_checkpoint.sh [MODEL_DIR]   (default: ./models)

set -Eeuo pipefail

POLICY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=${1:-$POLICY_DIR/models}
MODEL_REPO=87361Yly/robosyn-xr1-sim10
MODEL_REVISION=fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4
XR1_PYTHON=${XR1_PYTHON:-$POLICY_DIR/.xr1_env/bin/python}
HF_CLI=${HF_CLI:-$POLICY_DIR/.xr1_env/bin/hf}

if [[ $MODEL_DIR != /* ]]; then
    MODEL_DIR=$POLICY_DIR/$MODEL_DIR
fi
if [[ $XR1_PYTHON != /* ]]; then
    XR1_PYTHON=$POLICY_DIR/$XR1_PYTHON
fi
if [[ $HF_CLI != /* ]]; then
    HF_CLI=$POLICY_DIR/$HF_CLI
fi

if [[ ! -x $XR1_PYTHON ]]; then
    echo "Error: cannot find XR1 inference Python: $XR1_PYTHON" >&2
    echo "Run bash $POLICY_DIR/install.sh first, or set XR1_PYTHON." >&2
    exit 1
fi
if [[ ! -x $HF_CLI ]]; then
    echo "Error: cannot find Hugging Face CLI: $HF_CLI" >&2
    echo "Run bash $POLICY_DIR/install.sh first, or set HF_CLI." >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"
MODEL_DIR=$(cd -- "$MODEL_DIR" && pwd)

echo "Downloading $MODEL_REPO@$MODEL_REVISION"
echo "Destination: $MODEL_DIR"
"$HF_CLI" download "$MODEL_REPO" \
    --revision "$MODEL_REVISION" \
    --local-dir "$MODEL_DIR"

"$XR1_PYTHON" - "$MODEL_DIR" "$POLICY_DIR/task_mapping.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
mapping_path = Path(sys.argv[2])
manifest_path = root / "manifest.json"
processor = root / "processor"
if not manifest_path.is_file():
    raise SystemExit(f"Missing model manifest: {manifest_path}")
if not processor.is_dir() or not any(path.is_file() for path in processor.rglob("*")):
    raise SystemExit(f"Missing or empty processor directory: {processor}")

manifest = json.loads(manifest_path.read_text())
records = manifest.get("files")
if not isinstance(records, dict):
    raise SystemExit(f"Invalid files table in {manifest_path}")
mapping = json.loads(mapping_path.read_text())
required = set()
for entry in mapping["tasks"].values():
    artifact = entry["artifact"]
    required.update((f"{artifact}/model.pt", f"{artifact}/config.yaml"))

missing_records = sorted(required - records.keys())
if missing_records:
    raise SystemExit("Manifest is missing selected checkpoint files: " + ", ".join(missing_records))

for index, (name, record) in enumerate(sorted(records.items()), 1):
    path = (root / name).resolve()
    if root not in path.parents:
        raise SystemExit(f"Unsafe manifest path: {name}")
    if not path.is_file():
        raise SystemExit(f"Missing model file: {path}")
    if path.stat().st_size != record.get("bytes"):
        raise SystemExit(f"Size mismatch: {name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != record.get("sha256"):
        raise SystemExit(f"SHA-256 mismatch: {name}")
    print(f"Verified [{index}/{len(records)}] {name}", flush=True)

print(f"Model verification passed: {len(records)} files, 10 task mappings, processor present")
PY

echo "Models ready: $MODEL_DIR"
