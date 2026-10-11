"""Cloud-only: add the selected single-task checkpoint to a new immutable model bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
import yaml



def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    p = argparse.ArgumentParser()
    for name in ("base-models", "checkpoint", "config", "output", "evaluation-record"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument('--task', choices=['table_rearrangement', 'drawer_open_place', 'click_bell'], required=True)
    a = p.parse_args()
    step = {'table_rearrangement': 5000, 'drawer_open_place': 10000, 'click_bell': 3000}[a.task]
    artifact = f'{a.task}_160k_ft{step // 1000}k'
    selection = json.loads(a.evaluation_record.read_text())
    assert selection['task'] == a.task and selection['checkpoint_step'] == step
    assert selection['passed'] and selection['episodes'] == 50 and selection['successes'] == {'table_rearrangement': 50, 'drawer_open_place': 48, 'click_bell': 50}[a.task]
    assert Path(selection['checkpoint']).resolve() == a.checkpoint.resolve()
    source_sha = sha(a.checkpoint)
    assert source_sha == selection['checkpoint_sha256']
    torch.set_num_threads(4)
    base = json.loads((a.base_models / "manifest.json").read_text())
    cfg = yaml.safe_load(a.config.read_text())
    data = cfg["data"]["params"]["train_datasets"]
    assert data["action_representation"] == "joint_absolute"
    assert data['gripper_encodings'][a.task] == {'table_rearrangement':'physical_0.05', 'drawer_open_place':'normalized_1', 'click_bell':'zero'}[a.task]
    assert set(data["paths"]) == {a.task}
    model_params = cfg["model"]["params"]["model"]
    assert model_params["action_length"] == 30
    shared_cfg = yaml.safe_load((a.base_models / "shared_160k/config.yaml").read_text())
    for key in ("mean", "std", "q01", "q99", "action_representation"):
        assert data[key] == shared_cfg["data"]["params"]["train_datasets"][key], key
    for key, value in shared_cfg["model"]["params"]["model"].items():
        assert model_params[key] == value, key
    a.output.mkdir(parents=True, exist_ok=False)
    for name, record in base["files"].items():
        src = a.base_models / name
        assert src.resolve().is_relative_to(a.base_models.resolve()), name
        assert src.stat().st_size == record["bytes"] and sha(src) == record["sha256"], name
        dest = a.output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Large baseline tensors stay byte-identical; never edit these linked files.
        if src.suffix == ".pt":
            os.link(src, dest)
        else:
            shutil.copy2(src, dest)
    target = a.output / artifact
    target.mkdir()
    raw = torch.load(a.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = raw.get("module", raw)
    assert state and all(k.startswith("model.") and isinstance(v, torch.Tensor) for k,v in state.items())
    baseline = torch.load(a.base_models / "shared_160k/model.pt", map_location="cpu", mmap=True, weights_only=True)
    assert set(state) == set(baseline)
    for k,v in state.items():
        assert v.shape == baseline[k].shape and v.dtype == baseline[k].dtype, k
        assert torch.isfinite(v).all(), k
    torch.save(state, target / "model.pt")
    exported = torch.load(target / "model.pt", map_location="cpu", mmap=True, weights_only=True)
    assert set(exported) == set(state)
    for k in state:
        assert torch.equal(state[k], exported[k]), k
    clean = {"data": {"params": {"train_datasets": {k:data[k] for k in
        ("mean", "std", "q01", "q99", "action_representation")}}},
        "model": {"params": {"model": model_params}}}
    (target / "config.yaml").write_text(yaml.safe_dump(clean, sort_keys=False))
    evidence = {
        "passed": True, "task": a.task, "artifact": artifact,
        "source_run": a.config.parent.name, "checkpoint_step": step, "base_checkpoint_step": 160000,
        "source_checkpoint_sha256": source_sha, "selection_record_sha256": sha(a.evaluation_record), "source_config_sha256": sha(a.config),
        "tensor_count": len(state), "all_tensors_finite": True,
        "tensor_keys_shapes_dtypes_match_shared": True, "exported_tensors_exactly_equal_source": True,
        "normalization_equal_to_shared": True, "horizon": 30, "execute_steps": 29, "prefix_steps": 0,
        "training_data": {"kind": "official pre-collected Sim release", "task": a.task,
                          "episodes": 1000},
        "model_sha256": sha(target / "model.pt"), "config_sha256": sha(target / "config.yaml"),
        "runtime_inference_verified": False, "end_to_end_verified": False,
    }
    manifest = {"schema": 2, "artifact": "xr1_sim10_selected_seven_h30",
        "baseline_provenance": base.get("baseline_provenance", {k:v for k,v in base.items() if k != "files"}),
        "artifacts": {**base.get("artifacts", {}), artifact: evidence},
        "processor_revision": base["processor_revision"], "files": dict(base["files"])}
    for f in sorted(target.iterdir()):
        manifest["files"][str(f.relative_to(a.output))] = {"bytes": f.stat().st_size, "sha256": sha(f)}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (a.output.parent / f'{a.task}_export.json').write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)

if __name__ == "__main__":
    main()
