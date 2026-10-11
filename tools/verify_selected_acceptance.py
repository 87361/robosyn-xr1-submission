"""Audit a selected checkpoint through the packaged runtime after one real episode."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import numpy as np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--challenge-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert (a.run/"exit.code").read_text().strip() == "0"
    m = json.loads((a.run/"metrics.json").read_text())
    c = json.loads((a.run/"launch_contract.json").read_text())
    task = m['config']['task']
    mapping = json.loads((root/'task_mapping.json').read_text())
    entry = mapping['tasks'][task]
    artifact = entry['artifact']
    k = entry.get('execute_steps', mapping['execute_steps'])
    assert len(m['episodes']) == 1
    assert m["summary"]["success_count"] == 1
    assert c["artifact"].endswith("/" + artifact)
    assert (c["horizon"], c["execute_steps"], c["prefix_steps"]) == (30,k,0)
    assert c["attention_backend"] == "sdpa"
    import yaml
    assert yaml.safe_load((root/'policy/xr1_submission/deploy_policy.yml').read_text())['eval_reset_sync_steps'] == 0
    lifecycle = json.loads((a.run/"evaluator_lifecycle.json").read_text())
    assert lifecycle["official_evaluator_returned"] and lifecycle["official_metrics_verified_before_teardown"]
    assert c["evaluator_wrapper_sha256"] == sha(root/"tools/run_official_evaluator.py")
    baseline = json.loads((root/"verification/bell50_preflight.json").read_text())
    for entry in baseline["verified_official_files"]:
        f = a.challenge_root/entry["path"]
        data = f.read_bytes()
        digest = hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
        assert digest == entry["git_blob"], entry["path"]
    assert sha(a.challenge_root/"scripts/eval_policy.py") == c["evaluator_sha256"]
    rows = [json.loads(l) for l in (a.run/"server/requests.jsonl").read_text().splitlines()]
    assert rows and len({r["episode_id"] for r in rows}) == 1
    assert [r["chunk_index"] for r in rows] == list(range(len(rows)))
    assert all(r["policy_seed"] == 20260908 and r["prefix_length"] == 0 for r in rows)
    traces = list((a.run/"traces").glob("*.jsonl"))
    chunks = []
    step_counts = []
    for f in traces:
        for line in f.read_text().splitlines():
            row = json.loads(line)
            if "target_joint_action" in row:
                arr = np.asarray(row["target_joint_action"])
                assert arr.shape == (30,14) and np.isfinite(arr).all()
                assert row['k'] == k
                chunks.append(row)
                step_counts.append(0)
            elif row.get('kind') == 'step':
                assert step_counts
                step_counts[-1] += 1
    assert len(chunks) == len(rows)
    assert all(count == k for count in step_counts[:-1]) and 0 < step_counts[-1] <= k
    assert sum(step_counts) == m['episodes'][0]['action_steps']
    assert c['task_mapping_sha256'] == sha(root/'task_mapping.json')
    assert c['eval_command'][c['eval_command'].index('--xr1_execute_steps')+1] == str(k)
    videos = list((a.run/"raw").rglob("*.mp4"))
    assert len(videos) == 1
    probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries",
        "stream=width,height,nb_frames:format=duration","-of","json",str(videos[0])],text=True))
    assert float(probe["format"]["duration"]) > 0
    assert probe["streams"][0]["width"] == 1920 and probe["streams"][0]["height"] == 480
    port = int(c["server_command"][c["server_command"].index("--port")+1])
    with socket.socket() as sock:
        sock.settimeout(2)
        assert sock.connect_ex(("127.0.0.1",port)) != 0
    files = c["model_manifest"]["files"]
    report = {"passed":True, "task":task, "episodes":1, "successes":1,
        "model_sha256":files[artifact + "/model.pt"]["sha256"],
        "config_sha256":files[artifact + "/config.yaml"]["sha256"],
        "summary":m["summary"], "metrics_sha256":sha(a.run/"metrics.json"),
        "evaluator_sha256":c["evaluator_sha256"], "official_files_verified":len(baseline["verified_official_files"]),
        "runtime_source_sha256":{str(f.relative_to(root)):sha(f)
            for directory in ("runtime","policy") for f in (root/directory).rglob("*")
            if f.is_file() and f.suffix in (".py",".sh",".yml")},
        "evaluator_lifecycle":lifecycle,
        "model_versions":{n:importlib.metadata.version(n) for n in ("torch","numpy","transformers")},
        "gpu":subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],text=True).strip(),
        "attention_backend":"sdpa", "execute_steps":k, "actual_steps_per_chunk":step_counts, "eval_reset_sync_steps":0, "prefix_steps":0,
        "finite_action_chunks":len(chunks), "inference_listener_cleaned_up":True,
        "video":probe, "boundaries":["one episode on existing L20 environments",
            "not fresh CUDA12.8/RTX5090 installation acceptance", "not a new 50-episode evaluation"]}
    for name in ("tools/launch.py", "tools/run_official_evaluator.py", "tools/sim_runtime.py", "task_mapping.json"):
        report["runtime_source_sha256"][name] = sha(root/name)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":True,"summary":report["summary"]}))

if __name__ == "__main__":
    main()
