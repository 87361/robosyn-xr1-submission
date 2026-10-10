"""Lifecycle regression tests; these do not replace simulator acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

WRAPPER = Path(__file__).resolve().parents[1] / "tools/run_official_evaluator.py"

class LifecycleTest(unittest.TestCase):
    def run_case(self, script):
        with tempfile.TemporaryDirectory(prefix="xr1_lifecycle_") as tmp:
            root = Path(tmp)
            (root/"gymnasium.py").write_text(
                "import os\nclass Env:\n def close(self): os._exit(0)\ndef make(*a,**k): return Env()\n")
            source = root/"evaluator.py"
            source.write_text(script)
            output = root/"raw"
            result = subprocess.run([sys.executable,str(WRAPPER),str(source),
                "--eval_result_dir",str(output),"--max_episodes","1","--task_name","item_assembly"],
                env={**os.environ,"PYTHONPATH":str(root)},capture_output=True,text=True,timeout=10)
            metrics = (output/"evaluation_metrics.json").exists()
            marker = root/"evaluator_lifecycle.json"
            phase = json.loads(marker.read_text()) if marker.exists() else None
            return result, metrics, phase

    def test_native_exit_cannot_prevent_official_metric_write(self):
        result, metrics, phase = self.run_case(
            'import gymnasium as gym,json,sys\nfrom pathlib import Path\n'
            'env=gym.make()\nenv.close()\n'
            'p=Path(sys.argv[sys.argv.index("--eval_result_dir")+1]);p.mkdir()\n'
            '(p/"evaluation_metrics.json").write_text(json.dumps({"config":{"task":"item_assembly"},'
            '"episodes":[{}],"summary":{"episode_count":1}}))\n')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(metrics)
        self.assertTrue(phase["official_metrics_verified_before_teardown"])
        self.assertEqual(phase["native_teardown"],"started")

    def test_evaluator_failure_is_not_masked_by_native_zero_exit(self):
        result, metrics, phase = self.run_case(
            'import gymnasium as gym\nenv=gym.make()\n'
            'try: raise RuntimeError("intentional evaluator failure")\n'
            'finally: env.close()\n')
        self.assertEqual(result.returncode,1)
        self.assertIn("intentional evaluator failure",result.stderr)
        self.assertFalse(metrics)
        self.assertIsNone(phase)

if __name__ == "__main__":
    unittest.main()

