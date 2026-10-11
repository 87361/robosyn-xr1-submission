"""Read the task-specific execution contract shared by launcher and direct adapter."""
import json
from pathlib import Path

MAPPING = Path(__file__).resolve().parents[2] / 'task_mapping.json'


def load_contract(task, execute_steps=None):
    mapping = json.loads(MAPPING.read_text())
    if task not in mapping['tasks']:
        raise ValueError(f'Unknown submitted task: {task}')
    entry = dict(mapping['tasks'][task])
    k = entry.get('execute_steps', mapping['execute_steps'])
    horizon, prefix = mapping['horizon'], mapping['prefix_steps']
    if horizon != 30 or prefix != 0 or type(k) is not int or not 0 < k < horizon:
        raise ValueError(f'Invalid submitted inference contract for {task}')
    if execute_steps is not None and int(execute_steps) != k:
        raise ValueError(f'{task} requires execute_steps={k}; got {execute_steps}')
    return {**entry, 'execute_steps': k, 'horizon': horizon, 'prefix_steps': prefix}
