"""Check the actual CUDA driver and a PyTorch kernel before loading any weights."""
import ctypes
import json
import os
import sys


def main():
    record = {'python': sys.executable, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    try:
        for device in ('/dev/nvidiactl', '/dev/nvidia-uvm'):
            try:
                fd = os.open(device, os.O_RDWR)
                os.close(fd)
                record[device] = 'open_ok'
            except OSError as exc:
                record[device] = str(exc)
        cuda = ctypes.CDLL('libcuda.so.1')
        record['cuInit'] = cuda.cuInit(0)
        if record['cuInit']:
            raise RuntimeError('CUDA driver initialization failed before model/simulator startup')
        import torch
        record.update(torch=torch.__version__, torch_cuda=torch.version.cuda)
        x = torch.ones(8, device='cuda:0')
        assert (x + x).sum().item() == 16.0
        torch.cuda.synchronize()
        record.update(passed=True, device=torch.cuda.get_device_name(0))
    except Exception as exc:
        record.update(passed=False, error=str(exc))
    print(json.dumps(record), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
