"""Shared device policy; importing this module does not import PyTorch."""
import os
from pathlib import Path
import shutil

# Must be set before torch is imported. Unsupported MPS operators may use CPU.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def select_device(requested='auto', *, torch_module=None, allow_mps=True):
    if torch_module is None:
        import torch as torch_module
    requested = str(requested).lower()
    cuda = torch_module.cuda.is_available()
    backend = getattr(torch_module.backends, 'mps', None)
    mps = bool(backend and backend.is_available())
    if requested == 'auto':
        return 'cuda:0' if cuda else ('mps' if mps and allow_mps else 'cpu')
    if requested == 'cpu':
        return 'cpu'
    if requested == 'mps':
        if not allow_mps:
            raise ValueError('This stage supports CPU/CUDA; use --device cpu or auto.')
        if not mps:
            raise ValueError('MPS is unavailable in this PyTorch/OS installation; use --device cpu.')
        return 'mps'
    if requested == 'cuda':
        requested = 'cuda:0'
    if requested.isdigit():
        requested = f'cuda:{requested}'
    if requested.startswith('cuda:'):
        index = requested.split(':', 1)[1]
        if not cuda or not index.isdigit() or int(index) >= torch_module.cuda.device_count():
            raise ValueError(f'CUDA device {requested} is unavailable; use --device auto or cpu.')
        return requested
    raise ValueError('Device must be auto, cpu, mps, cuda, or cuda:N.')


def precision(device):
    return 16 if device.startswith('cuda') else 32


def default_batch_size(device):
    return 8 if device.startswith('cuda') else 1


def reset_peak_memory(device, *, torch_module=None):
    if not device.startswith('cuda'):
        return
    if torch_module is None:
        import torch as torch_module
    # The detector is still on CPU before its first prediction.
    torch_module.cuda.init()
    torch_module.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device, *, torch_module=None):
    if not device.startswith('cuda'):
        return None
    if torch_module is None:
        import torch as torch_module
    return round(torch_module.cuda.max_memory_allocated(device) / 1024**2, 1)


def find_ffmpeg():
    override = os.environ.get('FFMPEG_BINARY')
    if override:
        executable = shutil.which(override) or (override if Path(override).is_file() else None)
        if not executable:
            raise RuntimeError('FFMPEG_BINARY does not point to an executable.')
        return str(executable)
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    bundled = Path.home() / 'AppData/Local/Programs/Stremio/ffmpeg.exe'
    if os.name == 'nt' and bundled.is_file():
        return str(bundled)
    raise RuntimeError('Install FFmpeg and add it to PATH, or set FFMPEG_BINARY.')
