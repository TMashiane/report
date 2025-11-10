import os
import time
from typing import Dict
import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float,
                    outdir: str, name: str = 'best.pt'):
    ensure_dir(outdir)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric,
    }
    torch.save(state, os.path.join(outdir, name))


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, ckpt_path: str) -> Dict:
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['model'], strict=False)  # Allow missing/unexpected keys
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    return state

