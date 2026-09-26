"""Small shared helpers for configuration, provenance and reproducibility."""
from __future__ import annotations
import datetime, hashlib, json, os, random, socket, subprocess, sys
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import torch
import yaml

def read_jsonl(path: str|Path) -> list[dict[str,Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]
def write_jsonl(path: str|Path, rows: Iterable[dict[str,Any]]) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
def append_jsonl(path: str|Path,row:dict[str,Any])->None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf8') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
def sha256_file(path: str|Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge a child configuration without mutating its parent mapping."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_yaml(path: str|Path) -> dict[str,Any]:
    """Load YAML and resolve an optional relative ``_base_`` configuration.

    The resolved mapping is what model construction, calibration, and training
    all consume.  This keeps an inherited paper-ablation configuration from
    silently losing required sections such as ``alignment`` during calibration.
    """
    path = Path(path)
    value=yaml.safe_load(path.read_text(encoding='utf8'))
    if not isinstance(value,dict):raise ValueError(f'configuration must be a mapping: {path}')
    base_ref = value.pop("_base_", None)
    if base_ref is None:
        return value
    if not isinstance(base_ref, str) or not base_ref.strip():
        raise ValueError(f'_base_ must be a non-empty relative path: {path}')
    base_path = (path.parent / base_ref).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f'base configuration not found for {path}: {base_path}')
    return _deep_merge_config(load_yaml(base_path), value)
def save_yaml(path: str|Path,value:dict[str,Any]) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf8')
def seed_everything(seed:int,deterministic:bool=False)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if deterministic:torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
def worker_seed(_worker_id:int)->None:
    torch.set_num_threads(1)
    value=torch.initial_seed()%(2**32);random.seed(value);np.random.seed(value)

def configure_cuda_performance(main_threads: int = 2) -> None:
    """Use the already-frozen END execution policy, never a model change."""
    torch.set_num_threads(int(main_threads))
    if not torch.cuda.is_available(): return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

def distributed_context(enable: bool) -> tuple[int,int,int,torch.device]:
    """Initialize the one flat DDP contract used by train/validate/test."""
    if not enable:return 0,1,0,torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import torch.distributed as dist
    local=int(os.environ.get("LOCAL_RANK","0"));device=torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        torch.cuda.set_device(device)
        if not dist.is_initialized():dist.init_process_group("nccl",device_id=device)
    elif not dist.is_initialized():dist.init_process_group("gloo")
    rank,world=dist.get_rank(),dist.get_world_size()
    return rank,world,local,device

def distributed_cleanup(enable: bool) -> None:
    if enable:
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()

def all_gather_object(value: Any, enable: bool) -> list[Any]:
    if not enable:return [value]
    import torch.distributed as dist
    values=[None for _ in range(dist.get_world_size())];dist.all_gather_object(values,value);return values

def run_environment(world_size: int, device: torch.device) -> dict[str,Any]:
    gpus=[]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            info=torch.cuda.get_device_properties(index);gpus.append({"index":index,"name":info.name,"total_memory_bytes":info.total_memory})
    try:git=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:git=None
    try:status=subprocess.check_output(["git","status","--porcelain"],text=True,stderr=subprocess.DEVNULL).splitlines()
    except Exception:status=[]
    # The short UTC singleton was added after Python 3.10.  The training
    # Python 3.10-compatible implementation uses ``datetime.timezone.utc``.
    # compatible equivalent.
    return {"timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"hostname":"anonymous-host","python":sys.version,"pytorch":torch.__version__,"cuda":torch.version.cuda,"world_size":world_size,"device":str(device),"gpus":gpus,"git_commit":git,"git_dirty_paths":status}

def prepare_run(root: str|Path, run_type: str, run_id: str, overwrite: bool=False) -> Path:
    """Create the documented run layout and protect formal main runs."""
    if run_type not in {"main","smoke","validation","test","stride","benchmark","debug"}:raise ValueError("invalid run type")
    out=Path(root)/"runs"/run_type/run_id
    if out.exists() and any(out.iterdir()):
        if run_type=="main" or not overwrite:raise FileExistsError(f"refusing to overwrite {out}")
    for name in ("logs","checkpoints","metrics","predictions","reports"): (out/name).mkdir(parents=True,exist_ok=True)
    return out
