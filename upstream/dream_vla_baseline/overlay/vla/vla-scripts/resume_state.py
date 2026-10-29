"""Full training state for the official Dream-VLA trainer."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def unwrap(module):
    return module.module if hasattr(module,'module') else module


def plain_values(value):
    # Dream imports json_numpy, which patches json.dumps before its default
    # hook. Convert eagerly so rank0's serializer mutation cannot change hashes.
    if isinstance(value,dict):return {key:plain_values(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [plain_values(item) for item in value]
    if isinstance(value,np.ndarray):return plain_values(value.tolist())
    if isinstance(value,np.generic):return value.item()
    return value


def make_contract(cfg, world_size, dataset=None):
    config=asdict(cfg)
    for key in ('resume_checkpoint','resume','resume_step','stop_after_updates'):
        config.pop(key,None)
    config=json.loads(json.dumps(config,default=str))
    base=Path(__file__).parent
    vla=base.parent
    files=[base/'finetune_dream.py',base/'checkpoint_retention.py',base/'resume_state.py',vla/'dreamvla/modeling_dreamvl.py',vla/'prismatic/models/action_heads.py',vla/'prismatic/models/projectors.py',vla/'prismatic/vla/datasets/datasets.py']
    files.extend(sorted((vla/'prismatic/vla/datasets/rlds').rglob('*.py')))
    sources={str(p.relative_to(vla)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    checkpoint=Path(cfg.vla_path)
    identities={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkpoint/'download_receipt.json',checkpoint/'config.json',checkpoint/'model.safetensors.index.json') if p.exists()}
    for name in ('download_receipt.json','receipt.json','dataset_receipt.json'):
        p=Path(cfg.data_root_dir)/name
        if p.exists():identities[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    # TFDS schema and shard-size inventory prevent silently swapping dataset versions.
    root=Path(cfg.data_root_dir)
    inventory={str(p.relative_to(root)):p.stat().st_size for p in sorted(root.rglob('*.tfrecord*'))}
    for p in sorted(root.rglob('dataset_info.json')):
        identities[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    statistics=None if dataset is None else json.dumps(plain_values(dataset.dataset_statistics),sort_keys=True)
    if statistics is not None:
        audit_dir=Path(cfg.run_root_dir)/'contract_audit'
        audit_dir.mkdir(parents=True,exist_ok=True)
        (audit_dir/f'dataset_statistics_rank_{torch.distributed.get_rank()}.json').write_text(statistics)
    return dict(config=config,world_size=world_size,sources=sources,asset_identities=identities,
                dataset_shard_inventory_sha256=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest(),
                dataset_statistics_sha256=None if statistics is None else hashlib.sha256(statistics.encode()).hexdigest())


def save_state(directory, modules, optimizer, scheduler, rank, progress, contract, dataset):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    saved_rng=rng_state()
    try:
        dataset.save_iterator_state(directory/f'iterator_rank_{rank}')
        payload=dict(progress=progress,contract=contract,rng=saved_rng)
        if rank==0:
            payload.update(modules={name:{key:value.detach().cpu() for key,value in unwrap(module).named_parameters() if value.requires_grad}
                for name,module in modules.items() if module is not None},
                buffers={name:{key:value.detach().cpu() for key,value in unwrap(module).named_buffers()}
                    for name,module in modules.items() if module is not None},
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict())
        torch.save(payload,directory/f'training_rank_{rank}.pt')
    finally:
        restore_rng(saved_rng)


def load_state(directory,modules,optimizer,scheduler,rank,contract,dataset):
    directory=Path(directory)
    shared=torch.load(directory/'training_rank_0.pt',map_location='cpu',weights_only=False)
    local=torch.load(directory/f'training_rank_{rank}.pt',map_location='cpu',weights_only=False)
    differences = {label: [key for key in contract if stored.get(key)!=contract[key]]
        for label,stored in [('rank0',shared['contract']),('local',local['contract'])]}
    if any(differences.values()):
        raise ValueError(f'Resume contract mismatch: {differences}')
    with torch.no_grad():
        for name,module in modules.items():
            if module is None:continue
            params={key:value for key,value in unwrap(module).named_parameters() if value.requires_grad}
            state=shared['modules'][name]
            if params.keys()!=state.keys():raise ValueError(f'Resume parameter names mismatch: {name}')
            for key,value in params.items():
                if value.shape!=state[key].shape:raise ValueError(f'Resume shape mismatch: {name}.{key}')
                value.copy_(state[key])
            buffers=dict(unwrap(module).named_buffers())
            saved_buffers=shared['buffers'][name]
            if buffers.keys()!=saved_buffers.keys():raise ValueError(f'Resume buffer names mismatch: {name}')
            for key,value in buffers.items():
                if value.shape!=saved_buffers[key].shape:raise ValueError(f'Resume buffer shape mismatch: {name}.{key}')
                value.copy_(saved_buffers[key])
    optimizer.load_state_dict(shared['optimizer']);scheduler.load_state_dict(shared['scheduler'])
    dataset.restore_iterator_state(directory/f'iterator_rank_{rank}')
    restore_rng(local['rng'])
    return local['progress']
