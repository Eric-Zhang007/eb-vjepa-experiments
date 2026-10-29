"""Run inside the prepared Dream environment. No training or checkpoint mutations."""
import ctypes
import importlib
import json
import torch

report={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available()}
for name in ['transformers','peft','flash_attn','tensorflow','tensorflow_datasets','dlimp','robosuite','mujoco']:
    module=importlib.import_module(name);report[name]=getattr(module,'__version__','imported')
ctypes.CDLL('libEGL.so.1');report['libEGL']='loaded'
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: check scheduler GPU allocation, driver and torch build')
report['devices']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
report['bf16_supported']=torch.cuda.is_bf16_supported()
from flash_attn import flash_attn_func
q=torch.randn(1,16,4,32,device='cuda',dtype=torch.bfloat16)
out=flash_attn_func(q,q,q,dropout_p=0.0,causal=False)
if not torch.isfinite(out).all():raise RuntimeError('Flash attention produced non-finite values')
report['flash_attention_bf16']='passed'
print(json.dumps(report,indent=2))
