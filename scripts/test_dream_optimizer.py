import ast
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
import math
p=Path('/data/users/jiachangzhang/eb_vjepa/repos/Dream-VLX/vla/vla-scripts/finetune_dream.py')
tree=ast.parse(p.read_text())
fn=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='finetune')
start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='trainable_params' for t in n.targets))
end=next(i for i in range(start+1,len(fn.body)) if isinstance(fn.body[i],ast.Assign) and any(isinstance(t,ast.Name) and t.id=='optimizer' for t in fn.body[i].targets))
code=compile(ast.Module(body=fn.body[start:end+1],type_ignores=[]),str(p),'exec')
for mode in ['use_l1_regression','use_diffusion','use_flow_matching','discrete']:
 modules={k:torch.nn.Linear(2,2) for k in ['vla','action_head','noisy_action_projector','proprio_projector']}
 cfg=SimpleNamespace(use_l1_regression=mode=='use_l1_regression',use_diffusion=mode=='use_diffusion',use_flow_matching=mode=='use_flow_matching',use_proprio=True,learning_rate=1e-4)
 scope=dict(modules,cfg=cfg,AdamW=torch.optim.AdamW);exec(code,scope)
 actual=[p for g in scope['optimizer'].param_groups for p in g['params']]
 names=['vla','proprio_projector']+(['action_head'] if mode!='discrete' else [])+(['noisy_action_projector'] if mode in ['use_diffusion','use_flow_matching'] else [])
 expected={id(p) for k in names for p in modules[k].parameters()}
 assert {id(p) for p in actual}==expected and len(actual)==len(expected),mode
 loss=sum(m(torch.ones(1,2)).square().mean() for k,m in modules.items() if k in names)
 before={id(p):p.detach().clone() for p in actual};loss.backward();scope['optimizer'].step()
 assert all(not torch.equal(before[id(p)],p) for p in actual),mode
 print('PASS',mode,'optimizer coverage and parameter updates')

# Import-time code must never build the quadratic cart matrix. Function bodies
# are intentionally excluded: cart is computed only when that objective is used.
for statement in tree.body:
 if isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
  continue
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='context_adaptive_reweight' for n in ast.walk(statement))
weight_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='context_adaptive_reweight')
scope=dict(torch=torch,np=np,math=math)
exec(compile(ast.Module(body=[weight_fn],type_ignores=[]),str(p),'exec'),scope)
weight=scope['context_adaptive_reweight']
larger=weight(64,cart_p=.1)
for length in [1,2,7,16,31]:
 actual=weight(length,cart_p=.1)
 torch.testing.assert_close(actual,larger[:length,:length],rtol=0,atol=0)
 mask=torch.arange(length)[None,:]%3==0
 non_mask=~mask
 direct=non_mask.type_as(actual).matmul(actual).masked_fill(non_mask,0)
 reference=non_mask.type_as(actual).matmul(larger[:length,:length]).masked_fill(non_mask,0)
 torch.testing.assert_close(direct,reference,rtol=0,atol=0)
print('PASS cart matrix prefix and resulting weights exactly equivalent; no import-time allocation')
