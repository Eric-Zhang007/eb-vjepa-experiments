"""Real optimizer interruption/restore equivalence using official state helpers."""
from pathlib import Path
import random
import sys
import tempfile
import numpy as np
import torch

ROOT=Path('/data/users/jiachangzhang/eb_vjepa')
sys.path.insert(0,str(ROOT/'repos/Dream-VLX/vla/vla-scripts'))
from resume_state import save_state,load_state,plain_values
import json
import json_numpy
json_numpy.patch()
assert json.dumps(plain_values({'x':np.array([1.,2.]),'y':np.array(3)}),sort_keys=True)==json.dumps({'x':[1.,2.],'y':3},sort_keys=True)

class Dataset:
 def __init__(self):self.index=0
 def next(self):
  self.index+=1
  return torch.tensor([[self.index/10,1.]])
 def save_iterator_state(self,path):Path(path).write_text(str(self.index))
 def restore_iterator_state(self,path):self.index=int(Path(path).read_text())

def make():
 model=torch.nn.Linear(2,2)
 model.register_buffer('updates',torch.zeros((),dtype=torch.long))
 optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
 scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[2],gamma=.1)
 return model,optimizer,scheduler,Dataset()

def update(model,optimizer,scheduler,dataset):
 noise=torch.rand(1,2)+random.random()+float(np.random.rand())
 loss=(model(dataset.next())-noise).square().mean()
 optimizer.zero_grad();loss.backward();optimizer.step();scheduler.step()
 model.updates.add_(1)
 return loss.detach().clone()

torch.manual_seed(19);np.random.seed(19);random.seed(19)
model,opt,sch,data=make()
update(model,opt,sch,data)
with tempfile.TemporaryDirectory(dir=ROOT/'cache/tmp',prefix='resume-test-') as path:
 contract={'world_size':1,'config':'fixed'}
 save_state(path,{'vla':model},opt,sch,0,{'completed_updates':1},contract,data)
 expected_loss=update(model,opt,sch,data)
 expected={k:v.detach().clone() for k,v in model.state_dict().items()}
 expected_opt=opt.state_dict()
 restored,opt2,sch2,data2=make()
 progress=load_state(path,{'vla':restored},opt2,sch2,0,contract,data2)
 actual_loss=update(restored,opt2,sch2,data2)
 torch.testing.assert_close(expected_loss,actual_loss,rtol=0,atol=0)
 for key,value in expected.items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
 assert sch.state_dict()==sch2.state_dict() and progress['completed_updates']==1 and data.index==data2.index
 for key,state in expected_opt['state'].items():
  for field,value in state.items():torch.testing.assert_close(value,opt2.state_dict()['state'][key][field],rtol=0,atol=0)
 try:load_state(path,{'vla':restored},opt2,sch2,0,{'world_size':2},data2)
 except ValueError:pass
 else:raise AssertionError('contract mismatch accepted')
print('PASS interrupted optimizer/scheduler/Python NumPy Torch RNG/cursor resume bitwise equal; changed contract rejected')
