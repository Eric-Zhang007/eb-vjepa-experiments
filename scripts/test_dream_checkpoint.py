"""Exercise the real official serializer with small modules and retention."""
import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import torch

root = Path('/data/users/jiachangzhang/eb_vjepa')
script = root/'repos/Dream-VLX/vla/vla-scripts/finetune_dream.py'
sys.path.insert(0,str(script.parent))
from checkpoint_retention import save_retained, should_save, publish

class Dist:
 def barrier(self): pass
 def all_reduce(self,x): x.mul_(2)  # two identical ranks, global count doubles
 def broadcast_object_list(self,x,src): pass
 def get_world_size(self): return 2

class Saved:
 def save_pretrained(self,path):
  Path(path).mkdir(parents=True,exist_ok=True)
  (Path(path)/'config.json').write_text('{}')
 @classmethod
 def register_for_auto_class(cls,name): pass

@dataclass
class Config:
 save_latest_checkpoint_only: bool=True
 use_film: bool=False
 use_lora: bool=True
 merge_lora_during_training: bool=False

tree=ast.parse(script.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_training_checkpoint')
fn.decorator_list=[]
for arg in fn.args.args: arg.annotation=None
fn.returns=None
scope=dict(Path=Path,os=os,torch=torch,dist=Dist(),DreamVLAForActionPrediction=Saved,DreamVLAConfig=Saved,
 save_dataset_statistics=lambda stats,path:(Path(path)/'dataset_statistics.json').write_text(json.dumps(stats)))
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(script),'exec'),scope)
save=scope['save_training_checkpoint']
assert [u for u in range(1,7) if should_save(u,2,5)][:3]==[2,4,5]
with tempfile.TemporaryDirectory(prefix='checkpoint-test-',dir=root/'cache/tmp') as temporary:
 run=Path(temporary)/'run'
 orphan=run/'checkpoints/.saving_2'
 orphan.mkdir(parents=True)
 model=Saved();model.config=Saved()
 head=torch.nn.Linear(2,2)
 kwargs=dict(vla=SimpleNamespace(module=model),processor=Saved(),proprio_projector=head,
  noisy_action_projector=head,action_head=head,train_dataset=SimpleNamespace(dataset_statistics={'test':1}))
 for step,total,count,start in [(2,8.,4.,1),(4,12.,4.,3),(5,2.,2.,5)]:
  metadata=save_retained(save,Config(),run,step,torch.tensor([total,count],dtype=torch.float64),start,
   scope['dist'],SimpleNamespace(is_main_process=True),**kwargs)
  assert orphan.is_dir(), 'interrupted staging must remain untouched'
  latest=run/'checkpoints/latest';best=run/'checkpoints/best'
  assert metadata['selection_loss']==total/count
  assert metadata['global_microbatch_count']==count*2 and metadata['world_size']==2
  assert json.loads((latest/'checkpoint_metadata.json').read_text())['optimizer_update']==step
  expected_best=2 if step==4 else step
  assert json.loads((best/'checkpoint_metadata.json').read_text())['optimizer_update']==expected_best
  loaded=torch.load(latest/'action_head--latest_checkpoint.pt',weights_only=True)
  for key,value in head.state_dict().items():torch.testing.assert_close(value,loaded[key])
  assert (latest/'lora_adapter/config.json').exists()
  assert len(list((run/'checkpoints').glob('generation_*')))==(2 if step==4 else 1)
  previous=latest.resolve()
  failed=run/'checkpoints/.failed';failed.mkdir()
  try:publish(failed,run/'checkpoints',{'optimizer_update':step+1,'selection_loss':float('nan')})
  except ValueError:pass
  else:raise AssertionError('nonfinite publication accepted')
  assert latest.resolve()==previous
  failed.rmdir()
print('PASS real official serializer; exact update schedule/final; global window mean; latest/best rotation; failed publication preserves previous')
