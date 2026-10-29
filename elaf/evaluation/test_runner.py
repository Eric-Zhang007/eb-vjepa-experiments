"""Subprocess scheduling tests. Fake environments are explicitly unit-test fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from elaf.evaluation.runner import evaluate


def fake_env_worker(connection,config):
    try:
        while True:
            message=connection.recv()
            if message['command']=='close':
                connection.send({'ok':True}); return
            if message['command']=='reset':
                spec=message['spec']; steps=0
                if spec.get('error'):
                    connection.send({'ok':False,'error':'fixture simulator error'}); return
                connection.send({'ok':True,'observation':{'id':spec['task_id']},'steps':0,'done':False,'info':dict(horizon=3,settle_steps=30,image_rotation_degrees=180,gripper_convention='open_01')})
            else:
                steps+=1
                connection.send({'ok':True,'observation':{'id':spec['task_id']},'steps':1,
                    'done':steps>=spec['length'],'success':spec['task_id']%2==0,'info':dict(horizon=3)})
    finally:
        connection.close()


class FakePolicy:
    metadata={'unit_test':True}
    def request(self,request):
        return {'ok':True,'actions':np.zeros((len(request['observations']),2,7)), 'inference_seconds':.001}


class RunnerTests(unittest.TestCase):
    def spec(self,index,length=1):
        return {'benchmark':'clean','suite':'libero_spatial','task_id':index,'trial':0,'seed':239+index,'length':length}

    def test_parallel_complete_exact_episode_coverage(self):
        specs=[self.spec(i,i%3+1) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp, patch('elaf.evaluation.libero_env.run_env_worker',fake_env_worker):
            out=Path(tmp)/'run'
            result=evaluate(specs,{'execute_steps':2},FakePolicy(),out,environments=3,batch_size=2,timeout=15)
            self.assertEqual(result['episodes'],5)
            self.assertEqual(result['successes'],3)
            self.assertEqual(result['control_steps'],sum(s['length'] for s in specs))
            rows=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()]
            self.assertEqual(sorted(r['spec']['task_id'] for r in rows),list(range(5)))
            self.assertEqual(rows[0]['environment_protocol'],dict(horizon=3,settle_steps=30,image_rotation_degrees=180,gripper_convention='open_01',execute_steps=2))
            self.assertTrue((out/'summary.json').exists())
            self.assertFalse((out/'error.json').exists())

    def test_worker_error_never_publishes_success_summary(self):
        spec=self.spec(0); spec['error']=True
        with tempfile.TemporaryDirectory() as tmp, patch('elaf.evaluation.libero_env.run_env_worker',fake_env_worker):
            out=Path(tmp)/'run'
            with self.assertRaisesRegex(RuntimeError,'fixture simulator error'):
                evaluate([spec],{},FakePolicy(),out,environments=1,batch_size=1,timeout=15)
            self.assertFalse((out/'summary.json').exists())
            self.assertEqual(json.loads((out/'error.json').read_text())['completed_episodes'],0)

    def test_exact_duplicate_rejected_before_launch(self):
        spec=self.spec(0)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            with self.assertRaises(ValueError):
                evaluate([spec,spec],{},FakePolicy(),out,environments=1,batch_size=1)
            self.assertFalse(out.exists())

    def test_duplicate_with_changed_metadata_rejected(self):
        spec=self.spec(0)
        other=dict(spec, seed=999, note='different metadata, same episode')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                evaluate([spec,other],{},FakePolicy(),Path(tmp)/'run',environments=2,batch_size=2)


if __name__=='__main__':
    unittest.main()
