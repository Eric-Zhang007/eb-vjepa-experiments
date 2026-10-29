"""CPU tests for clone-to-workspace reconstruction, not model-quality tests."""
import importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('handoff',ROOT/'handoff.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

class HandoffTests(unittest.TestCase):
    def test_path_mapping_specific_before_root(self):
        cfg=dict(workspace='/new/work',data_root='/shared/rlds')
        value=h.OLD+'/repos/x /data/users/jiachangzhang/DNFT4RP/upstream/vla_adapter/datasets/rlds /home/jiachangzhang/dream_jepa_dualview_20261023'
        self.assertEqual(h.relocate(value,cfg),'/new/work/repos/x /shared/rlds /new/work/runs/dream_jepa')

    def test_policy_configs_keep_protocol_and_split_sources(self):
        cfg=dict(workspace='/new/work',data_root='/shared/rlds')
        a=h.policy_config(cfg,'baseline');b=h.policy_config(cfg,'dream-jepa')
        self.assertEqual(a['num_steps'],4);self.assertEqual(a['suite_norm_map'],b['suite_norm_map'])
        self.assertNotEqual(a['repo'],b['repo']);self.assertEqual(b['type'],'dream_jepa_dualview')
        self.assertIn('011000',b['encoder_checkpoint']);self.assertFalse(b['language_subgoals'])

    def test_local_git_checkout_patch_overlay_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);repo=base/'upstream';repo.mkdir();bundle=base/'bundle';workspace=base/'work'
            def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.STDOUT,text=True)
            git('init');git('config','core.autocrlf','false');git('config','user.email','test@example.invalid');git('config','user.name','Test')
            (repo/'model.py').write_bytes(b'answer = 1\n');git('add','.');git('commit','-m','base');commit=git('rev-parse','HEAD').strip()
            (repo/'model.py').write_bytes(b'answer = 2\n')
            directory=bundle/'upstream/dream_vla_baseline';(directory/'overlay').mkdir(parents=True)
            (directory/'changes.patch').write_bytes(git('diff','HEAD').encode());(directory/'overlay/added.py').write_text('root = '+repr(h.OLD)+'\n')
            item=dict(name='dream_vla_baseline',origin=str(repo),commit=commit)
            cfg=dict(workspace=str(workspace),data_root=str(base/'data'))
            with patch.object(h,'REPO',bundle):
                h.materialize_one(item,cfg)
                target=workspace/h.DESTINATIONS['dream_vla_baseline']
                self.assertEqual((target/'model.py').read_text(),'answer = 2\n')
                self.assertIn(workspace.as_posix(),(target/'added.py').read_text())
                h.materialize_one(item,cfg)
                (directory/'overlay/added.py').write_text('changed=True\n')
                with self.assertRaisesRegex(RuntimeError,'recipe changed'):h.materialize_one(item,cfg)

    def test_fresh_training_command_uses_actual_jepa_flag(self):
        from argparse import Namespace
        cfg=dict(workspace='/new/work',data_root='/shared/rlds')
        args=Namespace(model='dream-jepa',gpus='0,1',steps=2,wandb_mode='offline',execute=True)
        with patch.object(h,'run') as run:
            h.train(cfg,args)
        cmd=list(map(str,run.call_args.args[0]));self.assertIn('--jepa_checkpoint',cmd);self.assertNotIn('--resume_checkpoint',cmd)
        self.assertEqual(cmd[cmd.index('--nproc_per_node')+1],'2')

if __name__=='__main__':unittest.main()
