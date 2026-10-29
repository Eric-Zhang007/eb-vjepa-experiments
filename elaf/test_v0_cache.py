import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from elaf.v0_cache import ENCODER_SHA,normalization,sha256
from elaf.v0_dataset import CachedV0Dataset,collate_v0

class CacheTests(unittest.TestCase):
    def test_bounds_mask_and_constant_dimensions(self):
        stat=dict(q01=[0,0,2],q99=[2,1,2],min=[0,0,2],max=[2,1,2],mask=[True,False,True])
        np.testing.assert_array_equal(normalization([[3,.7,2]],stat),np.array([[1,.7,0]],np.float32))

    def test_episode_tails_and_bfloat_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episodes=[]
            for suite in ('spatial','10'):
                p=root/suite;p.mkdir();n=11
                feature=torch.arange(n*4*3).reshape(n,4,3).to(torch.bfloat16)
                np.save(p/'features.npy',feature.view(torch.uint16).numpy())
                for name,array in {'actions':np.ones((n,7),np.float32),'proprio':np.zeros((n,8),np.float32),
                    'images':np.zeros((n,2,2,3),np.uint8),'wrist_images':np.zeros((n,2,2,3),np.uint8)}.items():
                    np.save(p/(name+'.npy'),array)
                episodes.append(dict(suite=suite,path=suite,episode_id=suite,length=n,num_actions=n))
            (root/'stats.json').write_text('{}')
            manifest=dict(complete=True,encoder_sha256=ENCODER_SHA,stats_path='stats.json',
                stats_sha256=sha256(root/'stats.json'),episodes=episodes)
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            data=CachedV0Dataset(path)
            self.assertEqual(len(data),22);self.assertEqual(len(data.suite_indices['10']),11)
            self.assertTrue(torch.equal(data[0]['future_features'],feature[8]))
            self.assertTrue(data[2]['future_valid']);self.assertFalse(data[3]['future_valid'])
            tail=data[10];self.assertEqual(tail['action_valid'].sum(),1)
            self.assertEqual(tail['temporal_pair']['future_end'],10)
            batch=collate_v0([data[0],data[11]])
            self.assertEqual(batch['current_features'].shape,(2,4,3))
            manifest['complete']=False;path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):CachedV0Dataset(path)

if __name__=='__main__':unittest.main()
