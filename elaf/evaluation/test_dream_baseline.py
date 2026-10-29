"""Baseline preprocessing and identity checks without loading a second GPU model."""
import ast
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import numpy as np
from PIL import Image
import torch
from elaf.evaluation.policy import dream_eval_image, DreamVLABaselinePolicy

class DreamBaselineEvalTests(unittest.TestCase):
    def test_preprocessing_matches_official_source(self):
        import tensorflow as tf
        source=Path('/data/users/jiachangzhang/eb_vjepa/repos/Dream-VLX/vla/experiments/robot/openvla_utils.py')
        tree=ast.parse(source.read_text())
        names={'resize_image_for_policy','crop_and_resize','center_crop_image'}
        nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
        for node in nodes:
            node.returns=None
            for arg in node.args.args:arg.annotation=None
        namespace=dict(tf=tf,np=np,Image=Image,OPENVLA_IMAGE_SIZE=224)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
        image=np.random.default_rng(1).integers(0,256,(256,256,3),dtype=np.uint8)
        resized=namespace['resize_image_for_policy'](image,224)
        reference=np.asarray(namespace['center_crop_image'](Image.fromarray(resized)))
        np.testing.assert_array_equal(np.asarray(dream_eval_image(image)),reference)
        np.testing.assert_array_equal(np.asarray(dream_eval_image(image,False)),resized)

    def test_official_suite_normalization_is_explicit(self):
        from elaf.evaluation.official_dream import norm_key
        stats={f'libero_{s}_no_noops':{} for s in ['10','goal','object','spatial']}
        mapping={f'libero_{s}':f'libero_{s}_no_noops' for s in ['10','goal','object','spatial']}
        for suite,key in mapping.items():self.assertEqual(norm_key(suite,stats,mapping),key)
        with self.assertRaisesRegex(ValueError,'Explicit suite'):norm_key('foreign',stats,mapping)
        with self.assertRaisesRegex(ValueError,'no normalization'):norm_key('libero_10',{},mapping)

    def test_family_mismatch_fails_before_weight_load(self):
        with TemporaryDirectory() as tmp:
            Path(tmp,'training_contract.json').write_text(json.dumps({'model_family':'elaf','config':{}}))
            with self.assertRaisesRegex(ValueError,'baseline training contract'):
                DreamVLABaselinePolicy({'checkpoint':tmp})

if __name__=='__main__':unittest.main()
