"""Camera isolation and the new future supervision on real policy components."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from Roboactualizer.models.roboactualizer.backbone import VJEPABackbone
from Roboactualizer.models.roboactualizer.predictive_context import PredictiveContextPolicy


class Encoder(nn.Module):
    def forward(self, x):
        # Camera-global feature deliberately exposes accidental cross-view mixing.
        z=x.mean((1,2,3,4))[:,None,None]
        return z.expand(-1,8,16)


class TinyBackbone(nn.Module):
    embed_dim=16
    num_patches=8
    patch_grid=(2,4)
    num_cameras=2
    independent_views=True


class Tests(unittest.TestCase):
    def test_encoder_camera_isolation_and_order(self):
        b=VJEPABackbone.__new__(VJEPABackbone); nn.Module.__init__(b)
        b.encoder=Encoder();b.independent_views=True;b.num_cameras=2
        b.num_patches=8;b.embed_dim=16;b._dtype=torch.float32
        b.register_buffer('_mean',torch.zeros(1,3,1,1));b.register_buffer('_std',torch.ones(1,3,1,1))
        x=torch.zeros(2,4,3,4,8);a=b.encode_clip_last_group(x)
        x[:,:,:,:,4:]=1;b2=b.encode_clip_last_group(x)
        torch.testing.assert_close(a[:,:4],b2[:,:4],rtol=0,atol=0)
        self.assertGreater((a[:,4:]-b2[:,4:]).abs().max().item(),0)

    def test_l1_target_norm_mask_and_no_leak(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        m=PredictiveContextPolicy(TinyBackbone(),chunk_size=4,future_size=2,video_freq_ratio=2,
            action_dim=3,text_dim=12,proprio_dim=5,d_model=24,num_layers=3,num_heads=2,
            dim_head=12,mlp_dim=48,freq_dim=16,predictor_checkpointing=False,
            future_loss_type='l1',latent_normalization='target_layernorm')
        batch=dict(z0=torch.randn(2,8,16),z_future=torch.randn(2,2,8,16),
            action_chunk=torch.randn(2,4,3),proprio=torch.randn(2,5),text_context=torch.randn(2,6,12),
            text_mask=torch.ones(2,6,dtype=torch.bool),action_is_pad=torch.zeros(2,4,dtype=torch.bool),
            image_is_pad=torch.tensor([[False,False,True],[False,False,True]]))
        self.assertEqual(m.image_rope_freqs.shape,(24,1,6))
        torch.testing.assert_close(m.image_rope_freqs[:4],m.image_rope_freqs[4:8])
        n=torch.randn_like(batch['action_chunk']);s=torch.full((2,1,1),.5);t=torch.full((2,),500.)
        a=m.prediction_and_targets(batch,n,s,t)
        b=m.prediction_and_targets(dict(batch,z_future=batch['z_future']*3+5),n,s,t)
        torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
        torch.testing.assert_close(a[2],b[2],rtol=0,atol=0)
        torch.testing.assert_close(a[3],F.layer_norm(batch['z_future'],(16,)))
        with patch.object(m,'prediction_and_targets',return_value=a):
            loss,logs=m.training_loss(batch)
        torch.testing.assert_close(logs['image_loss'],(a[2][:,0]-a[3][:,0]).abs().mean())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(m.image_decoder.head.weight.grad)


if __name__=='__main__':unittest.main()
