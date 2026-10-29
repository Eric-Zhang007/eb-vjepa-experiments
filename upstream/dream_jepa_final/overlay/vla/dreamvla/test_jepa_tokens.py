"""Run with python -m unittest dreamvla.test_jepa_tokens."""
import unittest
import torch
from .jepa_tokens import DualViewJEPATokens, joint_attention_mask


class TokenTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = DualViewJEPATokens(32, (2,4,4), 16, 16)
        self.x = torch.randn(2,32,16,requires_grad=True)

    def test_fixed_target_and_gradients(self):
        target = self.model.future_target(self.x)
        current, future = self.model(self.x, self.x + 1, torch.randn_like(target))
        self.assertEqual(current.shape, (2,32,32))
        (current.square().mean()+future.square().mean()).backward()
        self.assertIsNone(self.x.grad)
        self.assertGreater(self.model.query.weight.grad.norm(), 0)
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.randn_like(p))
        self.assertTrue(torch.equal(target, self.model.future_target(self.x)))

    def test_spatial_order(self):
        x = torch.arange(16).float().repeat(2).view(1,32,1).expand(1,32,16)
        self.assertTrue(torch.equal(self.model.future_target(x)[0,:,0], torch.arange(16).float()))
        self.assertEqual(torch.unique(self.model.position,dim=0).shape[0],16)

    def test_multilayer_information_isolation(self):
        valid = torch.ones(2,9,dtype=torch.bool)
        valid[:,2] = False
        mask = joint_attention_mask(valid,6,torch.tensor([True,False]))
        h = torch.randn(2,9,8)
        changed = h.clone(); changed[:,6:] += 5
        for _ in range(3):
            def step(x):
                return x + torch.nn.functional.scaled_dot_product_attention(x[:,None],x[:,None],x[:,None],attn_mask=mask)[:,0]
            h,changed=step(h),step(changed)
        self.assertTrue(torch.equal(h[0,:6],changed[0,:6]))
        self.assertFalse(torch.allclose(h[1,:6],changed[1,:6]))

    def test_real_dream_layers(self):
        from .configuration_dreamvl import DreamVLAConfig
        from .modeling_dreamvl import DreamVLAForActionPrediction
        from prismatic.models.action_heads import MLPResNet, SinusoidalPositionalEncoding
        cfg=DreamVLAConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,
            mrope_section=[1,1,2],pad_token_id=0,projector_hidden_act="gelu",n_action_bins=16,norm_stats={},
            vision_config=dict(depth=1,embed_dim=32,hidden_size=32,num_heads=4))
        cfg._attn_implementation='sdpa'
        cfg.vision_config._attn_implementation='sdpa'
        model=DreamVLAForActionPrediction(cfg).eval()
        model.jepa_tokens=self.model
        model.jepa_time=SinusoidalPositionalEncoding(32)
        model.jepa_future_head=MLPResNet(2,32,32,16)
        args=dict(input_ids=torch.ones(2,10,dtype=torch.long),attention_mask=torch.ones(2,10,dtype=torch.long),
                  return_dict=True,use_cache=False)
        baseline=model(**args).hidden_states
        future=torch.randn(2,16,16)
        kwargs=dict(jepa_primary=self.x,jepa_wrist=self.x+1,noisy_future=future,
                    future_time=torch.full((2,),.5),future_dropped=torch.tensor([True,False]))
        first=model(**args,**kwargs)
        kwargs['noisy_future']=future+10
        second=model(**args,**kwargs)
        self.assertEqual(first.hidden_states.shape,baseline.shape)
        self.assertTrue(torch.equal(first.hidden_states[0],second.hidden_states[0]))
        self.assertFalse(torch.allclose(first.hidden_states[1],second.hidden_states[1]))
        self.assertEqual(first.future_velocity.shape,(2,16,16))
        (first.hidden_states.square().mean()+first.future_velocity.square().mean()).backward()
        self.assertGreater(model.jepa_tokens.query.weight.grad.norm(),0)


if __name__=='__main__':
    unittest.main()
