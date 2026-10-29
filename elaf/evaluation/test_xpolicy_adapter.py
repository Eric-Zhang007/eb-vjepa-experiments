import unittest
import numpy as np
from elaf.evaluation.xpolicy_adapter import Model


class FakeClient:
    metadata = {'action_dim': 14, 'action_horizon': 2, 'action_space': 'xpolicy_joint_absolute', 'proprio_dim': 14}
    def predict(self, observations, episode_keys=None):
        self.last = observations, episode_keys
        return np.tile(np.arange(14, dtype=np.float32), (len(observations), 2, 1))


def config():
    return dict(action_type='joint', action_space='xpolicy_joint_absolute', normalization='service_physical', arm_dims=[6,6], gripper_dims=[1,1], camera='cam_head', state_fields=[{'key':'left_arm_joint_state','dim':6},{'key':'left_ee_joint_state','dim':1},{'key':'right_arm_joint_state','dim':6},{'key':'right_ee_joint_state','dim':1}])


def observation(i):
    return dict(env_idx=i, vision={'cam_head':{'color':np.arange(18,dtype=np.uint8).reshape(2,3,3)}}, state={'left_arm_joint_state':[0]*6,'left_ee_joint_state':[1],'right_arm_joint_state':[2]*6,'right_ee_joint_state':[3]}, instruction='stack bowls')


class TestAdapter(unittest.TestCase):
    def test_batch_order_native_action_and_rgb(self):
        c=FakeClient(); m=Model(config(),c)
        m.update_obs_batch([observation(7),observation(2)])
        out=m.get_action_batch([2,7])
        self.assertTrue(c.last[1][0].endswith(':2'))
        np.testing.assert_array_equal(c.last[0][0]['image'],observation(2)['vision']['cam_head']['color'])
        np.testing.assert_array_equal(out[0][0]['right_arm_joint_state'],np.arange(7,13))
        self.assertEqual(out[0][0]['right_ee_joint_state'].tolist(),[13])
        old=c.last[1][0];m.reset();m.update_obs(observation(2));m.get_action()
        self.assertNotEqual(old,c.last[1][0])

    def test_reject_libero_model_and_semantics(self):
        c=FakeClient();c.metadata=dict(c.metadata,action_dim=7,action_space='libero_delta_ee')
        with self.assertRaisesRegex(ValueError,'incompatible'):Model(config(),c)
        cfg=config();cfg['action_space']='libero_delta_ee'
        with self.assertRaises(ValueError):Model(cfg,FakeClient())

    def test_missing_env_and_bad_camera(self):
        m=Model(config(),FakeClient());m.update_obs(observation(0))
        with self.assertRaisesRegex(ValueError,'Missing observations'):m.get_action_batch([4])
        o=observation(0);o['vision']['cam_head']['color']=np.zeros((2,3,3),dtype=float)
        with self.assertRaisesRegex(ValueError,'RGB'):m.update_obs(o)

    def test_seed_independent_of_batch_order_and_repeatable(self):
        a,b=FakeClient(),FakeClient();ma,mb=Model(config(),a),Model(config(),b)
        for m in [ma,mb]:m.update_obs_batch([observation(2),observation(7)])
        ma.get_action_batch([2,7]);mb.get_action_batch([7,2])
        self.assertEqual(a.last[0][0]['noise_seed'],b.last[0][1]['noise_seed'])
        first=a.last[0][0]['noise_seed']
        self.assertTrue(isinstance(first,int) and 0<=first<2**63)
        ma.get_action_batch([2]);self.assertNotEqual(first,a.last[0][0]['noise_seed'])
        ma.reset();self.assertEqual(ma.replan_counts,{})

    def test_nonfinite_actions_rejected(self):
        c=FakeClient();c.predict=lambda *a,**kw:np.full((1,2,14),np.nan)
        m=Model(config(),c);m.update_obs(observation(0))
        with self.assertRaisesRegex(ValueError,'Invalid action'):m.get_action()

    def test_explicit_flip(self):
        cfg=config();cfg['flip_vertical']=True;c=FakeClient();m=Model(cfg,c)
        m.update_obs(observation(0));m.get_action()
        np.testing.assert_array_equal(c.last[0][0]['image'],observation(0)['vision']['cam_head']['color'][::-1])

if __name__=='__main__':unittest.main()
