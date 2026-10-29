import unittest
import numpy as np
from elaf.evaluation.libero_env import make_manifest,convert_actions,horizon,observation

class EnvContracts(unittest.TestCase):
    def test_clean_protocol(self):
        entries=make_manifest('clean')
        self.assertEqual(len(entries),200)
        self.assertEqual(len({(e['suite'],e['task_id'],e['trial']) for e in entries}),200)
        self.assertEqual(horizon('libero_10_swap'),700)
        self.assertEqual(horizon('libero_goal_lan'),400)
    def test_gripper_conversion_nonmutating(self):
        a=np.zeros((2,7)); a[1,-1]=1
        b=convert_actions(a)
        self.assertEqual(b[:,-1].tolist(),[1,-1])
        self.assertEqual(a[:,-1].tolist(),[0,1])
    def test_rotation_and_proprio(self):
        image=np.arange(12,dtype=np.uint8).reshape(2,2,3)
        o=observation({'agentview_image':image,'robot0_eye_in_hand_image':image,'robot0_eef_pos':np.zeros(3),'robot0_eef_quat':np.array([0.,0.,0.,1.]),'robot0_gripper_qpos':np.zeros(2)},'task')
        np.testing.assert_array_equal(o['image'],image[::-1,::-1])
        self.assertEqual(o['proprio'].shape,(8,))

if __name__=='__main__': unittest.main()
