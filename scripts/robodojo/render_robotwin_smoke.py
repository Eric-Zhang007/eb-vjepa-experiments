"""Render one SAPIEN RGB frame; no policy or benchmark task is executed."""
import json
from pathlib import Path
import numpy as np
import sapien

scene=sapien.Scene()
scene.set_timestep(1/240)
scene.add_ground(altitude=0)
scene.set_ambient_light([0.5,0.5,0.5])
scene.add_directional_light([0,-1,-1],[1,1,1])
builder=scene.create_actor_builder()
builder.add_box_visual(half_size=[0.1,0.1,0.1],material=[1,0,0])
actor=builder.build_static(name='smoke_cube')
actor.set_pose(sapien.Pose([0,0,0.1]))
camera=scene.add_camera(name='smoke',width=64,height=64,fovy=1.0,near=0.01,far=10)
camera.set_pose(sapien.Pose([-0.7,0,0.4],[0.9659258,0,0.258819,0]))
scene.step();scene.update_render();camera.take_picture()
color=camera.get_picture('Color')
assert color.shape==(64,64,4) and np.isfinite(color).all(),color.shape
assert np.ptp(color[...,:3])>0.01,'Rendered image lacks variation'
result={'status':'passed','shape':list(color.shape),'rgb_min':float(color[...,:3].min()),'rgb_max':float(color[...,:3].max()),'scope':'one basic SAPIEN scene; no RoboTwin task rollout'}
Path('/data/users/jiachangzhang/eb_vjepa/experiments/robodojo_setup/robotwin_render_smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
