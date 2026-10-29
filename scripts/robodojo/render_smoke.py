"""Basic Isaac Sim renderer smoke, not a task or policy benchmark."""
from pathlib import Path
import json
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'width': 64, 'height': 64})
try:
    import omni.replicator.core as rep
    camera = rep.create.camera(position=(2, 2, 2), look_at=(0, 0, 0))
    rep.create.cube(position=(0, 0, 0))
    rep.create.light(light_type='Dome', intensity=1000)
    product = rep.create.render_product(camera, (64, 64))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach(product)
    rep.orchestrator.step(rt_subframes=4)
    frame = rgb.get_data()
    assert frame.size > 0 and frame.shape[:2] == (64, 64), frame.shape
    out = Path('/data/users/jiachangzhang/eb_vjepa/experiments/robodojo_setup/render_smoke.json')
    out.write_text(json.dumps({'shape': list(frame.shape), 'status': 'passed', 'scope': 'basic RGB render; no policy/task rollout'}))
finally:
    app.close()
