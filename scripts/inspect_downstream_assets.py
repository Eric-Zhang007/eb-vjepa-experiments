"""Read-only inventory and optional CPU LIBERO simulator smoke test."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-smoke", action="store_true")
    parser.add_argument("--physics-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--suite", default="libero_spatial")
    args = parser.parse_args()
    report = {"python": os.sys.version, "packages": {}}
    for package in ("mujoco", "robosuite", "libero", "sapien", "torch"):
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["packages"][package] = None
    root = Path("/data/users/jiachangzhang/DNFT4RP/upstream/libero/libero/datasets")
    report["hdf5_counts"] = {p.name: len(list(p.glob("*.hdf5")))
                            for p in root.iterdir() if p.is_dir()}
    print(json.dumps(report, indent=2), flush=True)
    if args.sim_smoke:
        import numpy as np
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.env_wrapper import ControlEnv

        report["benchmark_module"] = benchmark.__file__
        report["suite"] = args.suite
        suite = benchmark.get_benchmark_dict()[args.suite]()
        report["suite_tasks"] = suite.n_tasks
        task = suite.get_task(0)
        factory = ControlEnv if args.physics_only else OffScreenRenderEnv
        extra = dict(has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False) if args.physics_only else {}
        env = factory(
            bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                        task.problem_folder, task.bddl_file),
            camera_heights=64, camera_widths=64, **extra,
        )
        try:
            env.seed(239)
            env.reset()
            init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            # Trusted local LIBERO states contain numpy arrays, not model weights.
            init_states = torch.load(init_path, map_location="cpu", weights_only=False)
            env.set_init_state(init_states[0])
            for _ in range(3):
                obs, reward, done, info = env.step(np.zeros(7))
            result = {"sim_smoke": "PASS", "physics_only": args.physics_only,
                      "task": task.language, "zero_action_steps": 3,
                      "init_states_path": str(init_path)}
            result["state_shapes"] = {}
            for key, shape in (("robot0_eef_pos", (3,)),
                               ("robot0_eef_quat", (4,)),
                               ("robot0_gripper_qpos", (2,))):
                state = np.asarray(obs[key])
                assert state.shape == shape and np.isfinite(state).all(), key
                result["state_shapes"][key] = list(state.shape)
            if not args.physics_only:
                result["cameras"] = {}
                for key in ("agentview_image", "robot0_eye_in_hand_image"):
                    image = obs[key]
                    assert image.shape == (64, 64, 3) and image.std() > 0, key
                    result["cameras"][key] = {
                        "shape": list(image.shape), "std": float(image.std())}
            print(json.dumps(result), flush=True)
            report.update(result)
        finally:
            env.close()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
