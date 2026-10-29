"""Match a real converted episode to original HDF5 actions, without simulation."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hdf5-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    if args.check_images and not args.replay:
        parser.error("--check-images requires --replay")
    path = args.dataset / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(path, columns=["action", "observation.state", "task_index"], use_threads=False)
    actions = np.array(table["action"].to_pylist())
    states = np.array(table["observation.state"].to_pylist())
    task_id = table["task_index"][0].as_py()
    tasks = [json.loads(line) for line in (args.dataset / "meta/tasks.jsonl").read_text().splitlines()]
    task = next(item["task"] for item in tasks if item["task_index"] == task_id)
    source = args.hdf5_suite / (task.replace(" ", "_") + "_demo.hdf5")
    report = {"parquet": str(path), "hdf5": str(source), "task": task, "candidates": []}
    with h5py.File(source, "r") as h5:
        for name, demo in h5["data"].items():
            raw = demo["actions"][:]
            distance = np.max(np.abs(actions[:, None, :6] - raw[None, :, :6]), axis=-1)
            indices = distance.argmin(axis=1)
            error = float(distance[np.arange(len(actions)), indices].max())
            if error > 1e-5:
                continue
            obs = demo["obs"]
            candidate = {"demo": name, "motion_max_error": error,
                         "ordered_unique_matches": bool(np.all(np.diff(indices) > 0)),
                         "obs_keys": list(obs),
                         "gripper_inverse_error": float(np.max(np.abs(actions[:, -1] - (1 - raw[indices, -1]) / 2))),
                         "gripper_direct_error": float(np.max(np.abs(actions[:, -1] - (1 + raw[indices, -1]) / 2)))}
            if "gripper_states" in obs:
                candidate["gripper_state_max_error"] = float(np.max(np.abs(states[:, -2:] - obs["gripper_states"][:][indices])))
            if "ee_states" in obs:
                ee = obs["ee_states"][:][indices]
                candidate["ee_state_shape"] = list(ee.shape)
                if ee.shape[-1] == 6:
                    candidate["ee_state_max_error"] = float(np.max(np.abs(states[:, :6] - ee)))
                    candidate["state_shift_checks"] = []
                    for shift in (-2, -1, 0, 1, 2):
                        shifted = indices + shift
                        valid = (shifted >= 0) & (shifted < len(raw))
                        original = np.concatenate((obs["ee_states"][:], obs["gripper_states"][:]), axis=-1)
                        delta = np.abs(states[valid] - original[shifted[valid]])
                        candidate["state_shift_checks"].append({"shift": shift,
                            "max_error": float(delta.max()), "mean_error": float(delta.mean()),
                            "per_dim_max_error": delta.max(axis=0).tolist()})
            report["candidates"].append(candidate)
            if args.replay:
                from libero.libero import benchmark, get_libero_path
                from libero.libero.envs.env_wrapper import ControlEnv
                from robosuite.utils.transform_utils import quat2axisangle

                suite = benchmark.get_benchmark_dict()["libero_spatial"]()
                task_spec = next(suite.get_task(i) for i in range(suite.n_tasks)
                                 if suite.get_task(i).language == task)
                env = ControlEnv(
                    bddl_file_name=str(Path(get_libero_path("bddl_files")) /
                                       task_spec.problem_folder / task_spec.bddl_file),
                    has_renderer=False, has_offscreen_renderer=args.check_images,
                    use_camera_obs=args.check_images, camera_heights=256, camera_widths=256)
                try:
                    env.seed(0)
                    env.reset()
                    env.set_init_state(demo["states"][0])
                    for _ in range(10):
                        obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
                    recorded, kept = [], []
                    for action in raw:
                        noop = np.linalg.norm(action[:-1]) < 1e-4
                        if noop and (not kept or action[-1] == kept[-1][-1]):
                            continue
                        if args.check_images and not kept:
                            import av
                            candidate["first_frame_camera_checks"] = {}
                            for camera, obs_key in (("image", "agentview_image"),
                                                    ("wrist_image", "robot0_eye_in_hand_image")):
                                video = args.dataset / "videos/chunk-000" / f"observation.images.{camera}" / "episode_000000.mp4"
                                with av.open(str(video)) as container:
                                    reference = next(container.decode(video=0)).to_ndarray(format="rgb24")
                                rendered = obs[obs_key]
                                assert rendered.shape == reference.shape == (256, 256, 3)
                                variants = {"unchanged": rendered, "rotate_180": rendered[::-1, ::-1],
                                            "flip_vertical": rendered[::-1], "flip_horizontal": rendered[:, ::-1]}
                                errors = {key: float(np.abs(value.astype(float) - reference).mean())
                                          for key, value in variants.items()}
                                candidate["first_frame_camera_checks"][camera] = {
                                    "mae_0_255": errors, "best_orientation": min(errors, key=errors.get)}
                        recorded.append(np.concatenate((obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])))
                        kept.append(action)
                        obs, _, _, _ = env.step(action.tolist())
                    kept = np.asarray(kept)
                    assert kept.shape == actions.shape, (kept.shape, actions.shape)
                    np.testing.assert_allclose(kept[:, :6], actions[:, :6], atol=1e-6)
                    delta = np.abs(np.asarray(recorded) - states)
                    candidate["cpu_replay"] = {"frames": len(recorded),
                        "max_state_error": float(delta.max()), "mean_state_error": float(delta.mean()),
                        "per_dim_max_error": delta.max(axis=0).tolist(),
                        "matches_at_1e_5": bool(delta.max() < 1e-5)}
                finally:
                    env.close()
    report["scope"] = "Single converted episode; report only, not full dataset provenance validation."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["candidates"]:
        raise RuntimeError("No exact original demonstration match; conventions remain unverified")


if __name__ == "__main__":
    main()
