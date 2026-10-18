#!/usr/bin/env python3
"""Replay a 36-column UltimateBots G1 motion CSV on Unitree's 29-DoF G1 MJCF.

This is kinematic pose playback (mj_forward), NOT a torque-controlled simulation.
Assumed format: root_xyz (3), root quaternion xyzw (4), 29 joints (radians)
in the MJCF's joint order. Validate the pose against your Studio preview.

Examples:
  python play_motion.py --preview
  python play_motion.py --output motion.mp4
  python play_motion.py --output motion.mp4 --azimuth 90 --distance 3.2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

G1_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("motion.csv"))
    parser.add_argument(
        "--model", type=Path,
        default=Path.home() / "robotics-internship/unitree_rl_gym/resources/robots/g1_description/g1_29dof.xml",
    )
    parser.add_argument("--fps", type=int, default=30,
                        help="CSV samples per second (verify with Studio; default 30)")
    parser.add_argument("--quat-order", choices=["xyzw", "wxyz"], default="xyzw",
                        help="CSV quaternion order; MuJoCo uses wxyz")
    parser.add_argument("--preview", action="store_true",
                        help="Show the motion in the MuJoCo viewer; do not save MP4")
    parser.add_argument("--output", type=Path, default=Path("motion.mp4"))
    parser.add_argument("--azimuth", type=float, default=90.0)
    parser.add_argument("--elevation", type=float, default=-13.0)
    parser.add_argument("--distance", type=float, default=2.8)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    frames = np.loadtxt(args.csv, delimiter=",", dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != 36:
        raise ValueError(f"Expected an Nx36 CSV without a header; received shape {frames.shape}")
    if not np.isfinite(frames).all():
        raise ValueError("CSV contains NaN or infinity")

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    if model.nq != 36 or model.nu != 29:
        raise ValueError(
            f"Wrong model: nq={model.nq}, nu={model.nu}. "
            "Use g1_29dof.xml, NOT the 12-DoF walking scene.xml."
        )
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(model.njnt)]
    expected = ["floating_base_joint", *G1_JOINTS]
    if names != expected:
        raise ValueError(
            "Joint ordering in this MJCF differs from the expected G1 layout. "
            "Inspect model joint names before mapping CSV values.\n"
            f"Found: {names}"
        )

    q = np.empty_like(frames)
    q[:, :3] = frames[:, :3]  # root translation: metres
    if args.quat_order == "xyzw":
        q[:, 3:7] = frames[:, [6, 3, 4, 5]]  # xyzw -> MuJoCo wxyz
    else:
        q[:, 3:7] = frames[:, 3:7]
    norm = np.linalg.norm(q[:, 3:7], axis=1)
    if (norm < 1e-6).any():
        raise ValueError("At least one invalid root quaternion")
    q[:, 3:7] /= norm[:, None]
    q[:, 7:] = frames[:, 7:]  # angles in radians, same order as G1_JOINTS

    print(f"CSV: {len(frames)} frames x {frames.shape[1]} columns")
    print(f"Duration: {len(frames) / args.fps:.2f} seconds at {args.fps} FPS")
    print(f"Model: nq={model.nq}, nu={model.nu}; 29 ordered joints verified")
    print(f"Quaternion: {args.quat_order} in CSV -> wxyz in MuJoCo")
    print("Mode: kinematic pose replay (no physics integration or balance control)")

    data = mujoco.MjData(model)

    def set_pose(index: int) -> None:
        data.qpos[:] = q[index]
        data.qvel[:] = 0.0
        data.time = index / args.fps
        mujoco.mj_forward(model, data)

    set_pose(0)
    if args.preview:
        print("Opening preview. Close the viewer window to exit.", flush=True)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                for index in range(len(q)):
                    if not viewer.is_running():
                        break
                    start = time.monotonic()
                    set_pose(index)
                    viewer.sync()
                    time.sleep(max(0, 1 / args.fps - (time.monotonic() - start)))
        return

    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401 - required MP4 encoder backend
    except ImportError as exc:
        raise SystemExit(
            "To save MP4, install: python -m pip install imageio imageio-ffmpeg"
        ) from exc

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [float(np.median(q[:, 0])), float(np.median(q[:, 1])), 0.85]
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        with imageio.get_writer(str(args.output), fps=args.fps, codec="libx264",
                                macro_block_size=2) as writer:
            for index in range(len(q)):
                set_pose(index)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
                if (index + 1) % 60 == 0:
                    print(f"Rendered {index + 1}/{len(q)} frames", flush=True)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
