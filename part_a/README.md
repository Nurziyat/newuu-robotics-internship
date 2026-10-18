# Part A — Human Motion to Unitree G1

## Task and result

I generated a 10-second **light bow and wave** motion with UltimateBots Studio, exported the numerical motion as `motion.csv`, and replayed it on the 29-DOF Unitree G1 model in MuJoCo with `play_motion.py`. The MuJoCo-rendered output is `motion.mp4`.

**Source prompt:** `A person performing a light bow and waving hello`

The original prompt is also saved in `prompt.txt`.

## How I identified the CSV columns

The exported CSV has **300 rows and 36 numeric columns**, without a header. Inspecting the file and comparing it with the MuJoCo `g1_29dof.xml` model led to the following mapping (columns are numbered starting at 1):

| Columns | Interpretation | Values |
| --- | --- | --- |
| 1–3 | Floating-base/root translation | x, y, z (metres) |
| 4–7 | Floating-base orientation | Quaternion x, y, z, w |
| 8–36 | Actuated joint positions | 29 joint angles (radians) |

This interpretation is supported by: (1) the 36-element MuJoCo `qpos` vector, with seven floating-base coordinates plus 29 scalar joints; (2) the approximately unit-length four-element orientation values; (3) the joint names and ordering checked against the MJCF file; and (4) a visual playback comparison with the generated bow-and-wave motion. The CSV has no column headers, so this mapping is a **validated interpretation for this motion and model**, not information embedded in the CSV itself.

MuJoCo stores free-joint quaternions in **w, x, y, z** order, so the script reorders the CSV quaternion from **x, y, z, w** before assigning it to `qpos`.

## Reproduce the result

1. Clone the [Unitree RL Gym repository](https://github.com/unitreerobotics/unitree_rl_gym) and make sure `resources/robots/g1_description/g1_29dof.xml` and its referenced mesh assets are available.
2. Use Python 3.10 with `mujoco==3.2.3`, `numpy==1.26.4`, `imageio`, and `imageio-ffmpeg` installed.
3. Put `motion.csv`, `play_motion.py`, and `prompt.txt` in the same folder.
4. Run from that folder:

```bash
python play_motion.py --preview
python play_motion.py --output motion.mp4
```

By default the script looks for the G1 model in `~/robotics-internship/unitree_rl_gym/resources/robots/g1_description/g1_29dof.xml`. For a different checkout location, specify `--model /path/to/g1_29dof.xml` in both commands.

The script interprets the motion as **30 frames per second**, producing a 10-second video. Its `--fps` option can be changed if the source frame rate differs.

## Playback method and limitations

The script assigns each frame's 36 pose values directly to MuJoCo `qpos` and calls `mujoco.mj_forward`, then renders the scene. This is **kinematic pose playback**, not a trained motion-tracking controller or a dynamically stabilized simulation. It demonstrates the requested robot motion visually, but does not establish that a physical G1 could balance while executing the exact motion.

## Included files

- `motion.csv` — original LAFAN CSV export
- `prompt.txt` — original text prompt used for generation
- `play_motion.py` — Python playback and MP4-rendering script
- `motion.mp4` — rendered 10-second G1 bow-and-wave video
