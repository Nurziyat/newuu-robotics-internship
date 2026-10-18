# Part B — Unitree G1 Circular Walking

## Objective

The objective is to run the pretrained Unitree G1 walking policy in MuJoCo and modify the walking command so that the robot follows a circular trajectory instead of walking straight.

## Modification

The original G1 configuration uses:

```yaml
cmd_init: [0.5, 0, 0]
```

I created a separate configuration file named `g1_circle.yaml` and changed the command to:

```yaml
cmd_init: [0.5, 0, 0.5]
```

The three values represent forward linear velocity, lateral linear velocity, and yaw angular velocity.

With forward velocity of 0.5 m/s and angular velocity of 0.5 rad/s, the ideal turning radius is:

\[
R = \frac{v_x}{|\omega_z|} = 1 \text{ metre}
\]

Combining forward movement with continuous rotation causes the robot to follow a circular path. The actual trajectory depends on the robot's dynamic response and tracking accuracy.

The pretrained policy was reused without retraining.

## What inputs does the policy receive?

The neural-network policy receives a 47-dimensional observation vector assembled by `deploy_mujoco.py` from the MuJoCo simulation, the configuration file, and the controller's previous output.

| Input | Dimensions | Source |
|---|---:|---|
| Base angular velocity | 3 | MuJoCo generalized velocity |
| Projected gravity | 3 | Computed from base orientation quaternion |
| Desired velocity command | 3 | `cmd_init`, scaled by `cmd_scale` |
| Joint position offsets | 12 | MuJoCo joint positions minus default angles |
| Joint velocities | 12 | MuJoCo joint velocities |
| Previous actions | 12 | Previous policy output |
| Gait phase (sine and cosine) | 2 | Periodic phase calculated from simulation time |
| **Total** | **47** | |

The observations are scaled as specified by the deployment configuration before being passed to the policy.

## What does the policy output?

The policy produces 12 action values, one for each controlled joint.

The actions are converted into target joint positions:

\[
q_{\mathrm{target}} = q_{\mathrm{default}} + \text{action_scale} \cdot a
\]

The resulting targets are passed into a proportional-derivative (PD) controller:

\[
\tau = K_p(q_{\mathrm{target}}-q) + K_d(\dot q_{\mathrm{target}}-\dot q)
\]

The deployment uses zero target joint velocities. The computed torques are assigned to `d.ctrl` and applied by MuJoCo during simulation.

The physics time step is 0.002 seconds, and the policy is evaluated every 10 simulation steps, corresponding to a 50 Hz policy update rate.

## Reproduction

Use the official [Unitree RL Gym repository](https://github.com/unitreerobotics/unitree_rl_gym).

With Python dependencies installed, copy `g1_circle.yaml` into `deploy/deploy_mujoco/configs/` inside the cloned repository.

From the repository root, run:

```bash
PYTHONPATH="$PWD" python deploy/deploy_mujoco/deploy_mujoco.py g1_circle.yaml
```

This starts the pretrained G1 walking policy in MuJoCo with the modified circular-walking command.

## Result

The simulated Unitree G1 successfully walked along a circular trajectory using the modified command.

The result is demonstrated in `g1_circle.mp4`.
