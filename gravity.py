import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("PModel/PModel.urdf")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

print("gravity (world):", np.asarray(m.opt.gravity))

g = np.asarray(m.opt.gravity, dtype=np.float64)
ghat = g / (np.linalg.norm(g) + 1e-12)

for j in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE:
        continue
    bid = int(m.jnt_bodyid[j])
    axis_local = np.asarray(m.jnt_axis[j], dtype=np.float64).reshape(3)
    R = d.xmat[bid].reshape(3, 3)
    axis_world = R @ axis_local
    axis_world /= np.linalg.norm(axis_world) + 1e-12
    print(
        f"{name}: slide_axis_world={np.round(axis_world, 6)} | "
        f"dot(axis, gravity_hat)={np.dot(axis_world, ghat):+.8f}"
    )
