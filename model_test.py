import mujoco
import mujoco_viewer

model_path = "models/google_barkour_v0/scene_mjx.xml"
patriq = "PATRIQ/patriq.xml"
example = "PATRIQ/example.xml"  
# patriq = "PATRIQ/example.xml"
# load the model and data
model = mujoco.MjModel.from_xml_path(patriq)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)


# render the model
viewer.render()

joint_goal = [0, 0.1, -0.1]
error = [0, 0, 0]

k_p = -0.1
k_p2 = 5.1

angle = 0.1
temp = angle

# for i in range(8, model.nq):
#     temp = angle
#     data.qpos[i] = angle


while viewer.is_alive:

    # data.qpos[8] = 0.6
    # data.qpos[9] = -0.6

    # step the simulation
    mujoco.mj_step(model, data)
    mujoco.mj_kinematics(model, data)
     
    # render the model
    viewer.render()

# HI ADI HOW ARE YOU??
