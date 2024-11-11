import mujoco
import mujoco_viewer

model_path = "models/google_barkour_v0/scene_mjx.xml"

# load the model and data
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)


# render the model
viewer.render()


while viewer.is_alive:

    # step the simulation
    mujoco.mj_step(model, data)
    mujoco.mj_kinematics(model, data)
     
    # render the model
    viewer.render()