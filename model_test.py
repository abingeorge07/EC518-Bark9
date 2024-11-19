import mujoco
import mujoco_viewer
import cv2
import numpy as np

# function to render the window (pov)
def windowView(model, data, opt, camera, scene, context, window):
    # Render the scene
    viewport = mujoco.MjrRect(0, 0, 1200, 900)

    # make the current window context 
    mujoco.glfw.glfw.make_context_current(window)
    mujoco.glfw.glfw.swap_interval(1)

    # update the scene & render
    mujoco.mjv_updateScene(model, data, opt, None, camera, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)

    # Update the window
    mujoco.glfw.glfw.swap_buffers(window)
    mujoco.glfw.glfw.poll_events()


# Paths to models
patriq = "PATRIQ/patriq.xml"
example_quad = "models/google_barkour_v0/scene_mjx.xml"

# Load the model and data
model = mujoco.MjModel.from_xml_path(patriq)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)

opt = mujoco.MjvOption()


# Set the camera for the renderer
camera_name = "main_camera"
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    
if(camera_id < 0):
    # Set the camera for the viewer (onboard)
    camera = mujoco.MjvCamera()
    pov_working = False
else:
    # Set the camera for the viewer (onboard)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = camera_id
    pov_working = True

# Top View Camera
camera_top = mujoco.MjvCamera()



# Initialize the viewer
mujoco.glfw.glfw.init()


if(pov_working):
    # Create a window
    window = mujoco.glfw.glfw.create_window(1200, 900, "POV", None, None)
    # make the context current
    mujoco.glfw.glfw.make_context_current(window)
    mujoco.glfw.glfw.swap_interval(1)
    # Create a scene and context
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)


window_top = mujoco.glfw.glfw.create_window(1200, 900, "Top-Down", None, None)
mujoco.glfw.glfw.make_context_current(window_top)
mujoco.glfw.glfw.swap_interval(1)

scene_top = mujoco.MjvScene(model, maxgeom=10000)
context_top = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)


while not mujoco.glfw.glfw.window_should_close(window_top):
    simstart = data.time

    # data.ctrl[0] = 0.1
    data.ctrl[1] = 0.1
    data.ctrl[2] = 0.1

    while (data.time - simstart < 1.0/60.0):
        mujoco.mj_step(model, data)

    ## WINDOW 1 ##
    if pov_working:
        windowView(model, data, opt, camera, scene, context, window)
  
    ## WINDOW 2 ##
    windowView(model, data, opt, camera_top, scene_top, context_top, window_top)



mujoco.glfw.glfw.terminate()
