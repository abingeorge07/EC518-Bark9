import mujoco
import mujoco_viewer
import numpy as np

def run_simulation(xml_path):
  
    # PATRIQ with Slope
    model = mujoco.MjModel.from_xml_path('/projectnb/rlvn/students/rampal11/PATRIQ/EC518-Bark9/envs/slope.xml')

    # PATRIQ with obstcales
    # model = mujoco.MjModel.from_xml_path('/projectnb/rlvn/students/rampal11/PATRIQ/EC518-Bark9/envs/grids.xml')
    
    # PATRIQ on ICE
    # model = mujoco.MjModel.from_xml_path('/projectnb/rlvn/students/rampal11/PATRIQ/EC518-Bark9/envs/ice.xml')

    # PATRIQ with Stairs
    # model = mujoco.MjModel.from_xml_path('/projectnb/rlvn/students/rampal11/PATRIQ/EC518-Bark9/envs/stairs.xml')

    data = mujoco.MjData(model)

    # Create a viewer for rendering
    viewer = mujoco_viewer.MujocoViewer(model, data)

    try:
        print("Starting simulation...")
        for _ in range(1000):  # Run for 1000 timesteps
            # Generate random control signals for testing
            if model.nu > 0:  # If there are actuators
                data.ctrl[:] = np.random.uniform(-1, 1, size=model.nu)

            # Step the simulation
            mujoco.mj_step(model, data)

            # Render the environment
            viewer.render()
    except mujoco_viewer.viewer.ViewerExit:
        # Handle viewer exit gracefully
        print("Simulation interrupted by user.")
    finally:
        # Close the viewer properly
        viewer.close()

if __name__ == "__main__":
    # Specify the path to your XML file
    xml_path = "/projectnb/rlvn/students/rampal11/PATRIQ/EC518-Bark9/envs/slope.xml"  # Replace with your actual path
    run_simulation(xml_path)
