"""
Running the joint controller with an inverse kinematics path planner
for a Mujoco simulation. The path planning system will generate
a trajectory in joint space that moves the end effector in a straight line
to the target, which changes every n time steps.
"""
import numpy as np

from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.controllers import path_planners
from abr_control.interfaces.mujoco import AbrMujoco
from abr_control.utils import transformations

from main_window import MainWindow

# initialize our robot config for the jaco2
robot_config = arm("ur5", use_sim_state=False)

# create our path planner
n_timesteps = 2000
path_planner = path_planners.InverseKinematics(robot_config)

# create our sim_interface
DT = 0.001
OFFSCREEN_RENDERING=True
sim_interface = AbrMujoco(robot_config, dt=DT, visualize=True, create_offscreen_rendercontext=OFFSCREEN_RENDERING)
sim_interface.connect()
sim_interface.init_viewer()
sim_interface.send_target_angles(robot_config.START_ANGLES)
feedback = sim_interface.get_feedback()
count = 0

ee_id = sim_interface.model.name2id('EE', 'body')
target_id = sim_interface.sim.model.name2id('target', 'body')

def tick():
    global sim_interface, robot_config, path_planner, count, n_timesteps, target_id, ee_id
    #print(f'count {count} n_timesteps {n_timesteps}')
    if count % n_timesteps == 0:
        feedback = sim_interface.get_feedback()
        target_xyz = np.array(
            [
                np.random.random() * 0.5 - 0.25,
                np.random.random() * 0.5 - 0.25,
                np.random.random() * 0.5 + 0.5,
            ]
        )
        R = sim_interface.get_rotation_matrix_by_id(ee_id).reshape(3,3)
        target_orientation = transformations.euler_from_matrix(R, "sxyz")
        # update the position of the target
        sim_interface.set_mocap_xyz_by_id(target_id, target_xyz)

        # can use 3 different methods to calculate inverse kinematics
        # see inverse_kinematics.py file for details
        path_planner.generate_path(
            position=feedback["q"],
            target_position=np.hstack([target_xyz, target_orientation]),
            method=3,
            dt=0.005,
            n_timesteps=n_timesteps,
            plot=False,
        )

    # returns desired [position, velocity]
    target = path_planner.next()[0]

    # use position control
    #print("target angles: ", target[: robot_config.N_JOINTS])
    sim_interface.send_target_angles(target[: robot_config.N_JOINTS])

    count += 1
    return sim_interface.tick()

# Open main window
main_window = MainWindow(sim_interface, robot_config)
main_window.exec(tick)
