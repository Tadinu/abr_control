"""
Running operational space control using Mujoco. The controller will
move the end-effector to the target object's orientation.
"""
import sys
import numpy as np

from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.controllers import OSC, Damping
from abr_control.interfaces.mujoco import AbrMujoco
from abr_control.utils import transformations

from main_window import MainWindow

if len(sys.argv) > 1:
    arm_model = sys.argv[1]
else:
    arm_model = "jaco2"
# initialize our robot config for the jaco2
robot_config = arm(arm_model)

# create the Mujoco sim_interface and connect up
OFFSCREEN_RENDERING=True
sim_interface = AbrMujoco(robot_config, dt=0.001, visualize=True, create_offscreen_rendercontext=OFFSCREEN_RENDERING)
sim_interface.connect()
sim_interface.init_viewer()
sim_interface.send_target_angles(robot_config.START_ANGLES)

# damp the movements of the arm
damping = Damping(robot_config, kv=10)
# create opreational space controller
ctrlr = OSC(
    robot_config,
    kp=200,  # position gain
    ko=200,  # orientation gain
    null_controllers=[damping],
    # control (alpha, beta, gamma) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof=[False, False, False, True, True, True],
)

# set up lists for tracking data
ee_angles_track = []
target_angles_track = []

ee_id = sim_interface.model.name2id('EE', 'body')
target_id = sim_interface.model.name2id('target_orientation', 'body')

count = 0
rand_orient = transformations.random_quaternion()
def tick():
    global count, sim_interface, robot_config, ctrlr, rand_orient, ee_angles_track, target_angles_track, ee_id, target_id
    if (count % 1000 == 0):
        # generate a random target orientation
        rand_orient = transformations.random_quaternion()
        print("New target orientation: ", rand_orient)

    # get arm feedback
    feedback = sim_interface.get_feedback()
    hand_xyz = sim_interface.get_xyz_by_id(ee_id)

    # set our target to our ee_xyz since we are only focussing on orinetation
    sim_interface.set_mocap_xyz_by_id(target_id, hand_xyz)
    sim_interface.set_mocap_orientation_by_id(target_id, rand_orient)

    target_quat = sim_interface.get_mocap_orientation_by_id(target_id)
    target_euler_angles = transformations.euler_from_quaternion(target_quat, "rxyz")
    target = np.hstack(
        [
            sim_interface.get_xyz_by_id(target_id),
            target_euler_angles
        ]
    )

    u = ctrlr.generate(
        q=feedback["q"],
        dq=feedback["dq"],
        target=target,
    )

    # add gripper forces
    u = np.hstack((u, np.zeros(robot_config.N_GRIPPER_JOINTS)))

    # apply the control signal, step the sim forward
    sim_interface.send_forces(u)

    # track data
    ee_matrix = sim_interface.get_rotation_matrix_by_id(ee_id).reshape(3,3)
    ee_angles = transformations.euler_from_matrix(ee_matrix, axes="rxyz")

    ee_angles_track.append(ee_angles)
    target_angles_track.append(target_euler_angles)
    count += 1
    return sim_interface.tick()

# Open main window
try:
    main_window = MainWindow(sim_interface, robot_config)
    main_window.exec(tick)

finally:
    ee_angles_track = np.array(ee_angles_track)
    target_angles_track = np.array(target_angles_track)

    if ee_angles_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(ee_angles_track)
        plt.gca().set_prop_cycle(None)
        plt.plot(target_angles_track, "--")
        plt.ylabel("3D orientation (rad)")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()